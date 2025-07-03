from collections import deque
import numpy as np
import threading
import time
from ...src.phy.eda_signal_processing import EDA
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor
from ...src.registry.plot_registry import PlotRegistry
class EDANode:
    """
    Node for processing EDA (Electrodermal Activity) signa            # Debug print for processed signal data structure
            print(f"[EDA_PROCESSED_DEBUG] Signal data structure: {len(processed_signal_data['t'])} timestamps, {len(processed_signal_data['v'])} main signal points, {len(processed_signal_data['tonic_norm'])} tonic points", flush=True)
            print(f"[EDA_PROCESSED_DEBUG] Time range: {processed_signal_data['t'][0]:.2f}s - {processed_signal_data['t'][-1]:.2f}s ({processed_signal_data['t'][-1] - processed_signal_data['t'][0]:.2f}s window)", flush=True)
            print(f"[EDA_PROCESSED_DEBUG] Added {len(processed_signal_data.get('peaks', []))} peaks to signal data for direct visualization", flush=True)
            
            # Debug print for peak metadata
            print(f"[EDA_METADATA_DEBUG] Final metadata with {len(metadata['peak_timestamps'])} peak timestamps and {len(metadata.get('peak_values', []))} peak values (show_peaks={metadata['show_peaks']}, marker='{metadata.get('peak_marker', 'x')}')", flush=True)  Extracts tonic and phasic components, provides visualization-ready data,
    and allows selection of which components to output.
    Robust and registry-ready for MetricsView integration.
    """

    # Track background processing threads
    _processing_threads = {}
    _stop_flags = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),                
                "show_peaks": ("BOOLEAN", {"default": True}),  # Default to showing peaks
                "output_signal_id": ("STRING", {"default": "EDA_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("SCL", "SCR_Frequency", "Signal_ID")
    FUNCTION = "process_eda"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def __init__(self):
        self.eda = EDA()  # Instance for stateful processing
        self._last_scl = 0.0
        self._last_scr_frequency = 0.0

    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        registry = PlotRegistry.get_instance()
        metrics_registry = SignalRegistry.get_instance()
        fs = 1000
        nyquist_fs = fs / 2
        viz_window_sec = 40
        viz_buffer_size = fs * viz_window_sec
        
        # Feature buffer should preserve 1 minute (60 seconds) of data, accounting for decimation
        feature_window_sec = 60  # 1 minute for EDA features
        max_frequency_interest = 10  # EDA rarely above 10 Hz
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        
        # Feature buffer size accounts for downsampling - preserves 60 seconds of decimated data
        effective_fs_after_decimation = fs // decimation_factor if use_decimation else fs
        feature_buffer_size = effective_fs_after_decimation * feature_window_sec + effective_fs_after_decimation  # +1 second safety margin
        
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        
        # EDA metrics buffer: preserve 2 minutes of metric data at ~1Hz rate  
        eda_metrics_window_sec = 120.0  # 2 minutes for EDA metrics
        eda_metrics_update_rate = 1.0   # Approximately 1 Hz for metrics
        eda_metrics_buffer_size = int(eda_metrics_window_sec * eda_metrics_update_rate)
        metrics_deque = deque(maxlen=eda_metrics_buffer_size)
        
        last_registry_data_hash = None
        last_process_time = time.time()  # Initialize last process time
        processing_interval = 0.2  # 5 Hz processing rate (reduced for better performance)
        start_time = None
        while not stop_flag[0]:
            current_time = time.time()
            if current_time - last_process_time < processing_interval:
                continue  # Skip this iteration, don't sleep
            last_process_time = current_time
            signal_data = registry.get_signal(input_signal_id)
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                continue  # Skip this iteration, don't sleep
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            if len(timestamps) < 2 or len(values) < 2:  # Need at least 2 points for processing and visualization
                continue  # Skip this iteration, don't sleep
            # --- Retrieve start_time from metadata if available (for peak logic only) ---
            meta = None
            if hasattr(registry, 'get_signal_metadata'):
                meta = registry.get_signal_metadata(input_signal_id)
            elif isinstance(signal_data, dict) and 'meta' in signal_data:
                meta = signal_data['meta']
            meta_start_time = meta['start_time'] if meta and 'start_time' in meta else None
            if start_time is None and len(feature_timestamps_deque) > 0:
                start_time = time.time() - feature_timestamps_deque[0]

            # Process all new data - simplified logic for rotating deques
            # When deques are full, they automatically rotate old data out
            if use_decimation:
                if len(timestamps) > 1:
                    decimated_values = NumpySignalProcessor.robust_decimate(values, decimation_factor)
                    decimated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=len(decimated_values)) if len(timestamps) > 1 else timestamps
                    
                    # For rotating deques, we need to check if we have genuinely new data
                    # Compare with the last few timestamps to avoid duplicate processing
                    if len(viz_timestamps_deque) > 0:
                        last_processed_time = viz_timestamps_deque[-1]
                        # Only add data that's newer than what we last processed
                        new_mask = decimated_timestamps > last_processed_time
                        if np.any(new_mask):
                            new_decimated_timestamps = decimated_timestamps[new_mask]
                            new_decimated_values = decimated_values[new_mask]
                            viz_timestamps_deque.extend(new_decimated_timestamps)
                            viz_values_deque.extend(new_decimated_values)
                            feature_timestamps_deque.extend(new_decimated_timestamps)
                            feature_values_deque.extend(new_decimated_values)
                    else:
                        # First time processing - add all data
                        viz_timestamps_deque.extend(decimated_timestamps)
                        viz_values_deque.extend(decimated_values)
                        feature_timestamps_deque.extend(decimated_timestamps)
                        feature_values_deque.extend(decimated_values)
            else:
                # For rotating deques, check for new data based on timestamps
                if len(viz_timestamps_deque) > 0:
                    last_processed_time = viz_timestamps_deque[-1]
                    # Only add data that's newer than what we last processed
                    new_mask = timestamps > last_processed_time
                    if np.any(new_mask):
                        new_timestamps = timestamps[new_mask]
                        new_values = values[new_mask]
                        viz_timestamps_deque.extend(new_timestamps)
                        viz_values_deque.extend(new_values)
                        feature_timestamps_deque.extend(new_timestamps)
                        feature_values_deque.extend(new_values)
                else:
                    # First time processing - add all data
                    viz_timestamps_deque.extend(timestamps)
                    viz_values_deque.extend(values)
                    feature_timestamps_deque.extend(timestamps)
                    feature_values_deque.extend(values)

            # Convert deques to numpy arrays for processing
            viz_timestamps = np.array(viz_timestamps_deque)
            viz_values = np.array(viz_values_deque)
            feature_timestamps = np.array(feature_timestamps_deque)
            feature_values = np.array(feature_values_deque)

            # Adjust effective sampling rate if decimation was applied
            effective_fs = fs
            if use_decimation:
                effective_fs = fs / decimation_factor

            # Windowing for visualization
            if len(viz_timestamps) > 0:
                window_max = viz_timestamps[-1]
                window_min = max(window_max - viz_window_sec, viz_timestamps[0])
                window_mask = (viz_timestamps >= window_min) & (viz_timestamps <= window_max)
                viz_timestamps_window = viz_timestamps[window_mask]
                viz_values_window = viz_values[window_mask]
            else:
                viz_timestamps_window = viz_timestamps
                viz_values_window = viz_values

            # Tonic/phasic extraction
            feature_values = self.eda.convert_adc_to_eda(feature_values)  # Convert to microsiemens
            tonic_raw, phasic_raw = self.eda.extract_tonic_phasic(feature_values, fs=effective_fs) # Convert to microsiemens
            # Sanitize raw components to ensure they are 1D numpy arrays
            def sanitize_component(raw_comp):
                if raw_comp is None: return np.array([])
                # Attempt to convert to numpy array
                try:
                    arr = np.asarray(raw_comp)
                except Exception: # If conversion fails, return empty array
                    return np.array([])
                
                if arr.ndim == 0: # Handle scalar by converting to 1-element 1D array
                    return np.array([arr.item()]) 
                return arr.flatten() # Ensure 1D, or keep as 1D if already

            tonic_raw = sanitize_component(tonic_raw)
            phasic_raw = sanitize_component(phasic_raw)

            # Ensure viz_timestamps_window is a usable 1D numpy array for shaping tonic_viz/phasic_viz
            if not isinstance(viz_timestamps_window, np.ndarray) or viz_timestamps_window.ndim != 1:
                viz_timestamps_window = np.array([]) # Fallback to empty array

            # Initialize viz versions based on viz_timestamps_window
            if viz_timestamps_window.size == 0:
                tonic_viz = np.array([])
                phasic_viz = np.array([])
            else:
                tonic_viz = np.zeros_like(viz_timestamps_window)
                phasic_viz = np.zeros_like(viz_timestamps_window)

                # Attempt to populate tonic_viz and phasic_viz by interpolation
                # This requires feature_timestamps, tonic_raw, and phasic_raw to be valid 1D arrays
                # and raw components to align with feature_timestamps.
                if (tonic_raw.size > 0 and phasic_raw.size > 0 and
                    isinstance(feature_timestamps, np.ndarray) and feature_timestamps.ndim == 1 and feature_timestamps.size > 0 and
                    tonic_raw.ndim == 1 and phasic_raw.ndim == 1 and
                    len(tonic_raw) == len(feature_timestamps) and 
                    len(phasic_raw) == len(feature_timestamps)):
                    try:
                        # Interpolate to align with viz_timestamps_window
                        tonic_viz_interp = np.interp(viz_timestamps_window, feature_timestamps, tonic_raw)
                        phasic_viz_interp = np.interp(viz_timestamps_window, feature_timestamps, phasic_raw)
                        
                        # Ensure interpolation produced arrays of the correct shape
                        if tonic_viz_interp.shape == viz_timestamps_window.shape:
                             tonic_viz = tonic_viz_interp
                        if phasic_viz_interp.shape == viz_timestamps_window.shape:
                             phasic_viz = phasic_viz_interp
                        # If interpolation results in wrong shape, they remain as pre-initialized zeros_like
                    except Exception: 
                        # If interpolation fails, tonic_viz/phasic_viz remain as pre-initialized zeros_like
                        pass
            # tonic_viz and phasic_viz are now guaranteed to be 1D numpy arrays,
            # either empty or matching viz_timestamps_window length (filled with zeros or interpolated values).

            # --- Ensure no NaN/Inf in tonic_viz and phasic_viz before further processing or registration ---
            if tonic_viz.size > 0:
                tonic_viz = np.nan_to_num(tonic_viz, nan=0.0, posinf=np.finfo(tonic_viz.dtype).max, neginf=np.finfo(tonic_viz.dtype).min)
            if phasic_viz.size > 0:
                phasic_viz = np.nan_to_num(phasic_viz, nan=0.0, posinf=np.finfo(phasic_viz.dtype).max, neginf=np.finfo(phasic_viz.dtype).min)

            # --- Enhanced normalization for visualization ---
            # Apply visual enhancement to phasic component for better peak visibility
            if phasic_viz.size > 0:
                # Further enhance phasic visualization - normalize to better highlight peaks
                noise_floor = 0.01  # μS, minimum threshold to consider signal vs noise
                phasic_viz = np.where(phasic_viz > noise_floor, phasic_viz, 0)
                
                # Make sure there are no negative values
                phasic_viz = np.maximum(phasic_viz, 0)
                
                # Scale to ensure peaks are visible
                phasic_max = np.max(phasic_viz) if phasic_viz.size > 0 else 0
                
                print(f"[EDA_VISUAL_DEBUG] Phasic signal max: {phasic_max:.4f} μS", flush=True)
                
                # Minimum desired maximum value in μS
                min_desired_max = 0.5
                
                # Handle case where phasic signal is too small for good visualization
                if phasic_max <= 0.001:  # Essentially zero
                    print(f"[EDA_VISUAL_DEBUG] All phasic values are near zero, adding minimal baseline for visibility", flush=True)
                    # Add a small baseline to ensure visibility
                    phasic_viz = phasic_viz + 0.05
                elif phasic_max > 0 and phasic_max < min_desired_max:
                    # Ensure the maximum amplitude is visible, for better scale
                    scaling_factor = min_desired_max / phasic_max
                    phasic_viz = phasic_viz * scaling_factor
                    print(f"[EDA_VISUAL_DEBUG] Enhanced phasic component visualization by factor {scaling_factor:.2f}", flush=True)
            
            # SCR event (peak) detection and mapping to window (like ECG R-peak logic)
            scr_event_indices = np.array([], dtype=int)
            if len(phasic_viz) > 0:
                # Calculate dynamic threshold based on phasic signal baseline
                phasic_baseline = np.percentile(phasic_viz, 25)  # Use 25th percentile as baseline
                phasic_std = np.std(phasic_viz)
                
                # Dynamic threshold: baseline + 2 standard deviations, with minimum of 0.1
                dynamic_threshold = max(0.1, phasic_baseline + 2 * phasic_std)
                
                # Explicitly pass the dynamic threshold for more robust peak detection
                result = self.eda.detect_events(phasic_viz, effective_fs, threshold=dynamic_threshold)
                if isinstance(result, tuple) and len(result) == 2:
                    scr_event_indices_raw, _ = result
                else:
                    scr_event_indices_raw = result
                if scr_event_indices_raw is None:
                    _scr_event_indices_temp = np.array([], dtype=int)
                else:
                    _scr_event_indices_temp = np.asarray(scr_event_indices_raw)
                    if _scr_event_indices_temp.ndim == 0:
                        try:
                            _scr_event_indices_temp = np.array([int(_scr_event_indices_temp.item())], dtype=int)
                        except (ValueError, TypeError):
                            _scr_event_indices_temp = np.array([], dtype=int)
                    else:
                        try:
                            _scr_event_indices_temp = _scr_event_indices_temp.flatten().astype(int)
                        except (ValueError, TypeError):
                            _scr_event_indices_temp = np.array([], dtype=int)
                if len(viz_timestamps_window) > 0 and _scr_event_indices_temp.size > 0:
                    scr_event_indices = _scr_event_indices_temp[(_scr_event_indices_temp >= 0) & (_scr_event_indices_temp < len(viz_timestamps_window))]
                else:
                    scr_event_indices = np.array([], dtype=int)
            else:
                scr_event_indices = np.array([], dtype=int)
            # --- Vectorized mapping of SCR event indices to timestamps in window (like RR node) ---
            peak_timestamps_in_window = []
            peak_phasic_values_in_window = []
            
            # Always process peaks for metadata regardless of show_peaks setting
            if len(scr_event_indices) > 0 and len(viz_timestamps_window) > 0:
                # Safety check: ensure indices are within bounds
                valid_indices = (scr_event_indices >= 0) & (scr_event_indices < len(viz_timestamps_window))
                if not np.all(valid_indices):
                    print(f"[EDA_PEAKS_DEBUG] WARNING: {np.sum(~valid_indices)} out-of-bounds peak indices detected. Filtering them out.", flush=True)
                    scr_event_indices = scr_event_indices[valid_indices]
                
                # Skip if no valid indices remain
                if len(scr_event_indices) == 0:
                    print(f"[EDA_PEAKS_DEBUG] No valid peak indices remain after bounds checking.", flush=True)
                else:
                    # Get the event times and phasic values at the indices
                    scr_event_times = viz_timestamps_window[scr_event_indices]
                    peak_phasic_values = phasic_viz[scr_event_indices]
                
                # Get window boundaries
                window_min = viz_timestamps_window[0]
                window_max = viz_timestamps_window[-1]
                
                # Print detailed information about all detected peaks
                print(f"[EDA_PEAKS_DETAILED] All {len(scr_event_indices)} detected peaks:", flush=True)
                for i, (idx, peak_time, val) in enumerate(zip(scr_event_indices, scr_event_times, peak_phasic_values)):
                    print(f"  Peak {i+1}: index={idx}, time={peak_time:.2f}s, value={val:.4f}μS", flush=True)
                
                # Since these indices are already filtered to be within viz_timestamps_window range,
                # we don't need to filter them again - these peaks are already in the window
                peak_timestamps_in_window = scr_event_times.tolist()
                peak_phasic_values_in_window = peak_phasic_values.tolist()
                
                print(f"[EDA_PEAKS_DEBUG] All {len(peak_timestamps_in_window)} peaks are within window time range: {window_min:.2f}s - {window_max:.2f}s", flush=True)
                
                print(f"[EDA_PEAKS_DEBUG] Found {len(peak_timestamps_in_window)} of {len(scr_event_indices)} EDA peaks in window time range: {window_min:.2f}s - {window_max:.2f}s", flush=True)
                
                # Safety check: limit number of peaks to prevent visualization issues
                max_viz_peaks = 20  # Reasonable limit for visualization
                if len(peak_timestamps_in_window) > max_viz_peaks:
                    print(f"[EDA_PEAKS_DEBUG] WARNING: Too many peaks for visualization ({len(peak_timestamps_in_window)}). Limiting to {max_viz_peaks} highest amplitude peaks.", flush=True)
                    # Get peak amplitudes and sort
                    if len(peak_phasic_values_in_window) == len(peak_timestamps_in_window):
                        peak_amps = np.array(peak_phasic_values_in_window)
                        top_indices = np.argsort(peak_amps)[-max_viz_peaks:]
                        peak_timestamps_in_window = [peak_timestamps_in_window[i] for i in top_indices]
                        peak_phasic_values_in_window = [peak_phasic_values_in_window[i] for i in top_indices]
                        print(f"[EDA_PEAKS_DEBUG] Filtered to {len(peak_timestamps_in_window)} highest amplitude peaks for visualization", flush=True)
                
                if len(peak_timestamps_in_window) > 0:
                    # Print both timestamps and y-values of peaks
                    peak_info = []
                    for i in range(min(5, len(peak_timestamps_in_window))):
                        if i < len(peak_timestamps_in_window) and i < len(peak_phasic_values_in_window):
                            peak_info.append(f"({peak_timestamps_in_window[i]:.2f}s, {peak_phasic_values_in_window[i]:.4f}μS)")
                    print(f"[EDA_PEAKS_DEBUG] Peak timestamps and y-values: {peak_info}{' ...' if len(peak_timestamps_in_window) > 5 else ''}", flush=True)
                    
                    # We're forcing peaks to be shown in visualization now
                    print(f"[EDA_PEAKS_DEBUG] These peaks will be shown in visualization (overriding user setting, original show_peaks={show_peaks})", flush=True)
                    
                    # Log peak indices for visualization debugging
                    viz_indices = []
                    for peak_time in peak_timestamps_in_window:
                        # Find closest index in viz_timestamps_window to each peak time
                        if len(viz_timestamps_window) > 0:
                            idx = np.argmin(np.abs(viz_timestamps_window - peak_time))
                            viz_indices.append(int(idx))
                    if viz_indices:
                        print(f"[EDA_PEAKS_DEBUG] Peak visualization indices: {viz_indices}", flush=True)
            else:
                print(f"[EDA_PEAKS_DEBUG] No EDA peaks detected (indices: {len(scr_event_indices)}, window len: {len(viz_timestamps_window)})", flush=True)
            scr_frequency = (len(peak_timestamps_in_window) / viz_window_sec) * 60 if viz_window_sec > 0 and len(peak_timestamps_in_window) > 0 else 0.0
            scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0

            # Update instance variables for return values
            self._last_scl = scl
            self._last_scr_frequency = scr_frequency

            # Prepare peak_values for metadata if they exist
            peak_phasic_values_for_metadata = peak_phasic_values_in_window if 'peak_phasic_values_in_window' in locals() and len(peak_timestamps_in_window) > 0 else []
            
            # Calculate peak_indices before metadata creation
            peak_indices = []
            for peak_time in peak_timestamps_in_window:
                # Find the index in the timestamps array closest to each peak time
                if len(viz_timestamps_window) > 0:
                    idx = np.argmin(np.abs(viz_timestamps_window - peak_time))
                    peak_indices.append(int(idx))
            
            # Always show peaks in visualization regardless of user setting
            # The show_peaks parameter is only used for interactive debugging/visualization
            metadata = {
                "id": output_signal_id,
                "type": "eda_processed",  # Match ECG's naming pattern (ecg_processed)
                "viz_subtype": "eda",
                "scl": scl,
                "sck": sck,
                "scr_frequency": scr_frequency,
                "peak_timestamps": peak_timestamps_in_window,  # Timestamps for proper peak display
                "peak_values": peak_phasic_values_for_metadata,  # Y-values of peaks for debugging
                "show_peaks": True,  # Always show peaks in visualization regardless of user setting
                "peak_marker": "x",  # Add peak marker style like ECG node
                "color": "#55FF55",  # Green color for EDA peaks
                "decimation_factor": decimation_factor if use_decimation else 1,
                "effective_fs": effective_fs,
                # Add _original_data_dict for EDA visualization system
                "_original_data_dict": {
                    "phasic_norm": phasic_viz.tolist(),
                    "tonic_norm": tonic_viz.tolist(),
                    "peak_indices": peak_indices
                }
            }
            
            # Make sure peak data is always present in metadata
            if "peak_timestamps" not in metadata:
                metadata["peak_timestamps"] = []
            if "peak_values" not in metadata:
                metadata["peak_values"] = []
            
            # Always ensure we have enough data for visualization - REQUIRED for registry
            if len(viz_timestamps_window) < 2 or len(tonic_viz) < 2 or len(phasic_viz) < 2:
                print(f"[EDA_DEBUG] WARNING: Not enough data points for visualization. Timestamps: {len(viz_timestamps_window)}, Tonic: {len(tonic_viz)}, Phasic: {len(phasic_viz)}", flush=True)
                
                # Create a minimal synthetic dataset to avoid registry errors - CRITICAL for visualization
                print(f"[EDA_DEBUG] Creating minimal synthetic dataset to avoid visualization errors", flush=True)
                # Create at least 2 points of data
                current_time = time.time()
                viz_timestamps_window = np.array([current_time-1, current_time])
                phasic_viz = np.array([0.01, 0.01])  # Small non-zero values
                tonic_viz = np.array([1.0, 1.0])     # Baseline value
            
            # Check if all arrays have the same length before creating processed data
            if len(viz_timestamps_window) != len(tonic_viz) or len(viz_timestamps_window) != len(phasic_viz):
                print(f"[EDA_DEBUG] Length mismatch: timestamps={len(viz_timestamps_window)}, tonic={len(tonic_viz)}, phasic={len(phasic_viz)}", flush=True)
                # Truncate to shortest length for consistency
                min_len = min(len(viz_timestamps_window), len(tonic_viz), len(phasic_viz))
                viz_timestamps_window = viz_timestamps_window[:min_len]
                tonic_viz = tonic_viz[:min_len]
                phasic_viz = phasic_viz[:min_len]
                
            # Match ECG data structure for proper peak visualization
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),   # Time values
                "v": phasic_viz.tolist(),              # Main signal for visualization and peak detection
                "tonic_norm": tonic_viz.tolist(),      # Additional tonic component
                "phasic_norm": phasic_viz.tolist(),    # Keep for backward compatibility
                "id": output_signal_id,
                "peaks": peak_timestamps_in_window,    # Add peaks directly to signal data
                "peak_values": peak_phasic_values_in_window  # Include peak y-values
            }
            
            # Make sure peaks are always available, even if empty
            if "peaks" not in processed_signal_data:
                processed_signal_data["peaks"] = []
            if "peak_values" not in processed_signal_data:
                processed_signal_data["peak_values"] = []
            
            # Add peak_indices to processed_signal_data (already calculated earlier)
            processed_signal_data["peak_indices"] = peak_indices
            
            # Debug print for processed signal data structure
            print(f"[EDA_PROCESSED_DEBUG] Signal data structure: {len(processed_signal_data['t'])} timestamps, {len(processed_signal_data['v'])} primary signal points (phasic), {len(processed_signal_data['tonic_norm'])} tonic points", flush=True)
            if len(processed_signal_data['t']) > 1:
                print(f"[EDA_PROCESSED_DEBUG] Time range: {processed_signal_data['t'][0]:.2f}s - {processed_signal_data['t'][-1]:.2f}s ({processed_signal_data['t'][-1] - processed_signal_data['t'][0]:.2f}s window)", flush=True)
                print(f"[EDA_PROCESSED_DEBUG] Data sufficient for visualization (>= 2 points)", flush=True)
            else:
                print(f"[EDA_PROCESSED_DEBUG] WARNING: Insufficient data for visualization!", flush=True)
            
            # Debug print for peak metadata
            print(f"[EDA_METADATA_DEBUG] Final metadata with {len(metadata['peak_timestamps'])} peak timestamps and {len(metadata.get('peak_values', []))} peak values (show_peaks=True, overriding user setting {show_peaks})", flush=True)
            if len(metadata['peak_timestamps']) > 0:
                print(f"[EDA_METADATA_DEBUG] First few peak timestamps in metadata: {[f'{ts:.2f}s' for ts in metadata['peak_timestamps'][:5]]}", flush=True)
                if 'peak_values' in metadata and len(metadata['peak_values']) > 0:
                    print(f"[EDA_METADATA_DEBUG] First few peak values in metadata: {[f'{val:.4f}μS' for val in metadata['peak_values'][:5]]}", flush=True)
            
            # Debug the EDA visualization data structure
            original_data = metadata.get('_original_data_dict', {})
            print(f"[EDA_VIZ_DEBUG] _original_data_dict structure: phasic_norm len={len(original_data.get('phasic_norm', []))}, tonic_norm len={len(original_data.get('tonic_norm', []))}, peak_indices len={len(original_data.get('peak_indices', []))}", flush=True)
            if len(original_data.get('phasic_norm', [])) > 0:
                phasic_norm = original_data['phasic_norm']
                print(f"[EDA_VIZ_DEBUG] Phasic component for visualization: min={min(phasic_norm):.4f}μS, max={max(phasic_norm):.4f}μS, mean={sum(phasic_norm)/len(phasic_norm):.4f}μS", flush=True)
            if len(original_data.get('tonic_norm', [])) > 0:
                tonic_norm = original_data['tonic_norm']
                print(f"[EDA_VIZ_DEBUG] Tonic component for visualization: min={min(tonic_norm):.4f}μS, max={max(tonic_norm):.4f}μS, mean={sum(tonic_norm)/len(tonic_norm):.4f}μS", flush=True)
            
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            # Metrics registry pattern (similar to ECG node)
            last_timestamp = float(feature_timestamps[-1]) if len(feature_timestamps) > 0 else time.time()
            metrics_deque.append((last_timestamp, scl, sck, scr_frequency))
            metrics_t = [x[0] for x in metrics_deque]
            metrics_scl = [x[1] for x in metrics_deque]
            metrics_sck = [x[2] for x in metrics_deque]
            metrics_scr_freq = [x[3] for x in metrics_deque]
            scl_metrics_data = {
                't': metrics_t,
                'v': metrics_scl,
                'last': scl
            }
            metrics_registry.register_signal('SCL_METRIC', scl_metrics_data, {
                'id': 'SCL_METRIC',
                'type': 'eda_metrics',
                'label': 'Skin Conductance Level (SCL)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            sck_metrics_data = {
                't': metrics_t,
                'v': metrics_sck,
                'last': sck
            }
            metrics_registry.register_signal('SCK_METRIC', sck_metrics_data, {
                'id': 'SCK_METRIC',
                'type': 'eda_metrics',
                'label': 'Skin Conductance Response (SCK)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            scr_freq_metrics_data = {
                't': metrics_t,
                'v': metrics_scr_freq,
                'last': scr_frequency
            }
            metrics_registry.register_signal('scr_frequency', scr_freq_metrics_data, {
                'id': 'scr_frequency',
                'type': 'eda_metrics',
                'label': 'SCR Frequency (events/min)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            continue  # Continue processing loop without delay

    def process_eda(self, input_signal_id="", show_peaks=True, output_signal_id="EDA_PROCESSED", enabled=True):
        if not enabled:
            self._last_scl = 0.0
            self._last_scr_frequency = 0.0
            return self._last_scl, self._last_scr_frequency, output_signal_id

        signal_id = output_signal_id
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            return self._last_scl, self._last_scr_frequency, output_signal_id

        if signal_id in self._stop_flags:
            self._stop_flags[signal_id][0] = True

        stop_flag = [False]
        self._stop_flags[signal_id] = stop_flag
        thread = threading.Thread(
            target=self._background_process,
            args=(input_signal_id, show_peaks, stop_flag, output_signal_id),
            daemon=True
        )
        self._processing_threads[signal_id] = thread
        thread.start()

        return self._last_scl, self._last_scr_frequency, output_signal_id

    def __del__(self):
        for stop_flag in self._stop_flags.values():
            stop_flag[0] = True
        self._processing_threads.clear()
        self._stop_flags.clear()

NODE_CLASS_MAPPINGS = {
    "EDANode": EDANode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDANode": "🖐️ EDA Processor"
}
