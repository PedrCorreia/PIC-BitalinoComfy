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
    Node for processing EDA (Electrodermal Activity) signals.
    Extracts tonic and phasic components, provides visualization-ready data,
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
                "input_signal_id": ("STRING", {"default": ""}),                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "EDA_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("SCL", "SCR", "Signal_ID")
    FUNCTION = "process_eda"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def __init__(self):
        self.eda = EDA()  # Instance for stateful processing

    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        registry = PlotRegistry.get_instance()
        metrics_registry = SignalRegistry.get_instance()
        fs = 1000
        nyquist_fs = fs / 2
        viz_window_sec = 20
        viz_buffer_size = fs * viz_window_sec
        feature_buffer_size = viz_buffer_size + fs
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        metrics_buffer_size = 300  # e.g., 5 minutes at 1Hz
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.033
        start_time = None
        metrics_deque = deque(maxlen=metrics_buffer_size)
        max_frequency_interest = 250  # EDA is low freq, but keep for decimation logic
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        while not stop_flag[0]:
            current_time = time.time()
            if current_time - last_process_time < processing_interval:
                time.sleep(0.001)
                continue
            last_process_time = current_time
            signal_data = registry.get_signal(input_signal_id)
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                time.sleep(processing_interval)
                continue
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            if len(timestamps) < 2 or len(values) < 2:
                time.sleep(processing_interval)
                continue
            # --- Retrieve start_time from metadata if available (for peak logic only) ---
            meta = None
            if hasattr(registry, 'get_signal_metadata'):
                meta = registry.get_signal_metadata(input_signal_id)
            elif isinstance(signal_data, dict) and 'meta' in signal_data:
                meta = signal_data['meta']
            meta_start_time = meta['start_time'] if meta and 'start_time' in meta else None
            if start_time is None and len(feature_timestamps_deque) > 0:
                start_time = time.time() - feature_timestamps_deque[0]

            # Only append new data (timestamp-based, like ECG)
            if len(viz_timestamps_deque) > 0 and len(timestamps) > 0:
                last_ts = viz_timestamps_deque[-1]
                new_data_idx = np.searchsorted(timestamps, last_ts, side='right')
                if new_data_idx < len(timestamps):
                    if use_decimation:
                        remaining_points = min(len(timestamps), len(values)) - new_data_idx
                        if remaining_points > 0:
                            max_index = new_data_idx + remaining_points
                            new_timestamps = timestamps[new_data_idx:max_index]
                            new_values = values[new_data_idx:max_index]
                            decimated_values = NumpySignalProcessor.robust_decimate(new_values, decimation_factor)
                            decimated_timestamps = np.linspace(new_timestamps[0], new_timestamps[-1], num=len(decimated_values)) if len(new_timestamps) > 1 else new_timestamps
                            viz_timestamps_deque.extend(decimated_timestamps)
                            viz_values_deque.extend(decimated_values)
                            feature_timestamps_deque.extend(decimated_timestamps)
                            feature_values_deque.extend(decimated_values)
                    else:
                        viz_timestamps_deque.extend(timestamps[new_data_idx:])
                        viz_values_deque.extend(values[new_data_idx:])
                        feature_timestamps_deque.extend(timestamps[new_data_idx:])
                        feature_values_deque.extend(values[new_data_idx:])
            else:
                if use_decimation:
                    if len(timestamps) > 1:
                        decimated_values = NumpySignalProcessor.robust_decimate(values, decimation_factor)
                        decimated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=len(decimated_values))
                        viz_timestamps_deque.extend(decimated_timestamps)
                        viz_values_deque.extend(decimated_values)
                        feature_timestamps_deque.extend(decimated_timestamps)
                        feature_values_deque.extend(decimated_values)
                else:
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
            tonic_raw, phasic_raw = self.eda.extract_tonic_phasic(feature_values, fs=effective_fs)

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

            # --- Amplitude correction: check absolute change before/after filtering ---
            raw_window = viz_values_window
            filtered_tonic = tonic_viz
            filtered_phasic = phasic_viz

           # SCL/SCK calculation and rolling history
            scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0
            
            # Store for node output
            setattr(self, '_last_scl', scl)
            setattr(self, '_last_sck', sck)
            # SCR event (peak) detection and mapping to window (like ECG R-peak logic)
            scr_event_indices = np.array([], dtype=int) # Initialize
            if len(phasic_viz) > 0: # Only detect events if phasic_viz is not empty
                result = self.eda.detect_events(phasic_viz, effective_fs)
                # Handle tuple return (validated_events, envelope) or just events
                if isinstance(result, tuple) and len(result) == 2:
                    scr_event_indices_raw, _ = result  # Extract just the indices, ignore envelope
                else:
                    scr_event_indices_raw = result
                
                # Ensure scr_event_indices_raw becomes a 1D integer numpy array
                if scr_event_indices_raw is None:
                    _scr_event_indices_temp = np.array([], dtype=int)
                else:
                    _scr_event_indices_temp = np.asarray(scr_event_indices_raw)
                    if _scr_event_indices_temp.ndim == 0: # Scalar value
                        try:
                            # Attempt to convert scalar to a single-element 1D int array
                            _scr_event_indices_temp = np.array([int(_scr_event_indices_temp.item())], dtype=int)
                        except (ValueError, TypeError): # If not convertible to int
                            _scr_event_indices_temp = np.array([], dtype=int)
                    else: # Already an array (or list-like)
                        try:
                            # Flatten to 1D and ensure integer type
                            _scr_event_indices_temp = _scr_event_indices_temp.flatten().astype(int)
                        except (ValueError, TypeError): # If elements not convertible to int
                            _scr_event_indices_temp = np.array([], dtype=int)
                
                # _scr_event_indices_temp is now a 1D int array (possibly empty)
                
                # Only keep indices that are valid for the current window
                if len(viz_timestamps_window) > 0 and _scr_event_indices_temp.size > 0:
                    scr_event_indices = _scr_event_indices_temp[(_scr_event_indices_temp >= 0) & (_scr_event_indices_temp < len(viz_timestamps_window))]
                else:
                    scr_event_indices = np.array([], dtype=int) # No window or no raw indices, no valid indices
            else: # No phasic data, no events
                scr_event_indices = np.array([], dtype=int)

            # scr_event_indices is now guaranteed to be a 1D numpy array (int type, possibly empty)

            # --- Vectorized mapping of SCR event indices to timestamps in window (like RR node) ---
            peak_timestamps_in_window = []
            if show_peaks and len(scr_event_indices) > 0 and len(viz_timestamps_window) > 0:
                scr_event_times = viz_timestamps_window[scr_event_indices]
                window_min = viz_timestamps_window[0]
                window_max = viz_timestamps_window[-1]
                in_window_mask = (scr_event_times >= window_min) & (scr_event_times <= window_max)
                peak_timestamps_in_window = scr_event_times[in_window_mask].tolist()
            else:
                peak_timestamps_in_window = [] # Ensure it's a list
            scr_frequency = (len(peak_timestamps_in_window) / viz_window_sec) * 60 if viz_window_sec > 0 and len(peak_timestamps_in_window) > 0 else 0.0  # events per minute

            # --- RR-style peak metadata for visualization ---
            peak_marker = "o"
            peak_color = "#FF55AA"  # Choose a distinct color for EDA peaks

            # Ensure all components are 1D numpy arrays before use in hash, metadata, or processed_signal_data
            viz_timestamps_window = sanitize_component(viz_timestamps_window)
            tonic_viz = sanitize_component(tonic_viz)
            phasic_viz = sanitize_component(phasic_viz)
            # scr_event_indices is usually robustly handled to be 1D int array.
            # This call ensures it (e.g. if a 0D array sneaked in). Dtype is preserved by sanitize_component for existing arrays.
            scr_event_indices = sanitize_component(scr_event_indices) 

            # Hash for registry update optimization
            data_hash = hash((
                tuple(viz_timestamps_window[-5:]),
                tuple(phasic_viz[-5:]),
                tuple(tonic_viz[-5:]),  # Changed from phasic_viz to tonic_viz
                tuple(scr_event_indices[-3:].tolist() if len(scr_event_indices) > 0 else []),
                scl, sck
            ))
            if data_hash == last_registry_data_hash:
                time.sleep(0.01)
                continue
            last_registry_data_hash = data_hash
            
            # Metadata and processed signal for registry
            metadata = {
                "id": output_signal_id,
                "type": "processed",  # Changed from "eda_processed"
                "viz_subtype": "eda", # Added for specific EDA handling
                "scl": scl,
                "sck": sck,
                "scr_frequency": scr_frequency,
                "peak_timestamps": peak_timestamps_in_window # Kept for potential other uses
                # "over": True # Removed, as mode='eda' will handle it
            }
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),
                "tonic_norm": tonic_viz.tolist(),    # Renamed from tonic, assumed normalized
                "phasic_norm": phasic_viz.tolist(),   # Renamed from phasic, assumed normalized
                "peak_indices": scr_event_indices.tolist(), # Added peak indices relative to windowed data
                # "timestamps": viz_timestamps_window.tolist(), # Removed duplicate key, "t" is used
                "id": output_signal_id
            }
            # CRITICAL DEBUG PRINT for EDA signal registration
            # print(f"[EDA_NODE_DEBUG] Attempting to register signal: ID={output_signal_id}, TypeInMeta={metadata.get('type')}, VizSubtypeInMeta={metadata.get('viz_subtype')}", flush=True)
            # print(f"[EDA_NODE_DEBUG] Processed signal data keys: {list(processed_signal_data.keys())}", flush=True)
            # print(f"[EDA_NODE_DEBUG] Metadata keys: {list(metadata.keys())}", flush=True)
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            
            # Metrics registry pattern (similar to ECG node)
            # Get the last timestamp from feature_timestamps (if available)
            last_timestamp = float(feature_timestamps[-1]) if len(feature_timestamps) > 0 else time.time()
            
            # Append to metrics deque (timestamp, scl, sck, scr_frequency)
            metrics_deque.append((last_timestamp, scl, sck, scr_frequency))
            
            # Prepare metrics signals for registry (SCL, SCK, SCR frequency as separate metrics)
            metrics_t = [x[0] for x in metrics_deque]
            metrics_scl = [x[1] for x in metrics_deque]
            metrics_sck = [x[2] for x in metrics_deque]
            metrics_scr_freq = [x[3] for x in metrics_deque]
            
            # Register SCL metric as a time series for MetricsView compatibility
            scl_metrics_data = {
                't': metrics_t,
                'v': metrics_scl
            }
            metrics_registry.register_signal('SCL_METRIC', scl_metrics_data, { # Changed from scl
                'id': 'SCL_METRIC', # Changed from scl
                'type': 'eda_metrics',
                'label': 'Skin Conductance Level (SCL)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            
            # Register SCK metric (phasic component)
            sck_metrics_data = {
                't': metrics_t,
                'v': metrics_sck
            }
            metrics_registry.register_signal('SCK_METRIC', sck_metrics_data, { # Changed from sck
                'id': 'SCK_METRIC', # Changed from sck
                'type': 'eda_metrics',
                'label': 'Skin Conductance Response (SCK)', # Changed from Skin Conductance Response (SCK)
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            
            # Register SCR frequency metric
            scr_freq_metrics_data = {
                't': metrics_t,
                'v': metrics_scr_freq
            }
            metrics_registry.register_signal('scr_frequency', scr_freq_metrics_data, { # No change needed, already correct
                'id': 'scr_frequency', # No change needed
                'type': 'eda_metrics',
                'label': 'SCR Frequency (events/min)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            
            time.sleep(0.01)

    def process_eda(self, input_signal_id="", show_peaks=True, output_signal_id="EDA_PROCESSED", enabled=True):
        if not enabled:
            return 0.0, 0.0, output_signal_id
        signal_id = output_signal_id
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            scl = getattr(self, '_last_scl', 0.0)
            sck = getattr(self, '_last_sck', 0.0)
            return scl, sck, output_signal_id
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
        scl = getattr(self, '_last_scl', 0.0)
        sck = getattr(self, '_last_sck', 0.0)
        return scl, sck, output_signal_id

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
