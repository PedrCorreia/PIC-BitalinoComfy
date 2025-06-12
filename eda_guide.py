1from collections import deque
import numpy as np
import threading
import time
from ...src.phy.eda_signal_processing import EDA
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor

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
        from ...src.registry.plot_registry import PlotRegistry
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
        metrics_deque = deque(maxlen=metrics_buffer_size)
        max_frequency_interest = 100  # EDA is low freq, but keep for decimation logic
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.033
        start_time = None
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

            # Tonic/phasic extraction (use feature buffer for stability)
            tonic, phasic = self.eda.extract_tonic_phasic(feature_values, fs=effective_fs)
            # Map tonic/phasic to visualization window
            if len(tonic) == len(feature_values):
                # Align tonic/phasic to feature_timestamps, then map to viz window
                if len(feature_timestamps) == len(viz_timestamps_window):
                    tonic_viz = tonic[-len(viz_timestamps_window):]
                    phasic_viz = phasic[-len(viz_timestamps_window):]
                else:
                    # Interpolate to match window
                    if len(feature_timestamps) > 1 and len(viz_timestamps_window) > 1:
                        tonic_viz = np.interp(viz_timestamps_window, feature_timestamps, tonic)
                        phasic_viz = np.interp(viz_timestamps_window, feature_timestamps, phasic)
                    else:
                        tonic_viz = tonic[-len(viz_timestamps_window):]
                        phasic_viz = phasic[-len(viz_timestamps_window):]
            else:
                tonic_viz = tonic[-len(viz_timestamps_window):]
                phasic_viz = phasic[-len(viz_timestamps_window):]

            # --- Amplitude correction: check absolute change before/after filtering ---
            raw_window = viz_values_window
            filtered_tonic = tonic_viz
            filtered_phasic = phasic_viz
            raw_range = np.max(raw_window) - np.min(raw_window) if len(raw_window) > 0 else 1.0
            tonic_range = np.max(filtered_tonic) - np.min(filtered_tonic) if len(filtered_tonic) > 0 else 1.0
            phasic_range = np.max(filtered_phasic) - np.min(filtered_phasic) if len(filtered_phasic) > 0 else 1.0
            # If filtered amplitude is much lower than raw, amplify
            min_ratio = 0.2  # If filtered is less than 20% of raw, amplify
            amp_factor = 1.0
            if raw_range > 0 and (tonic_range/raw_range < min_ratio or phasic_range/raw_range < min_ratio):
                amp_factor = raw_range / max(tonic_range, phasic_range, 1e-6)
                tonic_viz = tonic_viz * amp_factor
                phasic_viz = phasic_viz * amp_factor

            # Normalize tonic and phasic for overlay plotting
            def normalize(arr):
                arr = np.asarray(arr)
                if arr.size == 0:
                    return arr
                minv, maxv = np.min(arr), np.max(arr)
                return (arr - minv) / (maxv - minv) if maxv > minv else arr * 0
            tonic_norm = normalize(tonic_viz)
            phasic_norm = normalize(phasic_viz)            # SCL/SCK calculation and rolling history
            scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0
            
            # Store for node output
            setattr(self, '_last_scl', scl)
            setattr(self, '_last_sck', sck)
            # SCR event (peak) detection and mapping to window (like ECG R-peak logic)
            result = self.eda.detect_events(phasic_viz, effective_fs)
            # Handle tuple return (validated_events, envelope) or just events
            if isinstance(result, tuple) and len(result) == 2:
                scr_event_indices, _ = result  # Extract just the indices, ignore envelope
            else:
                scr_event_indices = result
            
            # Ensure we have a clean array of indices
            if scr_event_indices is None:
                scr_event_indices = np.array([], dtype=int)
            else:
                scr_event_indices = np.array(scr_event_indices, dtype=int)
            
            # Only keep indices that are valid for the current window
            scr_event_indices = scr_event_indices[(scr_event_indices >= 0) & (scr_event_indices < len(viz_timestamps_window))]
            scr_event_times = viz_timestamps_window[scr_event_indices] if len(scr_event_indices) > 0 and len(viz_timestamps_window) > 0 else []
            scr_frequency = (len(scr_event_indices) / viz_window_sec) * 60 if viz_window_sec > 0 and len(scr_event_indices) > 0 else 0.0  # events per minute

            # --- Vectorized mapping of SCR event indices to timestamps in window (like RR node) ---
            peak_timestamps_in_window = []
            if show_peaks and len(scr_event_indices) > 0 and len(viz_timestamps_window) > 0:
                scr_event_times = viz_timestamps_window[scr_event_indices]
                window_min = viz_timestamps_window[0]
                window_max = viz_timestamps_window[-1]
                in_window_mask = (scr_event_times >= window_min) & (scr_event_times <= window_max)
                peak_timestamps_in_window = scr_event_times[in_window_mask].tolist()
            else:
                peak_timestamps_in_window = []
            scr_frequency = (len(peak_timestamps_in_window) / viz_window_sec) * 60 if viz_window_sec > 0 and len(peak_timestamps_in_window) > 0 else 0.0  # events per minute

            # --- RR-style peak metadata for visualization ---
            peak_marker = "o"
            peak_color = "#FF55AA"  # Choose a distinct color for EDA peaks

            # Hash for registry update optimization
            data_hash = hash((
                tuple(viz_timestamps_window[-5:]),
                tuple(tonic_norm[-5:]),
                tuple(phasic_norm[-5:]),
                tuple(peak_timestamps_in_window[-3:] if len(peak_timestamps_in_window) > 0 else []),
                scl, sck
            ))
            if data_hash == last_registry_data_hash:
                time.sleep(0.01)
                continue
            last_registry_data_hash = data_hash
            
            # Metadata and processed signal for registry
            metadata = {
                "id": output_signal_id,
                "type": "eda_processed",
                "scl": scl,
                "sck": sck,
                "scr_frequency": scr_frequency,
                # --- RR-style peak overlay fields for visualization ---
                "peak_timestamps": peak_timestamps_in_window,
                "peak_marker": "o",
                "color": "#FF55AA",  # Magenta for EDA peaks
                # --- End RR-style fields ---
                "scr_peak_timestamps": peak_timestamps_in_window,
                "scr_peak_indices": scr_event_indices.tolist(),  # Add indices for visualization
                "show_peaks": show_peaks,  # Add show_peaks flag for visualization
                "scl_metric_id": "SCL_METRIC",
                "sck_metric_id": "SCK_METRIC",
                "scr_metric_id": "SCR_METRIC",  # Add SCR frequency metric reference
                "tonic_norm": tonic_norm.tolist(),
                "phasic_norm": phasic_norm.tolist(),
                "timestamps": viz_timestamps_window.tolist(),
                "over": True
            }
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),
                "v": tonic_viz.tolist(),  # fallback for main plot
                "tonic": tonic_viz.tolist(),
                "phasic": phasic_viz.tolist(),
                "tonic_norm": tonic_norm.tolist(),
                "phasic_norm": phasic_norm.tolist(),
                "timestamps": viz_timestamps_window.tolist(),
                "id": output_signal_id            }
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
            metrics_registry.register_signal('SCL_METRIC', scl_metrics_data, {
                'id': 'SCL_METRIC',
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
            metrics_registry.register_signal('SCK_METRIC', sck_metrics_data, {
                'id': 'SCK_METRIC',
                'type': 'eda_metrics',
                'label': 'Skin Conductance Response (SCK)',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            
            # Register SCR frequency metric
            scr_freq_metrics_data = {
                't': metrics_t,
                'v': metrics_scr_freq
            }
            metrics_registry.register_signal('SCR_METRIC', scr_freq_metrics_data, {
                'id': 'SCR_METRIC',
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
4