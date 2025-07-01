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
                "input_signal_id": ("STRING", {"default": ""}),                
                "show_peaks": ("BOOLEAN", {"default": True}),
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
            tonic_raw = EDA.convert_adc_to_eda(tonic_raw)  # Convert to microsiemens
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

            # --- Custom normalization for visualization ---
            # REMOVE normalization here; output raw values for visualization
            # processed_signal_data should contain raw tonic and phasic
            # SCR event (peak) detection and mapping to window (like ECG R-peak logic)
            scr_event_indices = np.array([], dtype=int)
            if len(phasic_viz) > 0:
                result = self.eda.detect_events(phasic_viz, effective_fs)
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
            if show_peaks and len(scr_event_indices) > 0 and len(viz_timestamps_window) > 0:
                scr_event_times = viz_timestamps_window[scr_event_indices]
                window_min = viz_timestamps_window[0]
                window_max = viz_timestamps_window[-1]
                in_window_mask = (scr_event_times >= window_min) & (scr_event_times <= window_max)
                peak_timestamps_in_window = scr_event_times[in_window_mask].tolist()
            else:
                peak_timestamps_in_window = []
            scr_frequency = (len(peak_timestamps_in_window) / viz_window_sec) * 60 if viz_window_sec > 0 and len(peak_timestamps_in_window) > 0 else 0.0
            scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0

            # Update instance variables for return values
            self._last_scl = scl
            self._last_scr_frequency = scr_frequency

            metadata = {
                "id": output_signal_id,
                "type": "processed",
                "viz_subtype": "eda",
                "scl": scl,
                "sck": sck,
                "scr_frequency": scr_frequency,
                "peak_timestamps": peak_timestamps_in_window
            }
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),
                "tonic": tonic_viz.tolist(),    # Raw values
                "phasic": phasic_viz.tolist(),  # Raw values
                "peak_indices": scr_event_indices.tolist(),
                "id": output_signal_id
            }
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
            time.sleep(0.01)

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
