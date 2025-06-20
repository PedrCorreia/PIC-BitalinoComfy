from collections import deque
import numpy as np
import threading
import time
from ...src.test_src.rr_signal_processing import RR
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor
from ...src.utils.utils import Arousal

class RRNode:
    """
    Node for processing RR (Respiratory Rate) signals.
    Provides visualization-ready data and calculates respiration rate (breaths/min) using detected peaks.
    This is the ONLY node whose output should be used for RR_METRIC in MetricsView.
    """
    
    # Track background processing threads
    _processing_threads = {}
    _stop_flags = {}
#================================================================================================================================== Input Types ============================================================================================================================
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the RR node.
        - signal_deque: Deque of (timestamp, value) pairs.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.
        - print_rr: Print RR value if True.
        """
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),
                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "RR_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }
        

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("Respiration_Rate", "Is_Peak", "Signal_ID","Arousal")
    FUNCTION = "process_rr"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"    
    
    @classmethod
    def IS_CHANGED(elf, input_signal_id, show_peaks, output_signal_id, enabled):
        return float("NaN")

    def __init__(self):
        self._recent_peak_times = []
        self._last_rr = 0.0
        self._last_is_peak = False
        self.arousal= 0.5
#================================================================================================================================== Background Processing ============================================================================================================================
    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        #--- Initialize signal registry and buffers ---
        registry = SignalRegistry.get_instance()
        fs = 1000  # RR typical sampling frequency
        viz_window_sec = 10
        viz_buffer_size = fs * viz_window_sec
        feature_buffer_size = viz_buffer_size + fs
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        max_peaks_to_average = 10
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.033 # 30 Hz processing rate
        start_time = None
        # --- Filtering parameters for RR, adapted for decimation if needed ---
        nyquist_fs = fs / 2
        max_frequency_interest = 1000  # RR rarely above 1 Hz (60 bpm)
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1

        # Deque for storing (timestamp, rr_value) for RR_METRIC
        rr_metrics_deque = deque(maxlen=viz_buffer_size) # Match viz buffer size for now

        while not stop_flag[0]:
            current_time = time.time()
            if current_time - last_process_time < processing_interval:
                time.sleep(0.001)
                continue
            last_process_time = current_time
            signal_data = registry.get_signal(input_signal_id)
            #print(f"[RR_NODE_DEBUG] input_signal_id: {input_signal_id}, signal_data available: {signal_data is not None}") # DEBUG PRINT
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                time.sleep(processing_interval)
                continue
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            #print(f"[RR_NODE_DEBUG] Timestamps len: {len(timestamps)}, Values len: {len(values)}") # DEBUG PRINT
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
            # Only append new data
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
                            # Ensure feature deques also get decimated data if decimation is used
                            feature_timestamps_deque.extend(decimated_timestamps)
                            feature_values_deque.extend(decimated_values)
                    else:
                        viz_timestamps_deque.extend(timestamps[new_data_idx:])
                        viz_values_deque.extend(values[new_data_idx:])
                        feature_timestamps_deque.extend(timestamps[new_data_idx:])
                        feature_values_deque.extend(values[new_data_idx:])
            else: # Initial fill of the deques
                if use_decimation:
                    if len(timestamps) > 1:
                        decimated_values = NumpySignalProcessor.robust_decimate(values, decimation_factor)
                        decimated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=len(decimated_values))
                        viz_timestamps_deque.extend(decimated_timestamps)
                        viz_values_deque.extend(decimated_values)
                        # Ensure feature deques also get decimated data if decimation is used
                        feature_timestamps_deque.extend(decimated_timestamps)
                        feature_values_deque.extend(decimated_values)
                    # If len(timestamps) <= 1, robust_decimate might not be ideal,
                    # and extending with non-decimated (original) points might be safer
                    # or simply skip if not enough points for decimation.
                    # Current logic correctly handles if len(timestamps) > 1.
                    # If timestamps has 1 point, it won't decimate, deques remain empty until more data.
                else:
                    viz_timestamps_deque.extend(timestamps)
                    viz_values_deque.extend(values)
                    feature_timestamps_deque.extend(timestamps)
                    feature_values_deque.extend(values)
            
            viz_timestamps = np.array(viz_timestamps_deque)
            viz_values = np.array(viz_values_deque)
            feature_timestamps = np.array(feature_timestamps_deque)
            feature_values = np.array(feature_values_deque)
            # Adjust effective sampling rate if decimation was applied
            effective_fs = fs
            if use_decimation:
                effective_fs = fs / decimation_factor
            # Pre-process RR with bandpass filter adapted for RR (0.1-1 Hz)
            lowcut = 0.1
            highcut = 2
            filtered_rr = NumpySignalProcessor.bandpass_filter(
                feature_values,
                lowcut=lowcut,
                highcut=highcut,
                fs=effective_fs
            )
            #print(f"[RR_NODE_DEBUG] filtered_rr len: {len(filtered_rr)}, first 5: {filtered_rr[:5]}") # DEBUG PRINT
            # Calculate RR and maintain recent peak times
            detected_peaks = NumpySignalProcessor.find_peaks(filtered_rr, fs=effective_fs)
            #print(f"[RR_NODE_DEBUG] detected_peaks: {detected_peaks}") # DEBUG PRINT
            # Validate peaks with lag-based edge avoidance (similar to ECG)
            peaks =  detected_peaks# RR.validate_rr_peaks(filtered_rr, detected_peaks, lag=50, match_window=20)
            avg_rr = 0.0
            if isinstance(peaks, np.ndarray) and len(peaks) > 0 and len(feature_timestamps) > 0:
                peak_times = feature_timestamps[peaks]
                latest_peak_time = peak_times[-1]
                add_peak = False
                if not self._recent_peak_times or latest_peak_time != self._recent_peak_times[-1]:
                    if self._recent_peak_times:
                        last_time = self._recent_peak_times[-1]
                        if (latest_peak_time - last_time) >= 1.0:  # min breath interval
                            add_peak = True
                    else:
                        add_peak = True
                    if add_peak:
                        self._recent_peak_times.append(latest_peak_time)
                        if len(self._recent_peak_times) > max_peaks_to_average:
                            self._recent_peak_times.pop(0)
                if len(self._recent_peak_times) > 1:
                    breath_intervals = np.diff(self._recent_peak_times)
                    avg_breath = np.mean(breath_intervals)
                    if avg_breath > 0:
                        avg_rr = 60.0 / avg_breath  # Respiration rate in breaths/min
            #print(f"[RR_NODE_DEBUG] avg_rr: {avg_rr}") # DEBUG PRINT
            # Robust is_peak logic (timing + newness)
            is_peak, latest_peak_time = RR.is_peak(
                filtered_rr, feature_timestamps, fs, start_time=meta_start_time, rr=avg_rr, used_peaks=self._recent_peak_times[:-1] if len(self._recent_peak_times) > 1 else []
            )
            self._last_rr = avg_rr
            self._last_is_peak = is_peak
            # Update arousal based on RR
            self.arousal = Arousal.rr_arousal(avg_rr) if avg_rr > 0 else 0.5

            # --- RR Metric Registration ---
            current_metric_timestamp = feature_timestamps[-1] if len(feature_timestamps) > 0 else time.time()
            # Ensure avg_rr is a float and finite, default to 0.0 if not
            rr_metric_value = float(avg_rr) if avg_rr is not None and np.isfinite(avg_rr) else 0.0
            
            rr_metrics_deque.append((current_metric_timestamp, rr_metric_value))
            
            rr_metric_t = [x[0] for x in rr_metrics_deque]
            rr_metric_v = [x[1] for x in rr_metrics_deque]

            rr_metric_data = {
                "t": rr_metric_t,
                "v": rr_metric_v,
                "last": rr_metric_value
            }
            #print(f"[RR_NODE_DEBUG] RR_METRIC data: t_len={len(rr_metric_t)}, v_len={len(rr_metric_v)}, last={rr_metric_value}") # DEBUG PRINT
            # Use metrics_registry (which is SignalRegistry) for metrics
            metrics_registry = SignalRegistry.get_instance() # Ensure we use the correct registry instance
            metrics_registry.register_signal("RR_METRIC", rr_metric_data, {
                "id": "RR_METRIC",
                "type": "rr_metrics", # Consistent type for metrics
                "label": "Respiration Rate (bpm)",
                "source": output_signal_id,
                "scope": "global_metric"
            })
            # --- End RR Metric Registration ---

            # Visualization window
            if len(viz_timestamps) > 0:
                window_max = viz_timestamps[-1]
                window_min = max(window_max - viz_window_sec, viz_timestamps[0])
                window_mask = (viz_timestamps >= window_min) & (viz_timestamps <= window_max)
                viz_timestamps_window = viz_timestamps[window_mask]
                viz_values_window = viz_values[window_mask]
            else:
                viz_timestamps_window = viz_timestamps
                viz_values_window = viz_values


            # Vectorized peak mapping to window
            peak_timestamps_in_window = []
            if show_peaks and len(peaks) > 0 and len(feature_timestamps) > 0 and len(viz_timestamps_window) > 0:
                peak_times = feature_timestamps[peaks]
                in_window_mask = (peak_times >= window_min) & (peak_times <= window_max)
                peak_timestamps_in_window = peak_times[in_window_mask].tolist()
            # Prepare filtered RR for visualization (match window)
            filtered_viz_rr = np.zeros_like(viz_values_window)
            if len(filtered_rr) > 0 and len(viz_values_window) > 0:
                if len(window_mask) == len(filtered_rr):
                    filtered_viz_rr = filtered_rr[window_mask]
                else:
                    n_window = len(viz_values_window)
                    filtered_window = filtered_rr[-n_window:] if len(filtered_rr) > n_window else filtered_rr
                    if len(filtered_window) < n_window:
                        filtered_viz_rr[-len(filtered_window):] = filtered_window
                    else:
                        filtered_viz_rr = filtered_window
                viz_min, viz_max = np.min(viz_values_window), np.max(viz_values_window)
                filtered_min, filtered_max = np.min(filtered_viz_rr), np.max(filtered_viz_rr)
                if filtered_max != filtered_min and viz_max != viz_min:
                    filtered_viz_rr = (filtered_viz_rr - filtered_min) / (filtered_max - filtered_min) * (viz_max - viz_min) + viz_min

  

            data_hash = hash((
                tuple(viz_timestamps_window[-5:]),
                tuple(filtered_viz_rr[-5:]),
                tuple(peak_timestamps_in_window[-3:] if peak_timestamps_in_window else []),
                avg_rr
            ))
            # if data_hash == last_registry_data_hash: # Reverted debug change
            # time.sleep(0.01) # Reverted debug change
            # continue # Reverted debug change
            # last_registry_data_hash = time.time() # Reverted debug change
            if data_hash == last_registry_data_hash: # Original logic
                time.sleep(0.01)
                continue
            last_registry_data_hash = data_hash


            metadata = {
                "id": output_signal_id,
                "type": "processed",  # CHANGED from "rr_processed"
                "rr": avg_rr,
                "arousal": self.arousal,  # Add arousal value for metrics view
                "color": "#55F4FF",
                "show_peaks": show_peaks,
                "peak_marker": "o",
                "peak_timestamps": peak_timestamps_in_window
            }
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),
                "v": filtered_viz_rr.tolist()
            }
            #print(f"[RR_NODE_DEBUG] RR_PROCESSED data: t_len={len(processed_signal_data['t'])}, v_len={len(processed_signal_data['v'])}, metadata: {metadata}") # DEBUG PRINT
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            
            # Register arousal as a separate metric for the metrics view
            # metrics_registry = registry  # Use the same registry # This was potentially problematic if registry was PlotRegistry
            
            # Ensure we have a valid arousal value
            arousal_value = 0.5  # Default middle value
            if self.arousal is not None and isinstance(self.arousal, (float, int)) and np.isfinite(self.arousal):
                arousal_value = float(self.arousal)
                
            # Create timestamp - use the latest timestamp from the visualization window or current time
            current_timestamp = viz_timestamps_window[-1] if len(viz_timestamps_window) > 0 else time.time()
            
            # Prepare arousal metric data with both timeseries and 'last' value for easy access
            arousal_metrics_data = {
                "t": [current_timestamp],
                "v": [arousal_value],
                "last": arousal_value  # Add last value for easy access in metrics view
            }
            
            # Register with the metrics registry
            metrics_registry.register_signal("RR_AROUSAL_METRIC", arousal_metrics_data, {
                "id": "RR_AROUSAL_METRIC",
                "type": "arousal_metrics",
                "label": "RR Arousal",
                "source": output_signal_id,
                "scope": "global_metric",
                "arousal_value": float(arousal_value)  # Explicitly add arousal value as a float in metadata
            })
            time.sleep(0.01)
#================================================================================================================================== Processing Output ============================================================================================================================
    def process_rr(self, input_signal_id="", show_peaks=True, output_signal_id="RR_PROCESSED", enabled=True):
        if not enabled:
            return 0.0, False, output_signal_id, 0.5  # Include default arousal value of 0.5
        signal_id = output_signal_id
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            avg_rr = getattr(self, '_last_rr', 0.0)
            is_peak = getattr(self, '_last_is_peak', False)
            arousal = getattr(self, 'arousal', 0.5)  # Get current arousal value or default to 0.5
            return avg_rr, is_peak, output_signal_id, arousal
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
        return self._last_rr, self._last_is_peak, output_signal_id, self.arousal
        
    def __del__(self):
        """Clean up background threads when node is deleted"""
        for stop_flag in self._stop_flags.values():
            stop_flag[0] = True
        self._processing_threads.clear()
        self._stop_flags.clear()

NODE_CLASS_MAPPINGS = {
    "RRNode": RRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RRNode": "🌬️ RR Processor (Registry)"
}

