from collections import deque
import numpy as np
import threading
import time
import uuid
import scipy.signal
from ...src.phy.ecg_signal_processing import ECG
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor

class ECGNode:
    """
    Node for processing ECG (Electrocardiogram) signals.
    Takes an input signal ID, processes the ECG data, and outputs heart rate and peak detection.
    Supports registry integration for visualization with optional peak highlighting.
    
    NOTE: The RR value here is the R-R interval (heartbeat interval), NOT respiration rate. Do NOT use this for RR_METRIC in MetricsView.
    """
    
    # Track background processing threads
    _processing_threads = {}
    _stop_flags = {}

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the ECG node.
        - input_signal_id: Signal ID for data from registry.
        - show_peaks: Control peak visualization in registry.
        - output_signal_id: ID for the processed signal in registry.
        - enabled: Boolean to enable or disable processing.
        """
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),
                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "ECG_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }
        
    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("Heart_Rate", "Rpeak", "Signal_ID")
    FUNCTION = "process_ecg"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def __init__(self):
        self.ecg = ECG()  # Now an instance, stateful
        # Remove redundant state tracking; ECG class handles peak state

    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        """
        Background processing thread to continuously update registry with processed ECG data
        """
        registry = SignalRegistry.get_instance()
        metrics_registry = SignalRegistry.get_instance()
        fs = 1000  # Sampling frequency assumption
        nyquist_fs = fs / 2  # Nyquist frequency
        viz_window_sec = 5
        viz_buffer_size = fs * viz_window_sec
        feature_buffer_size = viz_buffer_size + fs
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        max_peaks_to_average = 10
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.033
        # --- define decimation_factor and use_decimation ---
        max_frequency_interest = 250  # Hz for QRS complex
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        start_time = None
        metrics_buffer_size = 300  # e.g., 5 minutes at 1Hz
        metrics_deque = deque(maxlen=metrics_buffer_size)
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
            # --- start_time sync for is_peak ---
            # Only use meta_start_time for is_peak, not for timestamp alignment
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
                
            # Pre-process ECG using more efficient filter settings
            # Adjust filter bands based on decimation
            lowcut = 8  # Hz
            highcut = min(18, 20)  # Stay well below Nyquist after decimation
            
            filtered_ecg = NumpySignalProcessor.bandpass_filter(
                feature_values, 
                lowcut=lowcut, 
                highcut=highcut, 
                fs=effective_fs
            )
            # Use ECG class for peak and HR calculation with separate methods
            avg_hr = self.ecg.calculate_hr(
                filtered_ecg, feature_timestamps, effective_fs, max_peaks_to_average
            )
            is_peak = self.ecg.is_peak(
                filtered_ecg, feature_timestamps, effective_fs, start_time=meta_start_time, hr=avg_hr
            )
            # Store latest state in ECG instance for access from process_ecg
            setattr(self.ecg, '_last_hr', avg_hr)
            setattr(self.ecg, '_last_is_peak', is_peak)
            # For visualization, get peaks for windowing
            peaks = self.ecg.detect_r_peaks(filtered_ecg, fs=effective_fs, mode="qrs")
            if len(peaks) > 0 and len(feature_timestamps) > 0:
                peak_times = feature_timestamps[peaks]
                #print(f"[ECGNode] Peak timestamps (s): {peak_times}")
            
            # Visualization window: use adaptive window size for better performance
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
                
            # Prepare filtered ECG for visualization - use more efficient array operations
            filtered_viz_ecg = np.zeros_like(viz_values_window)
            
            if len(filtered_ecg) > 0 and len(viz_values_window) > 0:
                # Match filtered_ecg to viz_window
                if len(window_mask) == len(filtered_ecg):
                    # Direct window masking if arrays align
                    filtered_viz_ecg = filtered_ecg[window_mask]
                else:
                    # Interpolate to match window size
                    # For visualization, simple truncation/padding is sufficient
                    n_window = len(viz_values_window)
                    filtered_window = filtered_ecg[-n_window:] if len(filtered_ecg) > n_window else filtered_ecg
                    
                    if len(filtered_window) < n_window:
                        # Pad with zeros if needed
                        filtered_viz_ecg[-len(filtered_window):] = filtered_window
                    else:
                        filtered_viz_ecg = filtered_window
                
                # Scale efficiently using vectorized operations
                viz_min, viz_max = np.min(viz_values_window), np.max(viz_values_window)
                filtered_min, filtered_max = np.min(filtered_viz_ecg), np.max(filtered_viz_ecg)
                
                if filtered_max != filtered_min and viz_max != viz_min:
                    filtered_viz_ecg = (filtered_viz_ecg - filtered_min) / (filtered_max - filtered_min) * (viz_max - viz_min) + viz_min
                  
            # Hash for registry update optimization - use fewer samples for hashing
            data_hash = hash((
                tuple(viz_timestamps_window[-5:]), 
                tuple(filtered_viz_ecg[-5:]), 
                tuple(peak_timestamps_in_window[-3:] if peak_timestamps_in_window else []), 
                avg_hr
            ))
            
            if data_hash == last_registry_data_hash:
                time.sleep(0.01)  # Short sleep to reduce CPU usage
                continue
                
            last_registry_data_hash = data_hash

            # Compute RR interval (mean of RR intervals in window) BEFORE metadata
            avg_rr = None
            if len(peaks) > 1 and len(feature_timestamps) > 0:
                rr_intervals = np.diff(feature_timestamps[peaks])
                if len(rr_intervals) > 0:
                    avg_rr = float(np.mean(rr_intervals))

            # Optimize metadata
            metadata = {
                "id": output_signal_id,
                "type": "ecg_processed",
                "hr": avg_hr,  
                "color": "#FF5555",
                "show_peaks": show_peaks,
                "peak_marker": "x",
                "peak_timestamps": peak_timestamps_in_window,
                "decimation_factor": decimation_factor if use_decimation else 1,
                "effective_fs": effective_fs,
                "hr_metric_id": "HR_METRIC"  # <-- Add reference to metric signal
            }
            
            # Efficiently prepare processed signal data
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),  # Convert to list for JSON compatibility
                "v": filtered_viz_ecg.tolist()
            }
            
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            
            # After ECG processing and before sleep:
            # Get the last timestamp from feature_timestamps (if available)
            last_timestamp = float(feature_timestamps[-1]) if len(feature_timestamps) > 0 else time.time()
            # Compute RR interval (mean of RR intervals in window)
            avg_rr = None
            if len(peaks) > 1 and len(feature_timestamps) > 0:
                rr_intervals = np.diff(feature_timestamps[peaks])
                if len(rr_intervals) > 0:
                    avg_rr = float(np.mean(rr_intervals))
            # Append to metrics deque
            metrics_deque.append((last_timestamp, avg_hr, avg_rr))
            # Prepare metrics signal for registry (per-node/local metrics)
            metrics_t = [x[0] for x in metrics_deque]
            metrics_hr = [x[1] for x in metrics_deque]
            # Register global HR metric as a time series for MetricsView compatibility
            metrics_data = {
                't': metrics_t,
                'v': metrics_hr
            }
            metrics_registry.register_signal('HR_METRIC', metrics_data, {
                'id': 'HR_METRIC',
                'type': 'ecg_metrics',
                'label': 'Global Heart Rate',
                'source': output_signal_id,
                'scope': 'global_metric'
            })
            #print(f"[ECGNode][metrics_registry] {output_signal_id + '_METRICS'}: t={metrics_t[-1] if metrics_t else None}, hr={metrics_hr[-1] if metrics_hr else None}")
            time.sleep(0.01)  # Reduced sleep time for faster updates

    def process_ecg(self, input_signal_id, show_peaks, output_signal_id, enabled=True):
        """
        Processes ECG signal from registry to extract heart rate and peak detection.
        Returns the latest average HR and Rpeak calculated by the background thread.
        """
        signal_id = output_signal_id
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            # Always get the latest state from the ECG instance
            avg_hr = getattr(self.ecg, '_last_hr', 0.0) if hasattr(self.ecg, '_last_hr') else 0.0
            is_peak = getattr(self.ecg, '_last_is_peak', False) if hasattr(self.ecg, '_last_is_peak') else False
            return avg_hr, is_peak, output_signal_id
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
        # On first call, return 0, False
        return 0.0, False, output_signal_id
        
    def __del__(self):
        """Clean up background threads when node is deleted"""
        for stop_flag in self._stop_flags.values():
            stop_flag[0] = True
        self._processing_threads.clear()
        self._stop_flags.clear()

NODE_CLASS_MAPPINGS = {
    "ECGNode": ECGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ECGNode": "💓 ECG Processor (Registry)"
}
