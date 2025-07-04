from collections import deque
import numpy as np
import threading
import time
from ...src.phy.ecg_signal_processing import ECG
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor
from ...src.utils.utils import Arousal

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
        self.ecg_arousal = 0.5  # ECG/HR arousal level (0.0-1.0)
        self._recent_peak_times = []  # Store recent R-peak timestamps for time-based HR calculation

    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        """
        Background processing thread to continuously update registry with processed ECG data
        """
        registry = SignalRegistry.get_instance()
        metrics_registry = SignalRegistry.get_instance()
        fs = 1000  # Sampling frequency assumption
        nyquist_fs = fs / 2  # Nyquist frequency
        viz_window_sec = 10
        viz_buffer_size = fs * viz_window_sec
        
        # --- define decimation_factor and use_decimation ---
        max_frequency_interest = 400  # Hz for QRS complex
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        
        # ECG feature buffer: preserve 15 seconds of data, accounting for decimation
        feature_window_sec = 15.0  # 15 seconds for ECG features
        effective_fs = fs / decimation_factor if use_decimation else fs
        feature_buffer_size = int(effective_fs * feature_window_sec)
        
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        max_peaks_to_average = 10
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.2  # 5 Hz processing rate (reduced for better performance)
        start_time = None
        # ECG metrics buffer: preserve 5 minutes of HR data at ~1Hz update rate
        metrics_window_sec = 300.0  # 5 minutes for HR metrics
        metrics_update_rate = 1.0   # Approximately 1 Hz for metrics
        metrics_buffer_size = int(metrics_window_sec * metrics_update_rate)
        metrics_deque = deque(maxlen=metrics_buffer_size)
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
            if len(timestamps) < 2 or len(values) < 2:
                continue  # Skip this iteration, don't sleep
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
                
            # Pre-process ECG using more efficient filter settings
            # Adjust filter bands based on decimation
            lowcut = 8  # Hz
            highcut = min(18, 20)  # Stay well below Nyquist after decimation
            
            filtered_ecg = NumpySignalProcessor.bandpass_filter(
                feature_values, 
                lowcut=lowcut, 
                highcut=highcut, 
                fs=effective_fs,
                order=4,
            )
            # Time-based HR calculation (same robust method as RR)
            # First detect R-peaks in the current data
            detected_peaks = self.ecg.detect_r_peaks(filtered_ecg, fs=effective_fs, mode="qrs")
            
            avg_hr = 0.0
            current_signal_time = feature_timestamps[-1] if len(feature_timestamps) > 0 else 0.0
            
            if isinstance(detected_peaks, np.ndarray) and len(detected_peaks) > 0 and len(feature_timestamps) > 0:
                peak_times = feature_timestamps[detected_peaks]
                latest_peak_time = peak_times[-1]
                
                # Add new peak if it's different from the last one (avoid duplicates)
                add_peak = False
                if not self._recent_peak_times or latest_peak_time != self._recent_peak_times[-1]:
                    if self._recent_peak_times:
                        last_time = self._recent_peak_times[-1]
                        if (latest_peak_time - last_time) >= 0.3:  # 0.3s min interval (realistic for heartbeat)
                            add_peak = True
                    else:
                        add_peak = True
                    if add_peak:
                        self._recent_peak_times.append(latest_peak_time)
                        
                # TIME-BASED FILTERING: Only keep peaks from last 30 seconds (signal time)
                time_window = 30.0  # 30 seconds
                cutoff_signal_time = current_signal_time - time_window
                self._recent_peak_times = [t for t in self._recent_peak_times if t >= cutoff_signal_time]
                
                # Calculate HR only if we have recent peaks (within 30 seconds of signal time)
                if len(self._recent_peak_times) >= 1:
                    # Check if most recent peak is recent enough (signal time domain)
                    most_recent_peak = max(self._recent_peak_times)
                    signal_time_since_last_peak = current_signal_time - most_recent_peak
                    
                    if signal_time_since_last_peak <= 5.0:  # If no peak in last 5 seconds of signal time, HR = 0
                        # Use fixed time window approach: peaks per fixed time window
                        actual_time_window = min(time_window, current_signal_time - min(self._recent_peak_times))
                        # Ensure we have a reasonable time window (at least 5s for reliable calculation)
                        if actual_time_window >= 5.0:
                            peak_count = len(self._recent_peak_times)
                            
                            # Use FIXED time window approach: peaks per fixed time window
                            beats_per_second = peak_count / actual_time_window
                            avg_hr = beats_per_second * 60.0  # Convert to beats per minute
                        else:
                            avg_hr = 0.0  # Not enough time window for reliable calculation
                    else:
                        avg_hr = 0.0  # No recent heartbeat detected
                else:
                    avg_hr = 0.0  # Not enough peaks for calculation
            
            # Handle case when no peaks detected - set HR to 0 for proper arousal calculation
            if avg_hr is None or not np.isfinite(avg_hr) or avg_hr <= 0:
                avg_hr = 0.0  # No heartbeat detected
            
            is_peak = self.ecg.is_peak(
                filtered_ecg, feature_timestamps, effective_fs, start_time=meta_start_time, hr=avg_hr
            )
            # Store latest state in ECG instance for access from process_ecg
            setattr(self.ecg, '_last_hr', avg_hr)
            setattr(self.ecg, '_last_is_peak', is_peak)
            
            # Calculate ECG/HR arousal using smart methods (same pattern as RR)
            # Handle edge cases: HR=0 (no heartbeat) should map to very low arousal
            if avg_hr <= 0:
                self.ecg_arousal = 0.0  # No heartbeat = no arousal
            else:
                self.ecg_arousal = Arousal.hr_arousal(avg_hr)
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
                  
            # Hash for registry update optimization - include deque length and range to detect rotation
            data_hash = hash((
                len(viz_timestamps_window),  # Length changes when deque rotates
                tuple(viz_timestamps_window[-5:]) if len(viz_timestamps_window) >= 5 else tuple(viz_timestamps_window), 
                tuple(viz_timestamps_window[:3]) if len(viz_timestamps_window) >= 3 else tuple(),  # First few values change on rotation
                tuple(filtered_viz_ecg[-5:]) if len(filtered_viz_ecg) >= 5 else tuple(filtered_viz_ecg), 
                tuple(peak_timestamps_in_window[-3:] if peak_timestamps_in_window else []), 
                avg_hr,
                viz_timestamps_window[0] if len(viz_timestamps_window) > 0 else 0,  # First timestamp changes on rotation
                viz_timestamps_window[-1] if len(viz_timestamps_window) > 0 else 0  # Last timestamp always changes
            ))
            
            if data_hash == last_registry_data_hash:
                continue  # Skip when no data available
                
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
            
            # Register ECG arousal metric (following RR/EDA pattern)
            ecg_arousal_value = 0.5  # Default middle value
            if self.ecg_arousal is not None and isinstance(self.ecg_arousal, (float, int)) and np.isfinite(self.ecg_arousal):
                ecg_arousal_value = float(self.ecg_arousal)
            
            # Prepare ECG arousal metric data
            ecg_arousal_metrics_data = {
                "t": [last_timestamp],
                "v": [ecg_arousal_value],
                "last": ecg_arousal_value
            }
            metrics_registry.register_signal('ECG_AROUSAL_METRIC', ecg_arousal_metrics_data, {
                'id': 'ECG_AROUSAL_METRIC',
                'type': 'arousal_metric',
                'label': 'ECG Arousal Level',
                'source': output_signal_id,
                'scope': 'global_metric',
                'arousal_value': ecg_arousal_value
            })
            
            #print(f"[ECGNode][metrics_registry] {output_signal_id + '_METRICS'}: t={metrics_t[-1] if metrics_t else None}, hr={metrics_hr[-1] if metrics_hr else None}")
            continue  # Continue processing loop without delay

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
