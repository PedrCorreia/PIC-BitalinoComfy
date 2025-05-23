from collections import deque
import numpy as np
import threading
import time
import uuid
from ...src.phy.ecg_signal_processing import ECG
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor

class ECGNode:
    """
    Node for processing ECG (Electrocardiogram) signals.
    Takes an input signal ID, processes the ECG data, and outputs heart rate and peak detection.
    Supports registry integration for visualization with optional peak highlighting.
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
        """
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),
                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "ECG_PROCESSED"})            }
        }
        
    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("Heart_Rate", "Rpeak", "Signal_ID")
    FUNCTION = "process_ecg"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    
    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        """
        Background processing thread to continuously update registry with processed ECG data
        """
        registry = SignalRegistry.get_instance()
        
        # Optimize buffer sizes based on Nyquist principles
        fs = 1000  # Sampling frequency assumption
        nyquist_fs = fs / 2  # Nyquist frequency
        viz_window_sec = 5  # Reduced from 60 to 5 seconds for more efficient visualization
        
        # Buffer sizes optimized for performance
        viz_buffer_size = fs * viz_window_sec  # Exactly 5 seconds of data at full rate
        feature_buffer_size = viz_buffer_size + fs  # Add 1 second for processing margin
        
        # Initialize deques for processing
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        
        # Determine if decimation is needed (only if fs > 2*max_frequency_of_interest)
        # For ECG QRS complex, max frequency of interest is ~40Hz
        max_frequency_interest = 40  # Hz for QRS complex
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1

        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.033  # ~30Hz updates for visualization
        
        while not stop_flag[0]:
            current_time = time.time()
            
            # Only process at the specified interval
            if current_time - last_process_time < processing_interval:
                time.sleep(0.001)  # Short sleep to prevent CPU hogging
                continue
                
            last_process_time = current_time
                
            # Dynamically fetch the latest signal data from the registry each iteration
            signal_data = registry.get_signal(input_signal_id)
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                time.sleep(processing_interval)
                continue
                
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            
            if len(timestamps) < 2 or len(values) < 2:
                time.sleep(processing_interval)
                continue
                
            # Only append new data
            if len(viz_timestamps_deque) > 0 and len(timestamps) > 0:
                last_ts = viz_timestamps_deque[-1]
                new_data_idx = np.searchsorted(timestamps, last_ts, side='right')
                
                if new_data_idx < len(timestamps):
                    # Apply decimation if needed
                    if use_decimation:
                        # Keep only every Nth sample for processing
                        new_indices = np.arange(new_data_idx, len(timestamps), decimation_factor)
                        viz_timestamps_deque.extend(timestamps[new_indices])
                        viz_values_deque.extend(values[new_indices])
                        feature_timestamps_deque.extend(timestamps[new_indices])
                        feature_values_deque.extend(values[new_indices])
                    else:
                        # Use all samples
                        viz_timestamps_deque.extend(timestamps[new_data_idx:])
                        viz_values_deque.extend(values[new_data_idx:])
                        feature_timestamps_deque.extend(timestamps[new_data_idx:])
                        feature_values_deque.extend(values[new_data_idx:])
            else:
                # First time initialization with optional decimation
                if use_decimation:
                    indices = np.arange(0, len(timestamps), decimation_factor)
                    viz_timestamps_deque.extend(timestamps[indices])
                    viz_values_deque.extend(values[indices])
                    feature_timestamps_deque.extend(timestamps[indices])
                    feature_values_deque.extend(values[indices])
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
            highcut = min(18, effective_fs * 0.4)  # Stay well below Nyquist after decimation
            
            filtered_ecg = NumpySignalProcessor.bandpass_filter(
                feature_values, 
                lowcut=lowcut, 
                highcut=highcut, 
                fs=effective_fs
            )
            
            # Detect peaks more efficiently (adjust fs based on decimation)
            peaks = ECG.detect_r_peaks(filtered_ecg, fs=effective_fs, mode="qrs")
            
            # Calculate heart rate
            heart_rate = ECG.extract_heart_rate(feature_values, fs=effective_fs, r_peaks=peaks)
            
            current_hr = 0
            if isinstance(heart_rate, np.ndarray) and heart_rate.size > 0:
                try:
                    current_hr = heart_rate[-1][0]
                except (IndexError, TypeError):
                    if heart_rate.size > 0:
                        current_hr = float(heart_rate[-1])
                    else:
                        current_hr = 0
                        
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
                current_hr
            ))
            
            if data_hash == last_registry_data_hash:
                time.sleep(0.01)  # Short sleep to reduce CPU usage
                continue
                
            last_registry_data_hash = data_hash
            
            # Optimize metadata
            metadata = {
                "id": output_signal_id,
                "type": "ecg_processed",
                "heart_rate": current_hr,
                "color": "#FF5555",
                "show_peaks": show_peaks,
                "peak_marker": "x",
                "peak_timestamps": peak_timestamps_in_window,
                "decimation_factor": decimation_factor if use_decimation else 1,
                "effective_fs": effective_fs
            }
            
            # Efficiently prepare processed signal data
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),  # Convert to list for JSON compatibility
                "v": filtered_viz_ecg.tolist()
            }
            
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            time.sleep(0.01)  # Reduced sleep time for faster updates

    def process_ecg(self, input_signal_id, show_peaks, output_signal_id):
        """
        Processes ECG signal from registry to extract heart rate and peak detection.
        Automatically registers the processed signal back into the registry.

        Parameters:
        - input_signal_id: Signal ID to get data from registry.
        - show_peaks: Whether to include peak markings in visualization.
        - output_signal_id: ID for the processed signal in registry.

        Returns:
        - heart_rate: Calculated heart rate in beats per minute (bpm).
        - Rpeak: Boolean indicating if the current sample is a peak.
        - signal_id: The signal ID for registry integration.
        """
        registry = SignalRegistry.get_instance()
        input_signal = registry.get_signal(input_signal_id)
        if not input_signal or "t" not in input_signal or "v" not in input_signal:
            print(f"[ECGNode] No valid signal found with ID {input_signal_id}. Make sure to provide a valid signal ID.")
            return 0.0, False, output_signal_id
        timestamps = np.array(input_signal["t"])
        values = np.array(input_signal["v"])
        if len(timestamps) < 2 or len(values) < 2:
            print("[ECGNode] Insufficient data in signal.")
            return 0.0, False, output_signal_id
        # Use optimized buffer size based on sampling frequency
        fs = 1000  # Assumed sampling frequency
        feature_buffer_size = fs * 3  # 3 seconds of data is typically enough for ECG analysis
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_values_deque.extend(values[-feature_buffer_size:] if len(values) > feature_buffer_size else values)
        feature_values = np.array(feature_values_deque)
        # Add margin to avoid edge effects for filtfilt (adds latency but improves quality)
        margin = 500  # samples, e.g. 0.5s at 1000Hz
        if len(feature_values) > 2 * margin:
            # Only process the central region, avoid edges
            process_start = margin
            process_end = len(feature_values) - margin
            process_values = feature_values[process_start:process_end]
            filtered_ecg = NumpySignalProcessor.bandpass_filter(process_values, lowcut=8, highcut=18, fs=1000)
            # Pad filtered_ecg to match feature_values length
            filtered_ecg = np.pad(filtered_ecg, (process_start, len(feature_values) - process_end), mode='constant')
        else:
            filtered_ecg = NumpySignalProcessor.bandpass_filter(feature_values, lowcut=8, highcut=18, fs=1000)
        peaks = ECG.detect_r_peaks(filtered_ecg, fs=1000, mode="qrs")
        hr_result = ECG.extract_heart_rate(feature_values, fs=1000, r_peaks=peaks)
        heart_rate = 0
        if isinstance(hr_result, np.ndarray) and hr_result.size > 0:
            try:
                heart_rate = hr_result[-1][0]
            except (IndexError, TypeError):
                if hr_result.size > 0:
                    heart_rate = float(hr_result[-1])
                else:
                    heart_rate = 0
        Rpeak = False
        if len(peaks) > 0 and peaks[-1] >= len(feature_values) - 3:
            Rpeak = True
        signal_id = output_signal_id
        # Defensive: If already processing this signal_id, just return
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            return heart_rate, Rpeak, output_signal_id
        # Defensive: Stop any existing processing thread for this signal_id
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
        return heart_rate, Rpeak, output_signal_id
        
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
