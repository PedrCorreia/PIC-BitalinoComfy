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
        # Constants for processing
        
        viz_window_sec = 60  # Visualization window: 1 minute
        viz_buffer_size = 1000 * viz_window_sec  # 1000 Hz assumed, 60,000 samples for 1 min
        feature_buffer_size =viz_buffer_size + 500  # Increased buffer size for better processing
        # Initialize deques for processing
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)

        last_registry_data_hash = None
        while not stop_flag[0]:
            # Dynamically fetch the latest signal data from the registry each iteration
            signal_data = registry.get_signal(input_signal_id)
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                time.sleep(0.5)
                continue
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            if len(timestamps) < 2 or len(values) < 2:
                time.sleep(0.5)
                continue
            # Only append new data
            # If we have existing timestamps, only add new data points
            if len(viz_timestamps_deque) > 0 and len(timestamps) > 0:
                last_ts = viz_timestamps_deque[-1]
                new_data_idx = np.searchsorted(timestamps, last_ts, side='right')
                if new_data_idx < len(timestamps):
                    # Deques will automatically maintain the maxlen
                    viz_timestamps_deque.extend(timestamps[new_data_idx:])
                    viz_values_deque.extend(values[new_data_idx:])
                    feature_timestamps_deque.extend(timestamps[new_data_idx:])
                    feature_values_deque.extend(values[new_data_idx:])
            else:
                # First time initialization
                viz_timestamps_deque.extend(timestamps)
                viz_values_deque.extend(values)
                feature_timestamps_deque.extend(timestamps)
                feature_values_deque.extend(values)
            # Convert deques to numpy arrays for processing
            viz_timestamps = np.array(viz_timestamps_deque)
            viz_values = np.array(viz_values_deque)
            feature_timestamps = np.array(feature_timestamps_deque)
            feature_values = np.array(feature_values_deque)
            # Pre-process ECG for better peak detection (on full feature buffer)
            filtered_ecg = NumpySignalProcessor.bandpass_filter(feature_values, lowcut=5, highcut=18, fs=1000)
            peaks = ECG.detect_r_peaks(filtered_ecg, fs=1000, mode="qrs")
            # Calculate heart rate
            heart_rate = ECG.extract_heart_rate(feature_values, fs=1000, r_peaks=peaks)
            #print(f"Heart Rate: {heart_rate}")
            current_hr = 0
            if isinstance(heart_rate, np.ndarray) and heart_rate.size > 0:
                try:
                    current_hr = heart_rate[-1][0]
                except (IndexError, TypeError):
                    if heart_rate.size > 0:
                        current_hr = float(heart_rate[-1])
                    else:
                        current_hr = 0
            # Visualization window: last 60 seconds
            if len(viz_timestamps) > 0:
                window_max = viz_timestamps[-1]
                window_min = window_max - viz_window_sec
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
                in_window_mask = (peak_times >= viz_timestamps_window[0]) & (peak_times <= viz_timestamps_window[-1])
                peak_timestamps_in_window = peak_times[in_window_mask].tolist()
            # Prepare filtered ECG for visualization (match window)
            n_window = len(viz_values_window)
            filtered_viz_ecg = filtered_ecg[-n_window:] if len(filtered_ecg) > n_window else filtered_ecg
            if len(filtered_viz_ecg) < n_window:
                padding = np.zeros(n_window - len(filtered_viz_ecg))
                filtered_viz_ecg = np.concatenate([padding, filtered_viz_ecg])
            # Scale filtered ECG to match amplitude
            if n_window > 0 and len(filtered_viz_ecg) > 0:
                viz_min, viz_max = np.min(viz_values_window), np.max(viz_values_window)
                filtered_min, filtered_max = np.min(filtered_viz_ecg), np.max(filtered_viz_ecg)
                if filtered_max != filtered_min:
                    filtered_viz_ecg = (filtered_viz_ecg - filtered_min) / (filtered_max - filtered_min) * (viz_max - viz_min) + viz_min
            # Hash for registry update optimization
            data_hash = hash((tuple(viz_timestamps_window[-10:]), tuple(filtered_viz_ecg[-10:]), tuple(peak_timestamps_in_window[-10:]), current_hr))
            if data_hash == last_registry_data_hash:
                time.sleep(0.033)
                continue
            last_registry_data_hash = data_hash
            metadata = {
                "id": output_signal_id,
                "type": "ecg_processed",
                "heart_rate": current_hr,
                "color": "#FF5555",
                "show_peaks": show_peaks,
                "peak_marker": "x",
                "peak_timestamps": peak_timestamps_in_window
            }
            processed_signal_data = {
                "t": viz_timestamps_window.copy() if isinstance(viz_timestamps_window, np.ndarray) else viz_timestamps_window[:],
                "v": filtered_viz_ecg.copy()
            }
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            time.sleep(0.033)
            
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
        feature_buffer_size = 10000
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
