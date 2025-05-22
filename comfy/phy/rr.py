from collections import deque
import numpy as np
import threading
import time
import uuid
from ...src.phy.rr_signal_processing import RR
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor

class RRNode:
    """
    Node for processing RR (Respiratory Rate) signals.
    Provides visualization-ready data and calculates respiration rate using detected peaks.
    Buffer sizes for visualization and feature extraction are configurable.
    Now also returns is_peak for the latest sample, and prints RR if requested.
    Supports registry integration for visualization with optional peak highlighting.
    """
    
    # Track background processing threads
    _processing_threads = {}
    _stop_flags = {}

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
                "output_signal_id": ("STRING", {"default": "RR_PROCESSED"})            }
        }
        

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("Respiration_Rate", "Is_Peak", "Signal_ID")
    FUNCTION = "process_rr"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"    
    
    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        """
        Background processing thread to continuously update registry with processed RR data
        """
        registry = SignalRegistry.get_instance()
        # Constants for processing
        
        viz_window_sec = 60  # Visualization window: 1 minute
        viz_buffer_size = 100 * viz_window_sec  # 100 Hz assumed, 6,000 samples for 1 min
        feature_buffer_size = viz_buffer_size + 500  # Increased buffer size for better processing
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
            # Pre-process RR signal
            filtered_rr = RR.preprocess_signal(feature_values, fs=100)
            peaks = NumpySignalProcessor.find_peaks(filtered_rr, fs=100)
            # Calculate respiration rate
            rr_value, _ = RR.extract_respiration_rate(filtered_rr, fs=100)
            #print(f"Respiration Rate: {rr_value}")
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
            # Prepare filtered RR for visualization (match window)
            n_window = len(viz_values_window)
            filtered_viz_rr = filtered_rr[-n_window:] if len(filtered_rr) > n_window else filtered_rr
            if len(filtered_viz_rr) < n_window:
                padding = np.zeros(n_window - len(filtered_viz_rr))
                filtered_viz_rr = np.concatenate([padding, filtered_viz_rr])
            # Scale filtered RR to match amplitude
            if n_window > 0 and len(filtered_viz_rr) > 0:
                viz_min, viz_max = np.min(viz_values_window), np.max(viz_values_window)
                filtered_min, filtered_max = np.min(filtered_viz_rr), np.max(filtered_viz_rr)
                if filtered_max != filtered_min:
                    filtered_viz_rr = (filtered_viz_rr - filtered_min) / (filtered_max - filtered_min) * (viz_max - viz_min) + viz_min
            # Hash for registry update optimization
            data_hash = hash((tuple(viz_timestamps_window[-10:]), tuple(filtered_viz_rr[-10:]), tuple(peak_timestamps_in_window[-10:]), rr_value))
            if data_hash == last_registry_data_hash:
                time.sleep(0.033)
                continue
            last_registry_data_hash = data_hash
            metadata = {
                "id": output_signal_id,
                "type": "rr_processed",
                "respiration_rate": rr_value,
                "color": "#55F4FF",
                "show_peaks": show_peaks,
                "peak_marker": "o",
                "peak_timestamps": peak_timestamps_in_window
            }
            processed_signal_data = {
                "t": viz_timestamps_window.copy() if isinstance(viz_timestamps_window, np.ndarray) else viz_timestamps_window[:],
                "v": filtered_viz_rr.copy()
            }
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            time.sleep(0.033)
            
    def process_rr(self, input_signal_id="", show_peaks=True, output_signal_id="RR_PROCESSED"):
        """
        Processes RR signal from registry to extract respiration rate and peak detection.
        Automatically registers the processed signal back into the registry.

        Parameters:
        - input_signal_id: Signal ID to get data from registry.
        - show_peaks: Whether to include peak markings in visualization.
        - output_signal_id: ID for the processed signal in registry.

        Returns:
        - respiration_rate: Calculated respiration rate in breaths per minute.
        - is_peak_latest: Boolean indicating if the latest sample is a peak.
        - signal_id: The signal ID for registry integration.
        """
        # Get data from registry
        registry = SignalRegistry.get_instance()
        input_signal = registry.get_signal(input_signal_id)
        if not input_signal or "t" not in input_signal or "v" not in input_signal:
            raise ValueError(f"No valid signal found with ID {input_signal_id}. Make sure to provide a valid signal ID.")
        
        # Extract timestamps and values
        timestamps = np.array(input_signal["t"])
        values = np.array(input_signal["v"])
        
        if len(timestamps) < 2 or len(values) < 2:
            raise ValueError("Insufficient data in signal.")
            
        # Default processing parameters
        viz_buffer_size = 300
        feature_buffer_size = 600        # Use deques for processing - provides better handling of streaming data
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_values_deque.extend(values[-feature_buffer_size:] if len(values) > feature_buffer_size else values)
        
        # Convert to numpy array for processing
        feature_values = np.array(feature_values_deque)
        
        # Pre-process RR signal for better peak detection
        filtered_rr = RR.preprocess_signal(feature_values, fs=100)
        
        # Find peaks in the filtered signal
        peaks = NumpySignalProcessor.find_peaks(filtered_rr, fs=100)
        
        # Calculate respiration rate from the peaks
        rr_result, _ = RR.extract_respiration_rate(filtered_rr, fs=100)
        
        # Safely extract respiration rate value
        respiration_rate = 0  # Default value
        if isinstance(rr_result, (float, int)):
            respiration_rate = float(rr_result)
        elif isinstance(rr_result, np.ndarray) and rr_result.size > 0:
            respiration_rate = float(rr_result)
                    
        # Check if the latest data point is a peak
        is_peak_latest = False
        if len(peaks) > 0 and peaks[-1] >= len(feature_values) - 3:  # Consider a peak if it's one of the last 3 samples
            is_peak_latest = True
        
        # Ensure we're using the output signal ID for the processed signal
        # This guarantees the output signal is different from the input
        signal_id = output_signal_id
        
        # Stop any existing processing thread for this signal_id
        if signal_id in self._stop_flags:
            self._stop_flags[signal_id][0] = True
            
        # Create new thread for continuous processing
        stop_flag = [False]
        self._stop_flags[signal_id] = stop_flag
        
        thread = threading.Thread(
            target=self._background_process,
            args=(input_signal_id, show_peaks, stop_flag, output_signal_id),
            daemon=True
        )
        
        self._processing_threads[signal_id] = thread
        thread.start()
        
        # Return the respiration rate, peak detection, and the output signal ID
        return respiration_rate, is_peak_latest, output_signal_id
        
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

