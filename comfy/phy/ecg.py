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
                "output_signal_id": ("STRING", {"default": "ECG_PROCESSED"})
            }
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
        feature_buffer_size = 5000
        viz_buffer_size = 2000  # Default visualization buffer size
        while not stop_flag[0]:
            # Dynamically fetch the latest signal data from the registry each iteration
            signal_data = registry.get_signal(input_signal_id)
            
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                # Not enough data yet, sleep and retry
                time.sleep(0.1)
                continue
                
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            
            if len(timestamps) < 2 or len(values) < 2:
                time.sleep(0.1)
                continue
            viz_timestamps = timestamps[-viz_buffer_size:] if len(timestamps) > viz_buffer_size else timestamps
            viz_values = values[-viz_buffer_size:] if len(values) > viz_buffer_size else values
            feature_values = values[-feature_buffer_size:] if len(values) > feature_buffer_size else values
            
            # Apply simplified processing - convert to binary signal (1 if above half amplitude, 0 if below)
            max_amplitude = np.max(feature_values)
            threshold = max_amplitude / 2
            
            # Create binary signal (1 if above threshold, 0 if below)
            binary_signal = np.zeros_like(feature_values)
            binary_signal[feature_values > threshold] = 1
            
            # Use this binary signal for processing
            filtered_ecg = binary_signal
            
            # Find transitions from 0 to 1 as peaks (rising edges)
            peaks = np.where(np.diff(binary_signal) > 0)[0]
            
            # Calculate heart rate as number of peaks per minute
            if len(peaks) > 1:
                # Calculate average time between peaks
                avg_peak_interval = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                # Convert to heart rate (beats per minute)
                heart_rate = 60 * 1000 / avg_peak_interval if avg_peak_interval > 0 else 0
            else:
                heart_rate = 0
            
            # Safely extract heart rate value
            current_hr = 0  # Default value
            if isinstance(heart_rate, np.ndarray) and heart_rate.size > 0:
                try:
                    current_hr = heart_rate[-1][0]
                except (IndexError, TypeError):
                    # If heart_rate doesn't have the expected structure
                    if heart_rate.size > 0:
                        current_hr = float(heart_rate[-1])
                    else:
                        current_hr = 0
            
            # Create metadata with peak information
            metadata = {
                "id": output_signal_id,
                "type": "ecg_processed", 
                "heart_rate": current_hr,
                "color": "#FF5555",  # Red color for ECG signal
                "show_peaks": show_peaks
            }
              # If we're showing peaks, calculate and add them
            if show_peaks and len(values) > 0:
                # Find indices of peaks within visualization window
                if len(values) <= viz_buffer_size:
                    # If we're showing all values, just use the peak indices directly
                    # But make sure they're within range of the signal
                    valid_indices = np.array([p for p in peaks if p < len(viz_values)])
                else:
                    # Calculate the offset between feature values and the original signal
                    feature_offset = len(values) - len(feature_values)
                      # Adjust peak indices to match the original signal scale
                    adjusted_peaks = peaks + feature_offset
                    
                    # Get indices for the visualization window
                    viz_start_idx = len(values) - len(viz_values)
                    
                    # Find peaks that are within the visualization window
                    viz_peak_indices = [p - viz_start_idx for p in adjusted_peaks if viz_start_idx <= p < len(values)]
                    
                    # Filter to valid indices only - they must be within the viz window range
                    valid_indices = np.array([idx for idx in viz_peak_indices if 0 <= idx < len(viz_values)])
                
                # Add peak information to metadata
                metadata["peaks"] = valid_indices.tolist()
                  # Create new processed signal data structure with binarized values
            # This ensures we're not just referencing the original data
            viz_max_amplitude = np.max(viz_values)
            viz_threshold = viz_max_amplitude / 2
            viz_binary = np.zeros_like(viz_values)
            viz_binary[viz_values > viz_threshold] = 1
            
            processed_signal_data = {
                "t": viz_timestamps.copy() if isinstance(viz_timestamps, np.ndarray) else viz_timestamps[:],
                "v": viz_binary  # Use the binarized signal for visualization
            }
            
            # Register the processed signal with a distinct ID from the input
            # Using output_signal_id ensures we're not overwriting the input signal
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            
            # Update every 50ms (20Hz) - provides smooth visualization without too much CPU load
            time.sleep(0.05)

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
        """        # Get data from registry
        registry = SignalRegistry.get_instance()
        input_signal = registry.get_signal(input_signal_id)
        
        if not input_signal or "t" not in input_signal or "v" not in input_signal:
            raise ValueError(f"No valid signal found with ID {input_signal_id}. Make sure to provide a valid signal ID.")
        
        # Extract timestamps and values
        timestamps = np.array(input_signal["t"])
        values = np.array(input_signal["v"])
        
        if len(timestamps) < 2 or len(values) < 2:
            raise ValueError("Insufficient data in signal.")
            
        # Default feature buffer size for processing
        feature_buffer_size = 5000
        feature_values = values[-feature_buffer_size:] if len(values) > feature_buffer_size else values
          # Apply simplified processing - convert to binary signal (1 if above half amplitude, 0 if below)
        max_amplitude = np.max(feature_values)
        threshold = max_amplitude / 2
        
        # Create binary signal (1 if above threshold, 0 if below)
        binary_signal = np.zeros_like(feature_values)
        binary_signal[feature_values > threshold] = 1
        
        # Use this binary signal for processing
        filtered_ecg = binary_signal
        
        # Find transitions from 0 to 1 as peaks (rising edges)
        peaks = np.where(np.diff(binary_signal) > 0)[0]
        
        # Calculate heart rate as number of peaks per minute
        if len(peaks) > 1:
            # Calculate average time between peaks
            avg_peak_interval = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
            # Convert to heart rate (beats per minute)
            hr_result = 60 * 1000 / avg_peak_interval if avg_peak_interval > 0 else 0
        else:
            hr_result = 0
        
        # Safely extract heart rate value
        heart_rate = 0  # Default value
        if isinstance(hr_result, np.ndarray) and hr_result.size > 0:
            try:
                heart_rate = hr_result[-1][0]
            except (IndexError, TypeError):
                # If heart_rate doesn't have the expected structure
                if hr_result.size > 0:
                    heart_rate = float(hr_result[-1])
                else:
                    heart_rate = 0
                    
        # Check if the latest data point is a peak
        Rpeak = False
        if len(peaks) > 0 and peaks[-1] >= len(feature_values) - 3:  # Consider a peak if it's one of the last 3 samples
            Rpeak = True
        
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
        
        # Return the heart rate, peak detection, and the output signal ID (different from input)
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
