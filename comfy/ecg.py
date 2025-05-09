from collections import deque
import numpy as np
from ..src.ecg_signal_processing import ECG

class ECGNode:
    """
    Node for processing ECG (Electrocardiogram) signals.
    Provides visualization-ready data and calculates heart rate using detected peaks.
    Buffer sizes for visualization and feature extraction are configurable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the ECG node.
        - signal_deque: Deque of (timestamp, value) pairs.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.
        """
        return {
            "required": {
                "signal_deque": ("DEQUE",),
                "viz_buffer_size": ("INT", {"default": 1000}),
                "feature_buffer_size": ("INT", {"default": 5000})
            }
        }

    RETURN_TYPES = ("ARRAY", "FLOAT")
    RETURN_NAMES = ("Visualization_Data", "Heart_Rate")
    FUNCTION = "process_ecg"
    CATEGORY = "Biosignal/ECG"

    def process_ecg(self, signal_deque, viz_buffer_size, feature_buffer_size):
        """
        Processes ECG signal to extract visualization-ready data and heart rate.

        Parameters:
        - signal_deque: Deque containing (timestamp, value) tuples.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.

        Returns:
        - visualization_data: Array of [timestamp, value, is_peak] rows.
        - heart_rate: Calculated heart rate in beats per minute (bpm).
        """
        if not signal_deque or len(signal_deque) < 2:
            raise ValueError("Insufficient data in deque.")

        data = np.array(signal_deque)
        timestamps, values = data[:, 0], data[:, 1]

        viz_data = data[-viz_buffer_size:]

        # Use NumpySignalProcessor for peak detection
        from ..src.signal_processing import NumpySignalProcessor
        peaks = NumpySignalProcessor.find_peaks(values[-feature_buffer_size:], fs=1000)
        peak_indices = np.intersect1d(np.arange(len(values))[-viz_buffer_size:], peaks, assume_unique=True)
        is_peak = np.zeros_like(viz_data[:, 1], dtype=int)
        is_peak[peak_indices - (len(values) - len(viz_data))] = 1
        visualization_data = np.column_stack((viz_data[:, 0], viz_data[:, 1], is_peak))

        # Heart rate from extract_heart_rate
        heart_rate = ECG.extract_heart_rate(values[-feature_buffer_size:], fs=1000, r_peaks=peaks)

        return visualization_data, heart_rate

NODE_CLASS_MAPPINGS = {
    "ECGNode": ECGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ECGNode": "💓 ECG Processor"
}
