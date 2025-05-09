from collections import deque
import numpy as np
from ..src.rr_signal_processing import RR

class RRNode:
    """
    Node for processing RR (Respiratory Rate) signals.
    Provides visualization-ready data and calculates respiration rate using detected peaks.
    Buffer sizes for visualization and feature extraction are configurable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the RR node.
        - signal_deque: Deque of (timestamp, value) pairs.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.
        """
        return {
            "required": {
                "signal_deque": ("ARRAY",),
                "viz_buffer_size": ("INT", {"default": 1000}),
                "feature_buffer_size": ("INT", {"default": 5000})
            }
        }

    RETURN_TYPES = ("ARRAY", "FLOAT")
    RETURN_NAMES = ("Visualization_Data", "Respiration_Rate","Peakk")
    FUNCTION = "process_rr"
    CATEGORY = "Biosignal/RR"

    def process_rr(self, signal_deque, viz_buffer_size, feature_buffer_size):
        """
        Processes RR signal to extract visualization-ready data and respiration rate.

        Parameters:
        - signal_deque: Deque containing (timestamp, value) tuples.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.

        Returns:
        - visualization_data: Array of [timestamp, value, is_peak] rows.
        - respiration_rate: Calculated respiration rate in breaths per minute.
        """
        if not signal_deque or len(signal_deque) < 2:
             raise ValueError("Insufficient data in deque.")

        data = np.array(signal_deque)
        timestamps, values = data[:, 0], data[:, 1]

        viz_timestamps = timestamps[-viz_buffer_size:]
        viz_values = values[-viz_buffer_size:]
        viz_data = np.column_stack((viz_timestamps, viz_values))

        # Use NumpySignalProcessor for peak detection
        from ..src.signal_processing import NumpySignalProcessor
        peaks = NumpySignalProcessor.find_peaks(values[-feature_buffer_size:], fs=1000)
        peak_indices = np.intersect1d(np.arange(len(values))[-viz_buffer_size:], peaks, assume_unique=True)
        is_peak = np.zeros(viz_data.shape[0], dtype=int)
        is_peak[peak_indices - (len(values) - viz_data.shape[0])] = 1

        # Efficiently build [[[timestamp, value], is_peak], ...]
        visualization_data = np.empty((viz_data.shape[0], 2), dtype=object)
        visualization_data[:, 0] = viz_data.tolist()
        visualization_data[:, 1] = is_peak
        visualization_data = visualization_data.tolist()

        # Respiration rate from extract_respiration_rate
        rr, _ = RR.extract_respiration_rate(values[-feature_buffer_size:], fs=1000)
        rr = rr[-1]
        if visualization_data[-1][1] == 1:
            peaks = True
        else:
            peaks = False
        return visualization_data, rr,peaks

NODE_CLASS_MAPPINGS = {
    "RRNode": RRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RRNode": "🌬️ RR Processor"
}
