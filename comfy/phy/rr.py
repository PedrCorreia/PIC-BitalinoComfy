from collections import deque
import numpy as np
from ...src.phy.rr_signal_processing import RR

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

    RETURN_TYPES = ("ARRAY", "FLOAT", "ARRAY", "FLOAT", "FLOAT")
    RETURN_NAMES = ("Visualization_Data", "Respiration_Rate", "Peak_Positions", "Ymin", "Ymax")
    FUNCTION = "process_rr"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"

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
        - peak_positions: List of peak positions (timestamps and values).
        - ymin: Minimum value in the visualization buffer.
        - ymax: Maximum value in the visualization buffer.
        """
        if signal_deque is None or len(signal_deque) < max(viz_buffer_size, feature_buffer_size):
            return [], 0.0, [], 0.0, 0.0

        data = signal_deque
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        timestamps = data[:, 0].astype(float)
        raw_values = data[:, 1].astype(float)



        # Only preprocess the region needed for feature extraction
        feature_raw_values = raw_values[-feature_buffer_size:]
        feature_timestamps = timestamps[-feature_buffer_size:]
        values = RR.preprocess_signal(feature_raw_values, fs=100)

        # Visualization buffer is always the last N points of the feature buffer
        viz_len = min(viz_buffer_size, len(values))
        viz_timestamps = feature_timestamps[-viz_len:]
        viz_values = values[-viz_len:]
        viz_data = np.column_stack((viz_timestamps, viz_values))

        from ...src.utils.signal_processing import NumpySignalProcessor
        peaks = NumpySignalProcessor.find_peaks(values, fs=100)
        # Only keep peaks that are within the visualization buffer
        peak_indices = peaks[peaks >= (len(values) - viz_len)] - (len(values) - viz_len)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < viz_len)]
        is_peak = np.zeros(viz_len, dtype=int)
        is_peak[peak_indices] = 1

        # Build visualization data
        visualization_data = np.empty((viz_len, 2), dtype=object)
        visualization_data[:, 0] = viz_data.tolist()
        visualization_data[:, 1] = is_peak
        visualization_data = visualization_data.tolist()

        # Prepare peak positions for plotting
        peak_positions = []
        if len(peak_indices) > 0:
            peak_positions = viz_data[peak_indices].tolist()

        # Respiration rate from extract_respiration_rate
        rr, _ = RR.extract_respiration_rate(values, fs=100)

        # Y-limits for visualization
        ymin = float(np.min(viz_values)) if viz_values.size > 0 else 0.0
        ymax = float(np.max(viz_values)) if viz_values.size > 0 else 0.0

        return visualization_data, rr, peak_positions, ymin, ymax

NODE_CLASS_MAPPINGS = {
    "RRNode": RRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RRNode": "🌬️ RR Processor"
}

