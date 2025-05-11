from collections import deque
import numpy as np
from ..src.eda_signal_processing import EDA

class EDANode:
    """
    Node for processing EDA (Electrodermal Activity) signals.
    Extracts tonic and phasic components, provides visualization-ready data,
    and allows selection of which components to output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the EDA node.
        - signal_deque: Deque of (timestamp, value) pairs.
        - output_type: Which components to output ("tonic", "phasic", or "both").
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.
        """
        return {
            "required": {
                "signal_deque": ("DEQUE",),
                "output_type": ("STRING", {
                    "default": "both",
                    "choices": ["tonic", "phasic", "both"]
                }),
                "viz_buffer_size": ("INT", {"default": 1000}),
                "feature_buffer_size": ("INT", {"default": 5000})
            }
        }

    RETURN_TYPES = ("ARRAY", "ARRAY", "ARRAY")
    RETURN_NAMES = ("Visualization_Data", "Tonic", "Phasic")
    FUNCTION = "process_eda"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"

    def process_eda(self, signal_deque, output_type, viz_buffer_size, feature_buffer_size):
        """
        Processes EDA signal to extract tonic and phasic components.

        Parameters:
        - signal_deque: Deque containing (timestamp, value) tuples.
        - output_type: "tonic", "phasic", or "both" (controls which outputs are filled).
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.

        Returns:
        - visualization_data: Array of [timestamp, value, tonic, phasic] rows.
        - tonic: Tonic component array or None.
        - phasic: Phasic component array or None.
        """
        if not signal_deque or len(signal_deque) < 2:
            raise ValueError("Insufficient data in deque.")

        data = np.array(signal_deque)
        timestamps, values = data[:, 0], data[:, 1]

        viz_data = data[-viz_buffer_size:]

        tonic, phasic = EDA.extract_tonic_phasic(values[-feature_buffer_size:], fs=1000)
        tonic_viz = tonic[-viz_buffer_size:]
        phasic_viz = phasic[-viz_buffer_size:]
        visualization_data = np.column_stack((viz_data[:, 0], viz_data[:, 1], tonic_viz, phasic_viz))

        tonic_out = tonic if output_type in ("tonic", "both") else None
        phasic_out = phasic if output_type in ("phasic", "both") else None

        return visualization_data, tonic_out, phasic_out

NODE_CLASS_MAPPINGS = {
    "EDANode": EDANode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDANode": "🖐️ EDA Processor"
}
