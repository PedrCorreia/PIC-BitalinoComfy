from collections import deque
import numpy as np
from ...src.phy.eda_signal_processing import EDA

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

        # --- Remove direct registration of SCL_METRIC and SCK_METRIC ---
        # Instead, store current and last values in metadata for MetricsSignalGenerator
        # Maintain last values for scl and sck
        if not hasattr(self, '_last_scl'):
            self._last_scl = None
        if not hasattr(self, '_last_sck'):
            self._last_sck = None
        last_scl = self._last_scl
        last_sck = self._last_sck
        self._last_scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
        self._last_sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0
        # Prepare metadata for processed signal
        metadata = {
            "scl": self._last_scl,
            "last_scl": last_scl,
            "sck": self._last_sck,
            "last_sck": last_sck,
            "scl_metric_id": "SCL_METRIC",  # Add reference to SCL metric signal
            "sck_metric_id": "SCK_METRIC"   # Add reference to SCK metric signal
        }
        # If a registry is available, register the processed signal with metadata
        try:
            from ...src.registry.plot_registry import PlotRegistry
            registry = PlotRegistry.get_instance()
            processed_signal_id = "EDA_PROCESSED"
            processed_signal_data = {
                "t": viz_data[:, 0].tolist(),
                "v": viz_data[:, 1].tolist(),
                "tonic": tonic_viz.tolist(),
                "phasic": phasic_viz.tolist()
            }
            registry.register_signal(processed_signal_id, processed_signal_data, metadata)
        except Exception as e:
            pass  # Registry may not be available in all contexts

        tonic_out = tonic if output_type in ("tonic", "both") else None
        phasic_out = phasic if output_type in ("phasic", "both") else None

        return visualization_data, tonic_out, phasic_out

NODE_CLASS_MAPPINGS = {
    "EDANode": EDANode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDANode": "🖐️ EDA Processor"
}
