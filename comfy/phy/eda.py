from collections import deque
import numpy as np
from ...src.phy.eda_signal_processing import EDA
import time
import threading

class EDANode:
    """
    Node for processing EDA (Electrodermal Activity) signals.
    Extracts tonic and phasic components, provides visualization-ready data,
    and allows selection of which components to output.
    Robust and registry-ready for MetricsView integration.
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
                "feature_buffer_size": ("INT", {"default": 5000}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("ARRAY", "ARRAY", "ARRAY")
    RETURN_NAMES = ("Visualization_Data", "Tonic", "Phasic")
    FUNCTION = "process_eda"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")
    def _background_process(self, input_signal_id, output_type, viz_buffer_size, feature_buffer_size, stop_flag, output_signal_id):
        from ...src.registry.plot_registry import PlotRegistry
        registry = PlotRegistry.get_instance()
        fs = 1000
        viz_buffer_size = int(viz_buffer_size)
        feature_buffer_size = int(feature_buffer_size)
        while not stop_flag[0]:
            signal_data = registry.get_signal(input_signal_id)
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                time.sleep(0.01)
                continue
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            if len(timestamps) < 2 or len(values) < 2:
                time.sleep(0.01)
                continue
            data = np.column_stack((timestamps, values))
            viz_data = data[-viz_buffer_size:]
            tonic, phasic = EDA.extract_tonic_phasic(values[-feature_buffer_size:], fs=fs)
            tonic_viz = tonic[-viz_buffer_size:]
            phasic_viz = phasic[-viz_buffer_size:]
            visualization_data = np.column_stack((viz_data[:, 0], viz_data[:, 1], tonic_viz, phasic_viz))
            # Maintain last values for scl and sck
            if not hasattr(self, '_last_scl'):
                self._last_scl = None
            if not hasattr(self, '_last_sck'):
                self._last_sck = None
            last_scl = self._last_scl
            last_sck = self._last_sck
            self._last_scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            self._last_sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0
            metadata = {
                "scl": self._last_scl,
                "last_scl": last_scl,
                "sck": self._last_sck,
                "last_sck": last_sck,
                "scl_metric_id": "SCL_METRIC",
                "sck_metric_id": "SCK_METRIC"
            }
            processed_signal_id = output_signal_id
            processed_signal_data = {
                "t": viz_data[:, 0].tolist(),
                "v": viz_data[:, 1].tolist(),
                "tonic": tonic_viz.tolist(),
                "phasic": phasic_viz.tolist()
            }
            registry.register_signal(processed_signal_id, processed_signal_data, metadata)
            time.sleep(0.01)

    def process_eda(self, input_signal_id="", output_type="both", viz_buffer_size=1000, feature_buffer_size=5000, output_signal_id="EDA_PROCESSED", enabled=True):
        if not enabled:
            return None, None, output_signal_id
        signal_id = output_signal_id
        if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
            # Return last known values (None for now, as EDA is not a metric node)
            return None, None, None
        stop_flag = [False]
        self._stop_flag = stop_flag
        thread = threading.Thread(
            target=self._background_process,
            args=(input_signal_id, output_type, viz_buffer_size, feature_buffer_size, stop_flag, output_signal_id),
            daemon=True
        )
        self._processing_thread = thread
        thread.start()
        return None, None, output_signal_id

NODE_CLASS_MAPPINGS = {
    "EDANode": EDANode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDANode": "🖐️ EDA Processor"
}
