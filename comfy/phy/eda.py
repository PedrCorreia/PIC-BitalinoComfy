from collections import deque
import numpy as np
from ...src.phy.eda_signal_processing import EDA
from ...src.utils.signal_processing import NumpySignalProcessor
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
        Defines the input types for the EDA node, matching ECG and RR nodes for consistency.
        - input_signal_id: Signal ID for data from registry.
        - show_peaks: Control peak visualization in registry (optional, for UI consistency).
        - output_signal_id: ID for the processed signal in registry.
        - enabled: Boolean to enable or disable processing.
        """
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),
                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "EDA_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("SCL", "SCK", "Signal_ID")
    FUNCTION = "process_eda"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        from ...src.registry.plot_registry import PlotRegistry
        registry = PlotRegistry.get_instance()
        fs = 1000
        viz_buffer_size = 1000
        feature_buffer_size = 5000
        max_frequency_interest = 10  # EDA: slowest component of interest
        nyquist_fs = fs / 2
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        scl_history = []
        sck_history = []
        history_len = 10
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
            # --- Convert ADC to μS if needed ---
            # Heuristic: if max value > 20, likely ADC (10-bit: max 1023), else already μS
            if np.max(values) > 20:
                values = EDA.convert_adc_to_eda(values)
            # --- Decimate for anti-aliasing if needed ---
            if use_decimation and len(timestamps) > 1:
                decimated_values = NumpySignalProcessor.robust_decimate(values, decimation_factor)
                decimated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=len(decimated_values))
                values = decimated_values
                timestamps = decimated_timestamps
                fs_eff = fs / decimation_factor
            else:
                fs_eff = fs
            data = np.column_stack((timestamps, values))
            viz_data = data[-viz_buffer_size:]
            tonic, phasic = EDA.extract_tonic_phasic(values[-feature_buffer_size:], fs=fs_eff)
            tonic_viz = tonic[-viz_buffer_size:]
            phasic_viz = phasic[-viz_buffer_size:]
            # Normalize tonic and phasic for overlay plotting
            def normalize(arr):
                arr = np.asarray(arr)
                if arr.size == 0:
                    return arr
                minv, maxv = np.min(arr), np.max(arr)
                return (arr - minv) / (maxv - minv) if maxv > minv else arr * 0
            tonic_norm = normalize(tonic_viz)
            phasic_norm = normalize(phasic_viz)
            # Maintain last values for scl and sck, and keep history for averaging
            scl = float(np.mean(tonic_viz)) if len(tonic_viz) > 0 else 0.0
            sck = float(np.mean(phasic_viz)) if len(phasic_viz) > 0 else 0.0
            scl_history.append(scl)
            sck_history.append(sck)
            if len(scl_history) > history_len:
                scl_history.pop(0)
            if len(sck_history) > history_len:
                sck_history.pop(0)
            avg_scl = float(np.mean(scl_history)) if scl_history else 0.0
            avg_sck = float(np.mean(sck_history)) if sck_history else 0.0
            # Detect SCR events in phasic (using a simple threshold, e.g., 0.01 μS)
            scr_event_indices = EDA.detect_events(phasic_viz, threshold=0.01)
            scr_frequency = (len(scr_event_indices) / (viz_buffer_size / fs_eff)) * 60 if viz_buffer_size > 0 else 0.0  # events per minute
            # Store for node output
            self._last_scl = avg_scl
            self._last_sck = avg_sck
            metadata = {
                "scl": scl,
                "sck": sck,
                "avg_scl": avg_scl,
                "avg_sck": avg_sck,
                "scr_frequency": scr_frequency,
                "scl_metric_id": "SCL_METRIC",
                "sck_metric_id": "SCK_METRIC",
                "tonic_norm": tonic_norm.tolist(),
                "phasic_norm": phasic_norm.tolist(),
                "timestamps": viz_data[:, 0].tolist(),
                "raw_tonic": tonic_viz.tolist(),
                "raw_phasic": phasic_viz.tolist()
            }
            processed_signal_id = output_signal_id
            processed_signal_data = {
                "t": viz_data[:, 0].tolist(),
                "v": viz_data[:, 1].tolist(),
                "tonic": tonic_viz.tolist(),
                "phasic": phasic_viz.tolist(),
                "tonic_norm": tonic_norm.tolist(),
                "phasic_norm": phasic_norm.tolist()
            }
            registry.register_signal(processed_signal_id, processed_signal_data, metadata)
            time.sleep(0.01)

    def process_eda(self, input_signal_id="", show_peaks=True, output_signal_id="EDA_PROCESSED", enabled=True):
        if not enabled:
            return 0.0, 0.0, output_signal_id
        signal_id = output_signal_id
        if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
            scl = getattr(self, '_last_scl', 0.0)
            sck = getattr(self, '_last_sck', 0.0)
            return scl, sck, output_signal_id
        stop_flag = [False]
        self._stop_flag = stop_flag
        thread = threading.Thread(
            target=self._background_process,
            args=(input_signal_id, show_peaks, stop_flag, output_signal_id),
            daemon=True
        )
        self._processing_thread = thread
        thread.start()
        return 0.0, 0.0, output_signal_id

NODE_CLASS_MAPPINGS = {
    "EDANode": EDANode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDANode": "🖐️ EDA Processor"
}
