"""
ComfyUI Node: Comfy Signal Connector
- Connects a signal from the generator registry to the plot registry
- Takes a single signal_id as input, no output
- Acts as a workflow sink for signal routing
"""
import threading
import time
# Use local imports inside the connect method to ensure registry independence and avoid import path issues
class ComfySignalConnectorNode:
    _threads = {}
    _stop_flags = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "GEN_0"}),
                "color": ("STRING", {"default": "red", "options": ["red", "orange", "yellow", "blue"]}),
                "type": ("STRING", {"default": "raw", "options": ["raw", "processed"]}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "connect"
    CATEGORY = "Registry/SignalConn"
    OUTPUT_NODE = True

    @staticmethod
    def cast_input(signal_id):
        # Accepts both dict (from generator) and string
        if isinstance(signal_id, dict) and "signal_id" in signal_id:
            return signal_id["signal_id"]
        return signal_id

    def _background_link(self, signal_id, stop_flag, color, signal_type):
        from ...src.registry.signal_registry import SignalRegistry
        from ...src.registry.plot_registry import PlotRegistry
        color_map = {
            'red': '#FF5555',
            'orange': '#FFB86C',
            'yellow': '#F1FA8C',
            'blue': '#8BE9FD',
        }
        color_hex = color_map.get(color.lower(), '#FFFFFF')
        while not stop_flag[0]:
            gen_registry = SignalRegistry.get_instance()
            plot_registry = PlotRegistry.get_instance()
            sig = gen_registry.get_signal(signal_id)
            meta = gen_registry.get_metadata(signal_id) or {}
            meta = dict(meta)
            # Always set color, even if not present
            if not color:
                color_hex = '#FF5555'  # fallback to red
            if sig is not None:
                gen_registry.register_signal(signal_id, sig, meta)  # update registry meta
                plot_registry.register_signal(signal_id, sig, meta)
            time.sleep(0.05)  # Increased frequency

    def connect(self, signal_id, color, type):
        from ...src.registry.signal_registry import SignalRegistry
        from ...src.registry.plot_registry import PlotRegistry
        signal_id = self.cast_input(signal_id)
        signal_type = type  # avoid using 'type' as a parameter internally
        # Start background thread if not already running
        if signal_id not in self._threads or not self._threads[signal_id].is_alive():
            # Stop any previous thread for this id
            if signal_id in self._stop_flags:
                self._stop_flags[signal_id][0] = True
            stop_flag = [False]
            self._stop_flags[signal_id] = stop_flag
            t = threading.Thread(target=self._background_link, args=(signal_id, stop_flag, color, signal_type), daemon=True)
            self._threads[signal_id] = t
            t.start()
        return ()

    def __del__(self):
        # Clean up all threads
        for stop_flag in self._stop_flags.values():
            stop_flag[0] = True
        self._threads.clear()
        self._stop_flags.clear()

NODE_CLASS_MAPPINGS = {"ComfySignalConnectorNode": ComfySignalConnectorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfySignalConnectorNode": "Comfy Signal Connector"}
