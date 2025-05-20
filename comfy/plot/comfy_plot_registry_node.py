"""
ComfyUI Node: Comfy Plot Registry Node
- Visualizes all signals currently in the plot registry
- No inputs, no outputs (pure output node)
- Auto-connects to the plot registry
- Initializes the UI like robust_view_switcher_registry_based.py
"""
from ...src.registry.plot_registry import PlotRegistry
import threading

class ComfyPlotRegistryNode:
    _stop_event = threading.Event()

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ()
    FUNCTION = "plot"
    CATEGORY = "Registry/Plot"
    OUTPUT_NODE = True

    def plot(self):
        # Launch the robust registry-based visualization UI in a background thread if not already running
        def run_ui_loop():
            import sys, os
            import pygame
            while not self._stop_event.is_set():
                if not pygame.get_init():
                    pygame.init()
                # Add the PIC-2025 root to sys.path so robust_view_switcher_registry_based can be imported
                pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                if pkg_root not in sys.path:
                    sys.path.insert(0, pkg_root)
                from robust_view_switcher_registry_based import main as robust_main
                try:
                    robust_main(start_generators=False, stop_event=self._stop_event)
                except SystemExit:
                    # Window closed, restart UI unless process is exiting
                    import time
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    print(f"[ComfyPlotRegistryNode] UI crashed: {e}")
                    import time
                    time.sleep(1)
                    continue
                break
            try:
                pygame.quit()
            except Exception:
                pass
        # Only start UI if not already running (avoid multiple windows)
        if not hasattr(self, '_ui_thread') or not self._ui_thread.is_alive():
            self._stop_event.clear()
            self._ui_thread = threading.Thread(target=run_ui_loop, daemon=True)
            self._ui_thread.start()
        # No output
        return ()

    def __del__(self):
        self._stop_event.set()

NODE_CLASS_MAPPINGS = {"ComfyPlotRegistryNode": ComfyPlotRegistryNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyPlotRegistryNode": "Comfy Plot Registry Node"}
