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
                "over": ("BOOLEAN", {"default": False}),
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

    def _background_link(self, signal_id, stop_flag, color, signal_type, over):
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
            meta['color'] = color_hex
            meta['type'] = signal_type
            # Set the overlay flag
            meta['over'] = over
            
            # Handle EDA signals and other components with overlay
            is_eda_processed = signal_id.upper().startswith('EDA') and signal_type == 'processed'
            
            # Remove debug prints for production
            # print(f"[DEBUG] ConnectorNode: signal={signal_id}, type={signal_type}, over={over}, is_eda={is_eda_processed}")
            # print(f"[DEBUG] Signal meta before registry update: {meta}")
            
            if sig is not None:
                # print(f"[DEBUG] Signal structure: {list(sig.keys()) if isinstance(sig, dict) else 'non-dict'}")
                # if isinstance(sig, dict) and 'phasic_norm' in sig:
                #     print(f"[DEBUG] Signal has phasic_norm directly in signal data")
                
                gen_registry.register_signal(signal_id, sig, meta)  # update registry meta
                already = plot_registry.get_signal_metadata(signal_id)
                # print(f"[DEBUG] Already in registry: {signal_id} = {already}")
                # print(f"[DEBUG] Existing metadata keys: {list(already.keys()) if already else 'None'}")
                  # Modify signal registration behavior to support proper overlay
                if is_eda_processed:
                    # Always set over=True for EDA processed signals to ensure they get overlaid
                    meta['over'] = True
                    
                    # print(f"[DEBUG] Registering EDA signal: signal_id={signal_id}, over={meta['over']}")
                    
                    # Check if the signal data has the necessary components for overlay
                    has_components = False
                    if isinstance(sig, dict):
                        if 'phasic_norm' in sig and 'tonic_norm' in sig:
                            # print(f"[DEBUG] EDA signal has components in signal data")
                            has_components = True
                        elif 'tonic_norm' in meta and 'phasic_norm' in meta:
                            # print(f"[DEBUG] EDA signal has components in metadata")
                            has_components = True
                    
                    # Make sure important component data is in metadata for the drawing function
                    if isinstance(sig, dict) and 'phasic_norm' in sig and 'phasic_norm' not in meta:
                        meta['phasic_norm'] = sig['phasic_norm']
                        meta['tonic_norm'] = sig['tonic_norm']
                        # print(f"[DEBUG] Copied component data from signal to metadata")
                    
                    # Always register EDA signals with overlay flag
                    plot_registry.register_signal(signal_id, sig, meta)
                    
                    # Double-check if overlay flag was properly registered
                    check_meta = plot_registry.get_signal_metadata(signal_id)
                    # print(f"[DEBUG] After EDA registration, over={check_meta.get('over')}, has_components={('phasic_norm' in check_meta) and ('tonic_norm' in check_meta)}")
                else:
                    # For non-EDA signals, always register
                    # print(f"[DEBUG] Registering regular signal: {signal_id}")
                    plot_registry.register_signal(signal_id, sig, meta)
            time.sleep(0.05)  # Increased frequency

    def connect(self, signal_id, color, type, over):
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
            t = threading.Thread(target=self._background_link, args=(signal_id, stop_flag, color, signal_type, over), daemon=True)
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
