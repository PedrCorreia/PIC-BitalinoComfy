import numpy as np
import torch
from ...src.plot.plot_unit import PlotUnit
from ...src.hubs.signal_registry import SignalRegistry

class SignalInputNode:
    """
    A node that acts as a switch to connect or disconnect a signal from the registry to the plot hub.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "EDA"}),  # Signal ID from registry
                "enabled": ("BOOLEAN", {"default": True}),    # Switch to enable/disable plotting
            },
            "optional": {
                "color_r": ("INT", {"default": 220, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 180, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_id",)
    FUNCTION = "process_signal"
    CATEGORY = "Pedro_PIC/🧰 Visualization"
    OUTPUT_NODE = True
    
    def __init__(self):
        # Get singleton PlotUnit instance
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        print("[DEBUG-INPUT] SignalInputNode initialized")
        print("[DEBUG-INPUT] PlotUnit started")
        # Register as a connected node
        self.plot_unit.increment_connected_nodes()
        print("[DEBUG-INPUT] Registered as connected node")
    
    def process_signal(self, signal_id, enabled, color_r=220, color_g=180, color_b=0):
        """Process and visualize the input signal if enabled"""
        # Define signal color from inputs
        color = (color_r, color_g, color_b)
        
        if not enabled:
            print(f"[DEBUG-INPUT] Signal '{signal_id}' is disabled, not plotting")
            return (signal_id,)
        
        # Handle comma-separated signal IDs from SynthNode
        if ',' in signal_id:
            print(f"[DEBUG-INPUT] Received multiple signals: {signal_id}, using first one")
            signal_id = signal_id.split(',')[0]
        
        print(f"[DEBUG-INPUT] Processing signal: '{signal_id}' with color {color}")
        
        # Get signal from registry
        registry = SignalRegistry.get_instance()
        print(f"[DEBUG-INPUT] Got registry instance, available IDs: {list(registry.signals.keys())}")
        
        signal = registry.get_signal(signal_id)
        
        if signal is not None:
            print(f"[DEBUG-INPUT] Retrieved signal '{signal_id}' from registry, shape: {signal.shape}")
            # Send signal to the PlotUnit
            try:
                self.plot_unit.add_signal_data(signal, name=signal_id, color=color)
                print(f"[DEBUG-INPUT] Signal '{signal_id}' sent to visualization")
            except Exception as e:
                print(f"[ERROR-INPUT] Failed to send signal to PlotUnit: {str(e)}")
        else:
            print(f"[WARNING-INPUT] Signal '{signal_id}' not found in registry - nothing to visualize")
        
        # Return the signal ID
        return (signal_id,)
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Unregister as a connected node
            plot_unit = PlotUnit.get_instance()
            plot_unit.decrement_connected_nodes()
        except:
            # This might fail during shutdown, so we'll just ignore errors
            pass

# Node registration
NODE_CLASS_MAPPINGS = {
    "SignalInputNode": SignalInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalInputNode": "Signal Input"
}
