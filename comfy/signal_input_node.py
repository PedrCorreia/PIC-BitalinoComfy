import numpy as np
import torch
from ..src.plot_unit import PlotUnit
from .mock_signal_node import SignalRegistry

class SignalInputNode:
    """
    A node that allows sending signals to the PlotUnit visualization hub.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_name": ("STRING", {"default": "signal1"}),
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
    CATEGORY = "signal/visualization"
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
    
    def process_signal(self, signal_name, color_r=220, color_g=180, color_b=0):
        """Process and visualize the input signal"""
        # Define signal color from inputs
        color = (color_r, color_g, color_b)
        print(f"[DEBUG-INPUT] Processing signal: '{signal_name}' with color {color}")
        
        # Get signal from registry
        registry = SignalRegistry.get_instance()
        print(f"[DEBUG-INPUT] Got registry instance, available IDs: {list(registry.signals.keys())}")
        
        signal = registry.get_signal(signal_name)
        
        if signal is not None:
            print(f"[DEBUG-INPUT] Retrieved signal '{signal_name}' from registry, shape: {signal.shape}")
            # Send signal to the PlotUnit
            try:
                self.plot_unit.add_signal_data(signal, name=signal_name, color=color)
                print(f"[DEBUG-INPUT] Signal '{signal_name}' sent to visualization")
            except Exception as e:
                print(f"[ERROR-INPUT] Failed to send signal to PlotUnit: {str(e)}")
        else:
            print(f"[WARNING-INPUT] Signal '{signal_name}' not found in registry - nothing to visualize")
        
        # Return the signal ID
        return (signal_name,)
    
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
