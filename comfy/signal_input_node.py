import numpy as np
import torch
from ..src.plot_unit import PlotUnit

class SignalInputNode:
    """
    A node that allows sending signals to the PlotUnit visualization hub.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("TENSOR",),
                "signal_name": ("STRING", {"default": "signal_1"}),
            },
            "optional": {
                "color_r": ("INT", {"default": 220, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 180, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("signal",)
    FUNCTION = "process_signal"
    CATEGORY = "signal/visualization"
    
    def __init__(self):
        # Get singleton PlotUnit instance
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        # Register as a connected node
        self.plot_unit.increment_connected_nodes()
    
    def process_signal(self, signal, signal_name="signal_1", color_r=220, color_g=180, color_b=0):
        """Process and visualize the input signal"""
        # Define signal color from inputs
        color = (color_r, color_g, color_b)
        
        # Send signal to the PlotUnit
        self.plot_unit.add_signal_data(signal, name=signal_name, color=color)
        
        # Return the original signal unmodified
        return (signal,)
    
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
