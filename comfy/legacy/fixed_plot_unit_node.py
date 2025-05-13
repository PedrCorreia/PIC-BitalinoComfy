import sys
import o

# Import from the fixed module
from src.plot.fixed_plot_unit import PlotUnit

# Import needed libraries for signal registry
import importlib
import numpy as np
import torch
import time

# Try to import the signal registry
try:
    from .mock_signal_node import SignalRegistry
except ImportError:
    # Define a minimal registry if import fails
    class SignalRegistry:
        _instance = None
        
        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
        
        def __init__(self):
            self.signals = {}
            
        def get_all_signals(self):
            return list(self.signals.keys())

class PlotUnitNode:
    """
    A visualization hub node that displays signals in a persistent window.
    This node has no inputs or outputs and operates independently through its GUI interface.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": ("BOOLEAN", {"default": False, "label_on": "Reset Plots", "label_off": "Keep Plots"}),
            },
            "optional": {
                "clear_all_signals": ("BOOLEAN", {"default": False, "label_on": "Clear Registry", "label_off": "Keep Registry"}),
                "auto_reset": ("BOOLEAN", {"default": False, "label_on": "Auto-Reset on Run", "label_off": "No Auto-Reset"}),
            }
        }
    
    RETURN_TYPES = ()  # No outputs
    OUTPUT_NODE = True
    FUNCTION = "run_visualization_hub"
    CATEGORY = "signal/visualization"
    
    def __init__(self):
        # Get singleton PlotUnit instance
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        # Register as a connected node
        self.plot_unit.increment_connected_nodes()
        print("[DEBUG-PLOT] PlotUnitNode initialized")
    
    def run_visualization_hub(self, reset=False, clear_all_signals=False, auto_reset=False):
        """
        Run the visualization hub. This function ensures the visualization
        window is running and can optionally reset it.
        """
        print("[DEBUG-PLOT] PlotUnitNode.run_visualization_hub called")
        print(f"[DEBUG-PLOT] Reset button value: {reset}")
        print(f"[DEBUG-PLOT] Clear registry button value: {clear_all_signals}")
        print(f"[DEBUG-PLOT] Auto-reset button value: {auto_reset}")
        
        # The actual work happens in the PlotUnit thread
        # We just make sure it's running
        if not self.plot_unit.initialized:
            self.plot_unit.start()
            print("[DEBUG-PLOT] PlotUnit started")
        
        # Reset visualization if requested explicitly or via auto-reset
        if reset or auto_reset:
            print("[DEBUG-PLOT] RESET REQUESTED! Clearing plots...")
            
            # Clear plots in PlotUnit
            self.plot_unit.clear_plots()
        
        # Clear signal registry if requested
        registry_signals = SignalRegistry().get_all_signals()
        print(f"[DEBUG-PLOT] Found {len(registry_signals)} signals in registry")

        # Show our status message
        print("[DEBUG-PLOT] PlotUnitNode processing complete")
        
        return ()

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FixedPlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FixedPlotUnitNode": "Fixed Plot Unit Node"
}
