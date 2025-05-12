import numpy as np
import torch
from ..src.plot_unit import PlotUnit

class PlotUnitNode:
    """
    A visualization hub node that displays signals in a persistent window.
    This node has no inputs or outputs and operates independently through its GUI interface.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
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
    
    def run_visualization_hub(self):
        """
        Run the visualization hub. This function just ensures the visualization
        window is running and returns nothing.
        """
        # The actual work happens in the PlotUnit thread
        # We just make sure it's running
        if not self.plot_unit.initialized:
            self.plot_unit.start()
            
        # Return empty tuple (no outputs)
        return ()
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Unregister as a connected node
            plot_unit = PlotUnit.get_instance()
            plot_unit.decrement_connected_nodes()
        except:
            # This might fail during shutdown, so we'll just ignore errors
            pass

# Node registration - moved to __init__.py
NODE_CLASS_MAPPINGS = {
    "PlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlotUnitNode": "Plot Unit"
}
