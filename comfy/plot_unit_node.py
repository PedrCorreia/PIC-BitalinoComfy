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
        
        # Patch the PlotUnit class if it doesn't have clear_plots method
        if not hasattr(self.plot_unit, 'clear_plots'):
            print("[DEBUG-PLOT] Adding clear_plots method to PlotUnit")
            setattr(self.plot_unit, 'clear_plots', self._patch_clear_plots)
    
    def _patch_clear_plots(self):
        """Patch method for the PlotUnit if clear_plots isn't available"""
        print("[DEBUG-PLOT] Using patched clear_plots method")
        try:
            # Attempt to clear data structures in PlotUnit
            if hasattr(self.plot_unit, 'signals'):
                self.plot_unit.signals = {}
                print("[DEBUG-PLOT] Cleared signals in PlotUnit")
            
            # Try to access and clear any matplotlib figures
            if hasattr(self.plot_unit, 'figure') and self.plot_unit.figure is not None:
                import matplotlib.pyplot as plt
                plt.figure(self.plot_unit.figure.number)
                plt.clf()
                print("[DEBUG-PLOT] Cleared matplotlib figure")
        except Exception as e:
            print(f"[ERROR-PLOT] Error in patched clear_plots: {str(e)}")
    
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
            if hasattr(self.plot_unit, 'clear_plots'):
                try:
                    self.plot_unit.clear_plots()
                    print("[DEBUG-PLOT] Plots successfully cleared")
                except Exception as e:
                    print(f"[ERROR-PLOT] Error clearing plots: {str(e)}")
            else:
                print("[DEBUG-PLOT] PlotUnit has no clear_plots method, attempting alternative clear")
                # Try alternative method if clear_plots doesn't exist
                self._patch_clear_plots()
        
        # Clear signal registry if requested
        if clear_all_signals:
            print("[DEBUG-PLOT] CLEAR REGISTRY REQUESTED! Resetting signal registry...")
            # Reset the signal registry
            from .mock_signal_node import SignalRegistry
            SignalRegistry.reset()
            print("[DEBUG-PLOT] Signal registry reset complete")
        
        print("[DEBUG-PLOT] PlotUnitNode processing complete")
        # Return empty tuple (no outputs)
        return ()
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Unregister as a connected node
            plot_unit = PlotUnit.get_instance()
            plot_unit.decrement_connected_nodes()
            print("[DEBUG-PLOT] PlotUnitNode cleanup complete")
        except:
            # This might fail during shutdown, so we'll just ignore errors
            pass

# Node registration
NODE_CLASS_MAPPINGS = {
    "PlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlotUnitNode": "Plot Unit"
}
