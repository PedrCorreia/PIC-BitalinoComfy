import numpy as np
import torch
import uuid
from ...src.plot.plot_unit import PlotUnit
from ...src.plot.plot_registry import PlotRegistry
from ...src.plot.plot_registry_integration import PlotRegistryIntegration

class PlotUnitNode:
    """
    A visualization hub node that displays signals from the PlotRegistry in a persistent window.
    
    This node follows the proper architecture:
    1. Connects to the PlotRegistry via the PlotRegistryIntegration
    2. Displays signals that have been registered in the PlotRegistry
    3. Works together with SignalInputNode, which connects signals from SignalRegistry to PlotRegistry
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
        # Generate a unique ID for this node
        self.node_id = f"plot_unit_{str(uuid.uuid4())[:8]}"
        
        # Get singleton instances
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        self.registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Connect plot unit to integration
        self.integration.connect_plot_unit(self.plot_unit)
        
        # Register this node with the integration
        self.integration.register_node(self.node_id)
        
        print(f"[Plot Unit] Node {self.node_id} initialized")
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect node from integration
            self.integration.disconnect_node(self.node_id)
            print(f"[Plot Unit] Node {self.node_id} disconnected")
        except:
            # This might fail during shutdown, so we'll just ignore errors
            pass
    def run_visualization_hub(self, reset=False, signal_id="", clear_all_signals=False, auto_reset=False):
        """
        Run the visualization hub. This function ensures the visualization
        window is running and can optionally reset it.
        """
        print(f"[Plot Unit] Node {self.node_id} executing")
        
        # Reset plots if requested explicitly or via auto-reset
        if reset or auto_reset:
            print("[Plot Unit] Resetting plots")
            self.plot_unit.clear_plots()
        
        # Clear all signals if requested
        if clear_all_signals:
            print("[Plot Unit] Clearing registry")
            self.integration.reset()
            
        # Connect to specific signal if provided
        # Ensure signal_id is a string before checking if it's non-empty
        if isinstance(signal_id, str) and signal_id.strip():
            print(f"[Plot Unit] Connecting to signal: {signal_id}")
            self.integration.connect_node_to_signal(self.node_id, signal_id)
        elif signal_id is not None and not isinstance(signal_id, str):
            print(f"[Plot Unit] Warning: Invalid signal ID type: {type(signal_id)}. Expected string.")
        else:
            # If no specific signal ID, visualize all signals from the registry
            signals = self.registry.get_all_signals()
            for signal_id in signals:
                print(f"[Plot Unit] Visualizing signal: {signal_id}")
                self.integration.connect_node_to_signal(self.node_id, signal_id)
        # The visualization happens in background threads
        return ()

# Node registration
NODE_CLASS_MAPPINGS = {
    "PlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlotUnitNode": "Plot Unit"
}
