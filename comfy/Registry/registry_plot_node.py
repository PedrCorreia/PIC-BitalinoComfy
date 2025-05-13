import sys
import os
import uuid
import importlib
# Import from plot modules
from ...src.plot.plot_unit import PlotUnit
from ...src.plot.plot_registry import PlotRegistry
from ...src.plot.plot_registry_integration import PlotRegistryIntegration

class RegistryPlotNode:
    """
    A visualization node that connects to a centralized plot registry.
    This node visualizes signals from the registry in a Pygame window.
    
    NOTE: This node uses the older architecture and may not handle 
    string validation properly. For the newer unified architecture,
    use the PlotUnitNode alongside the SignalInputNode which includes
    proper signal validation and error handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": ("BOOLEAN", {"default": False, "label_on": "Reset Plots", "label_off": "Keep Plots"}),
            },
            "optional": {
                "signal_id": ("STRING", {"default": "", "multiline": False}),
                "clear_registry": ("BOOLEAN", {"default": False, "label_on": "Clear Registry", "label_off": "Keep Registry"}),
                "auto_reset": ("BOOLEAN", {"default": False, "label_on": "Auto-Reset on Run", "label_off": "No Auto-Reset"}),
            }
        }
    
    RETURN_TYPES = ()  # No outputs
    OUTPUT_NODE = True
    FUNCTION = "run_visualization"
    CATEGORY = "signal/visualization"
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"registry_plot_{str(uuid.uuid4())[:8]}"
        
        # Get singletons
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        self.registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Connect plot unit to integration
        self.integration.connect_plot_unit(self.plot_unit)
        
        # Register this node with the integration
        self.integration.register_node(self.node_id)
        print(f"[Registry Plot] Node {self.node_id} initialized")
    
    def __del__(self):
        """Clean up when this node is deleted"""
        try:
            # Disconnect node from integration
            self.integration.disconnect_node(self.node_id)
            print(f"[Registry Plot] Node {self.node_id} disconnected")
        except:
            pass
    
    def run_visualization(self, reset=False, signal_id="", clear_registry=False, auto_reset=False):
        """
        Run visualization of signals from the registry
        
        Args:
            reset: Whether to reset the plots
            signal_id: Optional specific signal to visualize
            clear_registry: Whether to clear the registry
            auto_reset: Whether to auto-reset on each run
        """
        print(f"[Registry Plot] Node {self.node_id} executing")
        
        # Check if we need to reset everything
        if clear_registry:
            print("[Registry Plot] Clearing registry")
            self.integration.reset()
        
        # Reset plots if requested
        if reset or auto_reset:
            print("[Registry Plot] Resetting plots")
            self.plot_unit.clear_plots()
        
        # Connect to specific signal if provided
        if signal_id and signal_id.strip():
            print(f"[Registry Plot] Connecting to signal: {signal_id}")
            self.integration.connect_node_to_signal(self.node_id, signal_id)
        
        # The visualization happens in background threads
        return ()

# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "RegistryPlotNode": RegistryPlotNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegistryPlotNode": "⚠️ Registry Plot (Legacy)"
}
