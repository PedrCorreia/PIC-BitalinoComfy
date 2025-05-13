import numpy as np
import torch
import uuid
import time
from ...src.plot.plot_unit import PlotUnit
from ...src.registry.plot_registry import PlotRegistry
from ...src.registry.plot_registry_integration import PlotRegistryIntegration

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
                "performance_mode": ("BOOLEAN", {"default": False, "label_on": "Performance Mode", "label_off": "Quality Mode"}),
                "signal_id": ("STRING", {"default": ""}),
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
        
        # Make sure the plot_unit has these methods
        if not hasattr(self.plot_unit, 'update'):
            self.plot_unit.update = self._update_fallback
        if not hasattr(self.plot_unit, 'clear_plots'):
            self.plot_unit.clear_plots = self._clear_plots_fallback
        
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
            
    def _update_fallback(self):
        """Fallback method if PlotUnit doesn't have an update method"""
        print("[Plot Unit] Using fallback update method")
        # Try to push a message to the event queue if it exists
        if hasattr(self.plot_unit, 'event_queue'):
            self.plot_unit.event_queue.put({
                'type': 'refresh',
                'timestamp': time.time()
            })
    
    def _clear_plots_fallback(self):
        """Fallback method if PlotUnit doesn't have a clear_plots method"""
        print("[Plot Unit] Using fallback clear_plots method")
        # Try to clear data directly if possible
        if hasattr(self.plot_unit, 'data') and hasattr(self.plot_unit, 'data_lock'):
            with self.plot_unit.data_lock:
                for key in list(self.plot_unit.data.keys()):
                    self.plot_unit.data[key] = np.zeros(100)
                    
    def get_connected_nodes(self):
        """Get the number of connected nodes from the integration"""
        # Helper method to ensure we have get_connected_nodes method
        if hasattr(self.integration, 'get_connected_nodes'):
            return len(self.integration.get_connected_nodes())
        else:
            # Fallback to node connections attribute
            return len(getattr(self.integration, '_node_connections', {}))
            
    def run_visualization_hub(self, reset=False, clear_all_signals=False, auto_reset=False, 
                             performance_mode=False, signal_id=""):
        """
        Run the visualization hub. This function ensures the visualization
        window is running and can optionally reset it.
        """
        print(f"[Plot Unit] Node {self.node_id} executing")
        
        # Set performance mode if requested
        if hasattr(self.plot_unit, 'settings'):
            self.plot_unit.settings['performance_mode'] = performance_mode
            
        # Reset plots if requested explicitly or via auto-reset
        if reset or auto_reset:
            print("[Plot Unit] Resetting plots")
            self.plot_unit.clear_plots()
        
        # Clear all signals if requested
        if clear_all_signals:
            print("[Plot Unit] Clearing registry")
            self.registry.clear_signals()
            if hasattr(self.integration, 'reset'):
                self.integration.reset()
        
        # Check the registry for signals
        signals = self.registry.get_all_signals()
        
        # Connect to specific signal if provided
        if signal_id and signal_id.strip():
            print(f"[Plot Unit] Connecting to signal: {signal_id}")
            self.integration.connect_node_to_signal(self.node_id, signal_id)
        
        # Display node connection stats
        connected_nodes = self.get_connected_nodes()
        print(f"[Plot Unit] {connected_nodes} nodes connected to visualization")
        print(f"[Plot Unit] {len(signals)} signals available in registry")
        
        # Handle any force update needed
        try:
            # Try to call the update method
            self.plot_unit.update()
        except Exception as e:
            print(f"[Plot Unit] Warning: Failed to update plot unit: {str(e)}")
            # If update fails, try to force a refresh through other means
            if hasattr(self.plot_unit, 'event_queue'):
                self.plot_unit.event_queue.put({'type': 'refresh'})
        
        # The visualization happens in background threads
        return ()

# Node registration
NODE_CLASS_MAPPINGS = {
    "PlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlotUnitNode": "Plot Unit"
}
