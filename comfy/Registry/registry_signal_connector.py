# DEPRECATED: This file is deprecated and will be removed in future versions.
# Please use signal_input_node.py instead, which properly connects SignalRegistry to PlotRegistry.

import uuid
from ...src.plot.plot_registry import PlotRegistry

class RegistrySignalConnector:
    """
    DEPRECATED: Please use SignalInputNode instead.
    
    A simplified node that connects or disconnects signals to the PlotRegistry for visualization.
    This node uses the old architecture that connects directly to PlotRegistry without going through
    SignalRegistry first. The new architecture uses SignalInputNode to bridge between SignalRegistry
    and PlotRegistry.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Connect", "label_off": "Disconnect"}),
                "signal_id": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_id",)
    FUNCTION = "toggle_connection"
    CATEGORY = "signal/registry"
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"registry_connector_{str(uuid.uuid4())[:8]}"
        
        # Get registry singleton
        self.registry = PlotRegistry.get_instance()
        
        print(f"[Registry Connector] Node {self.node_id} initialized")
    def toggle_connection(self, enabled, signal_id):
        """
        Connect or disconnect a signal to/from the PlotRegistry
        
        Args:
            enabled: Boolean flag to connect (True) or disconnect (False)
            signal_id: ID of the signal to connect/disconnect
            
        Returns:
            signal_id: The same signal ID that was passed in
        """
        # Validate signal_id is a string
        if not isinstance(signal_id, str):
            print(f"[ERROR] Invalid signal ID type: {type(signal_id)}. Expected string.")
            return ("ERROR: Invalid signal ID type, must be a string",)
            
        # Check for empty signal ID
        if not signal_id.strip():
            print(f"[Registry Connector] Warning: Empty signal ID provided")
            return (signal_id,)
        
        # Handle comma-separated list of signal IDs
        signal_ids = [s.strip() for s in signal_id.split(",") if s.strip()]
        
        for sid in signal_ids:
            if enabled:
                # Connect the node to the signal
                self.registry.connect_node_to_signal(self.node_id, sid)
                print(f"[Registry Connector] Connected signal '{sid}' to plot registry")
            else:
                # Disconnect the node from the signal
                self.registry.disconnect_node_from_signal(self.node_id, sid)
                print(f"[Registry Connector] Disconnected signal '{sid}' from plot registry")
        
        # Return the original signal ID string
        return (signal_id,)

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RegistrySignalConnector": RegistrySignalConnector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegistrySignalConnector": "Registry Signal Connector (Legacy)"
}
