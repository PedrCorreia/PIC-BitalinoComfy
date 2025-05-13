"""
Signal Input Node for PIC-2025.
This node connects signals from SignalRegistry to PlotRegistry for visualization.
"""

import torch
import numpy as np
import uuid
import logging
from ...src.registry.signal_registry import SignalRegistry
from ...src.registry.plot_registry import PlotRegistry
from ...src.registry.plot_registry_integration import PlotRegistryIntegration

# Configure logger
logger = logging.getLogger("ComfyUI")

class SignalInputNode:
    """
    A node that connects signals from SignalRegistry to PlotRegistry for visualization.
    
    This node acts as a bridge between the signal generators and visualization tools,
    allowing signals to be properly displayed in the PlotUnit.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "signal_1"}),
                "signal_type": (["R", "P"], {"default": "R"}),  # 'R' for Raw, 'P' for Processed
                "connect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "signal": ("SIGNAL", ),
                "display_color": ("COLOR", ),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "connect_signal"
    CATEGORY = "Pedro_PIC/🌊 Signal Registry"  # Consistent with PlotUnitNode
    OUTPUT_NODE = True
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"signal_input_{str(uuid.uuid4())[:8]}"
        
        # Get registry singletons
        self.signal_registry = SignalRegistry.get_instance()
        self.plot_registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Track connected signals
        self.connected_signals = {}
        logger.info(f"SignalInputNode initialized with ID: {self.node_id}")
    
    def connect_signal(self, signal_id, signal_type, connect, signal=None, display_color=None):
        """Connect a signal from SignalRegistry to PlotRegistry for visualization"""
        status_msg = ""
        
        # Validate signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return ("Error: Invalid signal ID type - must be a string",)
        
        # Convert display_color to RGB tuple if provided
        if display_color:
            try:
                color_hex = display_color.lstrip('#')
                rgb_color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            except:
                # Default colors based on signal type
                rgb_color = (0, 0, 220) if signal_type == "R" else (220, 0, 0)
        else:
            # Default colors based on signal type
            rgb_color = (0, 0, 220) if signal_type == "R" else (220, 0, 0)
        
        # Handle connect/disconnect logic
        if connect:
            # If signal is provided directly, use that
            if signal is not None:
                # Convert tensor to numpy if needed
                if isinstance(signal, torch.Tensor):
                    if len(signal.shape) > 1:
                        signal_data = signal[0].flatten().cpu().numpy()
                    else:
                        signal_data = signal.cpu().numpy()
                elif isinstance(signal, np.ndarray):
                    signal_data = signal
                else:
                    try:
                        signal_data = np.array(signal)
                    except:
                        logger.error(f"Cannot convert signal to numpy array: {type(signal)}")
                        return (f"Error: Cannot process signal of type {type(signal)}",)
                
                # Register with signal registry if it's a new signal
                if self.signal_registry.get_signal(signal_id) is None:
                    self.signal_registry.register_signal(
                        signal_id=signal_id,
                        signal_data=signal_data,
                        metadata={
                            'color': rgb_color,
                            'signal_type': signal_type,
                            'source_node': self.node_id
                        }
                    )
                
                status_msg = f"Connected and registered signal: {signal_id}"
            else:
                # Try to get signal from registry
                signal_data = self.signal_registry.get_signal(signal_id)
                if signal_data is None:
                    status_msg = f"Error: Signal not found in registry: {signal_id}"
                    logger.warning(f"Signal {signal_id} not found in registry")
                    return (status_msg,)
                    
                status_msg = f"Connected to existing signal: {signal_id}"
            
            # Now connect to plot registry through integration
            metadata = self.signal_registry.get_signal_metadata(signal_id)
            if metadata is None:
                metadata = {'color': rgb_color, 'signal_type': signal_type}
            
            # Register with plot registry
            self.plot_registry.register_signal(
                signal_id=signal_id,
                signal_data=signal_data,
                metadata=metadata
            )
            
            # Connect node to signal for visualization
            self.plot_registry.connect_node_to_signal(self.node_id, signal_id)
            self.connected_signals[signal_id] = True
            
            logger.info(f"Signal {signal_id} connected for visualization")
            status_msg = f"Signal {signal_id} connected for visualization ({len(signal_data)} samples)"
        
        else:
            # Disconnect the signal
            if signal_id in self.connected_signals:
                self.plot_registry.disconnect_node_from_signal(self.node_id, signal_id)
                del self.connected_signals[signal_id]
                status_msg = f"Disconnected from signal: {signal_id}"
            else:
                status_msg = f"Signal was not connected: {signal_id}"
        
        return (status_msg,)
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect all signals
            for signal_id in list(self.connected_signals.keys()):
                self.plot_registry.disconnect_node_from_signal(self.node_id, signal_id)
            
            # Unregister from integration
            self.integration.disconnect_node(self.node_id)
            
            logger.info(f"SignalInputNode {self.node_id} cleaned up")
        except:
            # Might fail during shutdown
            pass

# Node registration
NODE_CLASS_MAPPINGS = {
    "SignalInputNode": SignalInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalInputNode": "🔌 Signal Input"
}
