# DEPRECATED: This file is deprecated and will be removed in future versions.
# Please use signal_input_node.py instead, which provides a more robust implementation
# with proper validation and connections between SignalRegistry and PlotRegistry.

import torch
import numpy as np
import logging
import sys
import os
from ...src.registry.plot_registry import PlotRegistry
from ...src.registry.plot_registry_integration import PlotRegistryIntegration

# Configure logger
logger = logging.getLogger('SignalConnectorNode')

class SignalConnectorNode:
    """
    DEPRECATED: Please use SignalInputNode from signal_input_node.py instead.
    
    Legacy node that connects signals to the registry for visualization.
    This node has been replaced by the more robust SignalInputNode which properly
    handles connections between SignalRegistry and PlotRegistry.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "default_signal"}),
                "signal_type": (["R", "P"], {"default": "R"}),  # 'R' for Raw, 'P' for Processed
                "connect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "signal": ("SIGNAL", ),
                "display_color": ("COLOR", {"default": "#42A5F5"}),  # Default blue color
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "connect_signal"
    CATEGORY = "Pedro_PIC/🌊 Signal Registry"  # Consistent with PlotUnitNode in __init__.py
    OUTPUT_NODE = True
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"signal_connector_{id(self)}"
        
        # Get registry and integration singletons
        self.registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Track connected signals
        self.connected_signals = {}
        logger.info(f"SignalConnectorNode initialized with ID: {self.node_id}")
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect node from registry
            if hasattr(self, 'registry'):
                self.registry.disconnect_node(self.node_id)
            logger.info(f"SignalConnectorNode {self.node_id} disconnected")
        except Exception as e:
            # This might fail during shutdown, so we'll just ignore errors
            pass
    
    def connect_signal(self, signal_id, signal_type, connect, signal=None, display_color="#42A5F5"):
        """Connect or disconnect a signal using the proper registry pattern"""
        status_msg = ""
        
        # Validate the signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return ("Error: Invalid signal ID type - must be a string",)
            
        # Convert hex color to RGB tuple
        try:
            color_hex = display_color.lstrip('#')
            rgb_color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            # Default color based on signal type
            rgb_color = (0, 180, 220) if signal_type == "F" else (220, 180, 0)
        
        # Handle connect/disconnect logic
        if connect:
            if signal is None:
                status_msg = f"Error: No signal provided for {signal_id}"
                logger.warning(f"No signal data provided for ID: {signal_id}")
            else:
                # Convert tensor to numpy if needed
                if isinstance(signal, torch.Tensor):
                    if len(signal.shape) > 1:
                        # Take the first row/channel if multidimensional
                        signal_data = signal[0].flatten().cpu().numpy()
                    else:
                        signal_data = signal.cpu().numpy()
                elif isinstance(signal, str):
                    # If signal is a string, try to get the signal from the registry
                    try:
                        registry_data = self.registry.get_signal(signal)
                        if registry_data is not None:
                            signal_data = registry_data
                        else:
                            logger.warning(f"String signal '{signal}' not found in registry")
                            status_msg = f"Error: Signal '{signal}' not found in registry"
                            return (status_msg,)
                    except Exception as e:
                        logger.error(f"Error processing string signal: {str(e)}")
                        status_msg = f"Error processing signal: {str(e)}"
                        return (status_msg,)
                else:
                    signal_data = np.array(signal)
                
                # Register the signal with the registry
                metadata = {
                    'color': rgb_color,
                    'source_node': self.node_id,
                    'signal_type': signal_type,
                }
                
                # Register the signal with the proper registry
                self.registry.register_signal(signal_id, signal_data, metadata)
                
                # Connect this node to the signal through integration
                result = self.integration.connect_node_to_signal(self.node_id, signal_id)
                
                if result:
                    # Store locally for tracking
                    self.connected_signals[signal_id] = signal_type
                    
                    logger.info(f"Signal {signal_id} registered and connected to registry")
                    status_msg = f"Signal {signal_id} connected as {signal_type} type"
                else:
                    status_msg = f"Failed to connect signal {signal_id}"
        else:
            # Disconnect signal if it was previously connected
            if signal_id in self.connected_signals:
                # Disconnect from registry through integration
                self.integration.disconnect_node(self.node_id)
                
                # Remove from local tracking
                del self.connected_signals[signal_id]
                logger.info(f"Signal {signal_id} disconnected from registry")
                status_msg = f"Signal {signal_id} disconnected"
            else:
                logger.warning(f"Signal {signal_id} was not connected")
                status_msg = f"Signal {signal_id} was not connected"
        
        return (status_msg,)


# For backward compatibility with existing workflows
class SignalRegistryConnector(SignalConnectorNode):
    """
    Compatibility class for existing workflows using the old SignalRegistryConnector.
    All functionality is now in SignalConnectorNode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "signal_1"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Register Signal", "label_off": "Remove Signal"}),
            },
            "optional": {
                "signal_data": ("SIGNAL", ),
                "display_color": ("COLOR", {"default": "#42A5F5"}),  # Default blue color
            }
        }
    
    RETURN_TYPES = ("STRING", )  # Return the signal_id for chaining
    RETURN_NAMES = ("signal_id", )
    FUNCTION = "process_signal"
    CATEGORY = "signal/registry"
    
    def process_signal(self, signal_id, enabled, signal_data=None, display_color="#42A5F5"):
        """
        Process a signal and decide whether to register it or remove it from the registry.
        Adapter method to connect with the new architecture.
        """
        # Convert parameters from old to new format
        result = self.connect_signal(
            signal_id=signal_id,
            signal_type="P",  # Default to P type
            connect=enabled,
            signal=signal_data,
            display_color=display_color
        )
        
        # Return signal_id for chaining as the old node did
        return (signal_id,)


# Node registration (all marked as deprecated)
NODE_CLASS_MAPPINGS = {
    "SignalConnectorNode": SignalConnectorNode,
    "SignalRegistryConnector": SignalRegistryConnector  # For backward compatibility
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalConnectorNode": "⚠️ Signal Connector (DEPRECATED)",
    "SignalRegistryConnector": "⚠️ Signal Registry Connector (DEPRECATED)"
}
