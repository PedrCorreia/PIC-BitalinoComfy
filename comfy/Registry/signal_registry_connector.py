import sys
import os
from ...src.plot.plot_registry import PlotRegistry

import torch
import numpy as np
import logging

# Configure logger
logger = logging.getLogger('SignalRegistryConnector')

class SignalRegistryConnector:
    """
    A node that connects a signal to the visualization registry.
    This node controls whether a signal is passed to the registry or removed from it.
    When enabled, the signal ID is registered with the plot registry.
    When disabled, the signal ID is removed from the registry if it exists.
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
    OUTPUT_NODE = True
    
    def __init__(self):
        # Get the registry instance
        self.registry = PlotRegistry.get_instance()
        self.node_id = f"registry_connector_{id(self)}"
        logger.info(f"SignalRegistryConnector initialized with ID: {self.node_id}")
    
    def __del__(self):
        # Clean up by disconnecting from the registry
        try:
            self.registry.disconnect_node(self.node_id)
            logger.info(f"SignalRegistryConnector {self.node_id} disconnected from registry")
        except:
            pass
    def process_signal(self, signal_id, enabled, signal_data=None, display_color="#42A5F5"):
        """
        Process a signal and decide whether to register it or remove it from the registry.
        
        Args:
            signal_id: ID of the signal
            enabled: Boolean toggle for registry connection
            signal_data: Optional signal data to register
            display_color: Color to use for displaying the signal
        
        Returns:
            signal_id: The input signal ID (for chaining)
        """
        logger.info(f"SignalRegistryConnector processing signal: {signal_id}, enabled: {enabled}")
        
        # Convert hex color to RGB tuple
        try:
            color_hex = display_color.lstrip('#')
            rgb_color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            # Default color if parsing fails
            rgb_color = (66, 165, 245)  # Default blue
        
        if enabled:
            # Connect the signal to the registry
            if signal_data is not None:                # Convert to numpy if tensor
                if isinstance(signal_data, torch.Tensor):
                    signal_data = signal_data.detach().cpu().numpy()
                elif isinstance(signal_data, str):
                    # If signal_data is a string, try to get the signal from the registry
                    try:
                        registry_data = self.registry.get_signal(signal_data)
                        if registry_data is not None:
                            signal_data = registry_data
                        else:
                            logger.warning(f"String signal_data '{signal_data}' not found in registry")
                    except Exception as e:
                        logger.error(f"Error processing string signal_data: {str(e)}")
                
                # Register the signal data with metadata
                metadata = {
                    'color': rgb_color,
                    'source_node': self.node_id,
                    'timestamp': id(signal_data)  # Use object id as timestamp for uniqueness
                }
                
                self.registry.register_signal(signal_id, signal_data, metadata)
                logger.info(f"Registered signal {signal_id} with registry")
            
            # Connect this node to the signal (even without data, to track connections)
            self.registry.connect_node_to_signal(self.node_id, signal_id)
            logger.info(f"Signal {signal_id} added to registry visualization")
        else:
            # Remove the signal from the registry
            try:
                # First disconnect this node
                self.registry.disconnect_node(self.node_id)
                logger.info(f"Signal {signal_id} removed from registry visualization")
            except Exception as e:
                logger.error(f"Error removing signal {signal_id} from registry: {str(e)}")
        
        return (signal_id, )

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SignalRegistryConnector": SignalRegistryConnector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalRegistryConnector": "Signal Registry Connector"
}
