# filepath: c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\comfy\Registry\signal_input_node.py
"""
Signal Input Node - The ONE node that connects signals from registry to visualization.
This is the unified implementation that replaces all other signal connector nodes.
"""

import torch
import numpy as np
import logging
from ...src.plot.plot_registry import PlotRegistry
from ...src.plot.plot_registry_integration import PlotRegistryIntegration

# Configure logger
logger = logging.getLogger('SignalInputNode')

class SignalInputNode:
    """
    A single node that connects signals from the registry to the visualization system.
    This follows the proper architecture pattern by using PlotRegistryIntegration.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "default_signal"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "color_r": ("INT", {"default": 220, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 180, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "signal_type": (["P", "F"], {"default": "P"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_id",)
    FUNCTION = "process_signal"
    CATEGORY = "Pedro_PIC/🧰 Visualization"
    OUTPUT_NODE = True
    
    def __init__(self):
        """Initialize the node with proper registry connections"""
        # Generate a unique ID for this node
        self.node_id = f"signal_input_{id(self)}"
        
        # Get registry and integration singletons
        self.registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Track connected signals
        self.connected_signals = {}
        
        # Register this node with the integration
        self.integration.register_node(self.node_id)
        
        logger.info(f"SignalInputNode initialized with ID: {self.node_id}")
    
    def process_signal(self, signal_id, enabled, color_r=220, color_g=180, color_b=0, signal_type="P"):
        """
        Process a signal by connecting it to the visualization system.
        
        Args:
            signal_id: ID of the signal in the registry
            enabled: Whether to connect the signal for visualization
            color_r/g/b: RGB color components for visualization
            signal_type: Signal type ('P' or 'F')
        
        Returns:
            signal_id: The input signal ID (for chaining)
        """
        # Validate the signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return (f"Error: Invalid signal ID type - must be a string",)
            
        # Define RGB color tuple from components
        color = (color_r, color_g, color_b)
        
        # Handle comma-separated signal IDs (typical in synthetic generators)
        if ',' in signal_id:
            logger.info(f"Received multiple signals: {signal_id}, using first one")
            signal_id = signal_id.split(',')[0].strip()
        
        if enabled:
            # Get signal from registry to validate it exists
            signal_data = self.registry.get_signal(signal_id)
            
            if signal_data is not None:
                # Update metadata with the new color
                metadata = self.registry.get_signal_metadata(signal_id) or {}
                metadata['color'] = color
                metadata['signal_type'] = signal_type
                metadata['last_used'] = id(self)  # Unique timestamp
                
                # Connect this node to the signal through integration
                result = self.integration.connect_node_to_signal(self.node_id, signal_id)
                
                if result:
                    # Store locally for tracking
                    self.connected_signals[signal_id] = signal_type
                    logger.info(f"Signal '{signal_id}' connected to visualization with color {color}")
                else:
                    logger.warning(f"Failed to connect signal '{signal_id}' to visualization")
            else:
                logger.warning(f"Signal '{signal_id}' not found in registry - nothing to connect")
        else:
            # If disabled, disconnect this node from all signals
            self.integration.disconnect_node(self.node_id)
            self.connected_signals.clear()
            logger.info(f"Node {self.node_id} disconnected from all signals")
        
        # Return the signal ID for chaining
        return (signal_id,)
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect node from registry integration
            if hasattr(self, 'integration'):
                self.integration.disconnect_node(self.node_id)
            logger.info(f"SignalInputNode {self.node_id} disconnected from registry")
        except Exception as e:
            # This might fail during shutdown, so we'll just ignore errors
            pass

# Node registration
NODE_CLASS_MAPPINGS = {
    "SignalInputNode": SignalInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalInputNode": "Signal Input"
}
