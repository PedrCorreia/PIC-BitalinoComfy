"""
Signal Input Node for PIC-2025.
This node connects signals from the SignalRegistry to the PlotRegistry for visualization.
"""

import torch
import numpy as np
import logging
import uuid
from typing import Dict, List, Tuple, Union

# Import registries
from ...src.plot.signal_registry import SignalRegistry
from ...src.plot.plot_registry import PlotRegistry

# Configure logger
logger = logging.getLogger('SignalInputNode')

class SignalInputNode:
    """
    Node that bridges between SignalRegistry and PlotRegistry.
    
    This node:
    1. Receives a signal ID from the SignalRegistry
    2. Retrieves the signal data from SignalRegistry
    3. Connects the signal to PlotRegistry for visualization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "default_signal"}),
                "visualize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "display_color": ("COLOR", {"default": "#42A5F5"}),  # Default blue color
                "signal_alias": ("STRING", {"default": ""}),  # Optional display name
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "connect_signal"
    CATEGORY = "Pedro_PIC/🧰 Tools"
    OUTPUT_NODE = True
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"signal_input_{uuid.uuid4().hex[:8]}"
        
        # Get registry singletons
        self.signal_registry = SignalRegistry.get_instance()
        self.plot_registry = PlotRegistry.get_instance()
        
        # Track connected signals
        self.connected_signals = {}
        
        logger.info(f"SignalInputNode initialized with ID: {self.node_id}")
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect node from registry
            if hasattr(self, 'plot_registry'):
                self.plot_registry.disconnect_node(self.node_id)
            logger.info(f"SignalInputNode {self.node_id} disconnected")
        except Exception as e:
            # This might fail during shutdown, so we'll just ignore errors
            pass
    
    def connect_signal(self, signal_id: str, visualize: bool, 
                     display_color: str = "#42A5F5", signal_alias: str = "") -> Tuple[str]:
        """
        Connect a signal from SignalRegistry to PlotRegistry for visualization
        
        Args:
            signal_id: ID of the signal in SignalRegistry
            visualize: Whether to visualize this signal
            display_color: Color for visualization (hex format)
            signal_alias: Optional display name for the signal
            
        Returns:
            Tuple containing a status message
        """
        status_msg = ""
        
        # Validate the signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return ("Error: Invalid signal ID type - must be a string",)
        
        # Use alias if provided, otherwise use the original ID
        display_id = signal_alias if signal_alias else signal_id
            
        # Convert hex color to RGB tuple
        try:
            color_hex = display_color.lstrip('#')
            rgb_color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            # Default to blue if color parsing fails
            rgb_color = (66, 165, 245)  # Default blue
        
        # Handle connect/disconnect logic
        if visualize:
            # Get signal from SignalRegistry
            signal_data = self.signal_registry.get_signal(signal_id)
            
            if signal_data is None:
                status_msg = f"Error: Signal '{signal_id}' not found in SignalRegistry"
                logger.warning(f"Signal '{signal_id}' not found in SignalRegistry")
                return (status_msg,)
            
            # Get metadata from SignalRegistry
            original_metadata = self.signal_registry.get_signal_metadata(signal_id)
            
            # Create metadata for PlotRegistry
            metadata = {
                'color': rgb_color,
                'source_node': self.node_id,
                'original_id': signal_id,
                'display_name': display_id
            }
            
            # Add any original metadata
            if original_metadata:
                for key, value in original_metadata.items():
                    if key not in metadata:
                        metadata[key] = value
            
            # Register the signal with PlotRegistry
            self.plot_registry.register_signal(display_id, signal_data, metadata)
            
            # Connect this node to the signal in PlotRegistry
            result = self.plot_registry.connect_node_to_signal(self.node_id, display_id)
            
            if result:
                # Store locally for tracking
                self.connected_signals[display_id] = signal_id
                
                logger.info(f"Signal '{signal_id}' connected to PlotRegistry as '{display_id}'")
                status_msg = f"Signal '{signal_id}' connected for visualization"
            else:
                status_msg = f"Failed to connect signal '{signal_id}' to PlotRegistry"
        else:
            # Disconnect signal if it was previously connected
            connections_to_remove = []
            for display_id, orig_id in self.connected_signals.items():
                if orig_id == signal_id:
                    connections_to_remove.append(display_id)
            
            if connections_to_remove:
                for display_id in connections_to_remove:
                    self.plot_registry.disconnect_node(self.node_id)
                    del self.connected_signals[display_id]
                
                logger.info(f"Signal '{signal_id}' disconnected from visualization")
                status_msg = f"Signal '{signal_id}' disconnected from visualization"
            else:
                logger.warning(f"Signal '{signal_id}' was not connected for visualization")
                status_msg = f"Signal '{signal_id}' was not connected"
        
        return (status_msg,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SignalInputNode": SignalInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalInputNode": "Signal Input"
}