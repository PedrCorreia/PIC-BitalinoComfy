"""
Enhanced Signal Node implementation that combines functionality from:
- SignalConnectorNode
- SignalRegistryConnector
- SignalInputNode

This is designed to be the one central node for all signal registry connections
in the PIC-2025 system, following the proper registry architecture.
"""

import torch
import numpy as np
import logging
from ...src.plot.plot_registry import PlotRegistry
from ...src.plot.plot_registry_integration import PlotRegistryIntegration
from ...src.plot.signal_registry import SignalRegistry
from ...src.plot.plot_unit import PlotUnit

# Configure logger
logger = logging.getLogger('EnhancedSignalNode')

class EnhancedSignalNode:
    """
    Universal signal node that provides bridge functionality between:
    - Signal Registry (for input signals)
    - Plot Registry (for plot registry architecture)
    - Plot Unit (for visualization)
    
    This node follows the proper architecture pattern by:
    1. Taking a signal ID (and optionally signal data)
    2. Fetching the signal from SignalRegistry if needed
    3. Registering it with PlotRegistry
    4. Connecting to the visualization system through PlotRegistryIntegration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "signal_1"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Connect", "label_off": "Disconnect"}),
                "signal_type": (["P", "F"], {"default": "P"}),
            },
            "optional": {
                "signal": ("SIGNAL", ),
                "display_color": ("COLOR", {"default": "#42A5F5"}),  # Default blue color
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process_signal"
    CATEGORY = "Pedro_PIC/🧰 Visualization"
    OUTPUT_NODE = True
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"signal_node_{id(self)}"
        
        # Get all required singletons
        self.plot_registry = PlotRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        self.signal_registry = SignalRegistry.get_instance()
        self.plot_unit = PlotUnit.get_instance()
        
        # Initialize plot unit if needed
        self.plot_unit.start()
        
        # Register as a connected node
        self.integration.register_node(self.node_id)
        
        # Track connected signals
        self.connected_signals = {}
        
        logger.info(f"EnhancedSignalNode initialized with ID: {self.node_id}")
    
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Disconnect node from the registry
            if hasattr(self, 'integration'):
                self.integration.disconnect_node(self.node_id)
            logger.info(f"EnhancedSignalNode {self.node_id} disconnected")
        except Exception as e:
            # This might fail during shutdown, so we'll just ignore errors
            pass
    
    def process_signal(self, signal_id, enabled, signal_type, signal=None, display_color="#42A5F5"):
        """
        Process and connect a signal to the visualization system using the registry pattern.
        
        Args:
            signal_id: ID of the signal to connect
            enabled: Whether to connect (True) or disconnect (False) the signal
            signal_type: Type of signal ("P" or "F")
            signal: Optional signal data. If not provided, attempts to fetch from SignalRegistry
            display_color: Color to use for displaying the signal
        
        Returns:
            status message describing the result
        """
        status_msg = ""
        
        # Validate the signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return ("Error: Invalid signal ID type - must be a string",)
        
        # Handle comma-separated signal IDs
        if ',' in signal_id:
            logger.info(f"Received multiple signals: {signal_id}, using first one")
            signal_id = signal_id.split(',')[0]
        
        # Convert hex color to RGB tuple
        try:
            color_hex = display_color.lstrip('#')
            rgb_color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        except:
            # Default color based on signal type
            rgb_color = (0, 180, 220) if signal_type == "F" else (220, 180, 0)
        
        # Process based on enabled/disabled state
        if enabled:
            # Step 1: Get the signal data (either from input or from SignalRegistry)
            signal_data = None
            
            if signal is not None:
                # Use the provided signal
                if isinstance(signal, torch.Tensor):
                    if len(signal.shape) > 1:
                        # Take the first row/channel if multidimensional
                        signal_data = signal[0].flatten().cpu().numpy()
                    else:
                        signal_data = signal.cpu().numpy()
                elif isinstance(signal, str):
                    # If signal is a string reference, try to get the signal from the registry
                    try:
                        registry_data = self.signal_registry.get_signal(signal)
                        if registry_data is not None:
                            signal_data = registry_data
                        else:
                            logger.warning(f"String signal '{signal}' not found in signal registry")
                    except Exception as e:
                        logger.error(f"Error processing string signal: {str(e)}")
                else:
                    signal_data = np.array(signal)
            else:
                # No signal provided, try to get from SignalRegistry
                signal_data = self.signal_registry.get_signal(signal_id)
                
                if signal_data is None:
                    status_msg = f"Warning: No signal data found for '{signal_id}'"
                    logger.warning(f"No signal data found for ID: {signal_id}")
                    return (status_msg,)
            
            # Step 2: Register the signal with PlotRegistry
            metadata = {
                'color': rgb_color,
                'source_node': self.node_id,
                'signal_type': signal_type,
            }
            
            self.plot_registry.register_signal(signal_id, signal_data, metadata)
            
            # Step 3: Connect this node to the signal through the integration layer
            result = self.integration.connect_node_to_signal(self.node_id, signal_id)
            
            if result:
                # Store locally for tracking
                self.connected_signals[signal_id] = signal_type
                
                logger.info(f"Signal '{signal_id}' connected and registered for visualization")
                status_msg = f"Signal '{signal_id}' connected as {signal_type} type"
            else:
                status_msg = f"Failed to connect signal '{signal_id}'"
        else:
            # Disconnect signal
            if signal_id in self.connected_signals:
                # Disconnect from registry through integration
                self.integration.disconnect_node(self.node_id)
                
                # Remove from local tracking
                del self.connected_signals[signal_id]
                logger.info(f"Signal '{signal_id}' disconnected from visualization")
                status_msg = f"Signal '{signal_id}' disconnected"
            else:
                logger.warning(f"Signal '{signal_id}' was not connected")
                status_msg = f"Signal '{signal_id}' was not connected"
        
        return (status_msg,)


# For backward compatibility
class SignalInputNode(EnhancedSignalNode):
    """Legacy class for backward compatibility with existing workflows"""
    pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "EnhancedSignalNode": EnhancedSignalNode,
    "SignalInputNode": SignalInputNode,  # For backward compatibility
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSignalNode": "Signal Node",
    "SignalInputNode": "Signal Input (Legacy)",
}
