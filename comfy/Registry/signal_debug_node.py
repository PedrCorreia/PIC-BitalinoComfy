import sys
import os
from ...src.plot.plot_registry import PlotRegistry
import numpy as np
import logging
import uuid
import time

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('SignalDebugNode')

class SignalDebugNode:
    """
    A diagnostic node that helps debug signal flows in the registry system.
    Shows detailed information about signals in the registry and their state.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "signal_1"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR"], {"default": "DEBUG"}),
            },
            "optional": {
                "inspect_registry": ("BOOLEAN", {"default": True, "label_on": "Inspect Registry", "label_off": "Skip Inspection"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "debug_signal"
    CATEGORY = "signal/diagnostics"
    OUTPUT_NODE = True
    
    def __init__(self):
        self.node_id = f"debug_{str(uuid.uuid4())[:8]}"
        self.registry = PlotRegistry.get_instance()
        self.debug_counter = 0
        logger.info(f"SignalDebugNode initialized with ID: {self.node_id}")
    
    def debug_signal(self, signal_id, log_level, inspect_registry=True):
        """Debug a signal and its flow through the registry"""
        # Set logging level
        numeric_level = getattr(logging, log_level)
        logger.setLevel(numeric_level)
        
        self.debug_counter += 1
        logger.debug(f"Debug run #{self.debug_counter} for signal: {signal_id}")
        
        status_lines = [f"===== Signal Debug Report [{time.strftime('%H:%M:%S')}] ====="]
        status_lines.append(f"Signal ID: {signal_id}")
        
        # Check if the signal exists in the registry
        signal_data = self.registry.get_signal(signal_id) if inspect_registry else None
        
        if signal_data is not None:
            # Signal exists
            status_lines.append(f"✓ Signal exists in registry")
            
            # Get shape and type
            try:
                if hasattr(signal_data, 'shape'):
                    status_lines.append(f"• Shape: {signal_data.shape}")
                elif hasattr(signal_data, '__len__'):
                    status_lines.append(f"• Length: {len(signal_data)}")
                    
                status_lines.append(f"• Type: {type(signal_data).__name__}")
                
                # Check for NaN values
                if hasattr(signal_data, 'isnan') and hasattr(signal_data, 'any'):
                    if signal_data.isnan().any():
                        status_lines.append(f"⚠️ Contains NaN values")
                
                # Get metadata
                metadata = self.registry.get_signal_metadata(signal_id)
                if metadata:
                    status_lines.append(f"• Metadata: {metadata}")
                    if 'color' in metadata:
                        status_lines.append(f"  - Color: {metadata['color']}")
                    if 'created' in metadata:
                        timestamp = metadata['created']
                        time_diff = time.time() - timestamp
                        status_lines.append(f"  - Age: {time_diff:.1f} seconds")
                
                # Check which nodes are connected
                connected_nodes = []
                for node_id, signals in self.registry.connections.items():
                    if signal_id in signals:
                        connected_nodes.append(node_id)
                
                if connected_nodes:
                    status_lines.append(f"• Connected to {len(connected_nodes)} nodes:")
                    for node in connected_nodes[:5]:  # Show at most 5 nodes
                        status_lines.append(f"  - {node}")
                    if len(connected_nodes) > 5:
                        status_lines.append(f"  - ...and {len(connected_nodes) - 5} more")
                else:
                    status_lines.append("⚠️ No nodes connected to this signal")
                
                # Check if it's in visualized signals
                if signal_id in self.registry.visualized_signals:
                    status_lines.append("✓ Signal marked for visualization")
                else:
                    status_lines.append("⚠️ Signal NOT marked for visualization")
                
            except Exception as e:
                status_lines.append(f"⚠️ Error inspecting signal: {str(e)}")
        else:
            # Signal doesn't exist
            status_lines.append("❌ Signal NOT found in registry")
            
            # Check available signals
            available_signals = self.registry.get_all_signals()
            if available_signals:
                status_lines.append(f"• Available signals: {len(available_signals)}")
                for sig in available_signals[:5]:  # Show at most 5 signals
                    status_lines.append(f"  - {sig}")
                if len(available_signals) > 5:
                    status_lines.append(f"  - ...and {len(available_signals) - 5} more")
            else:
                status_lines.append("• No signals in registry")
        
        # Registry statistics
        if inspect_registry:
            status_lines.append("\n=== Registry Statistics ===")
            status_lines.append(f"• Connected nodes: {self.registry.connected_nodes}")
            status_lines.append(f"• Total signals: {len(self.registry.signals)}")
            status_lines.append(f"• Visualized signals: {len(self.registry.visualized_signals)}")
        
        # Log the status
        status_text = "\n".join(status_lines)
        logger.debug("\n" + status_text)
        
        return (status_text,)

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SignalDebugNode": SignalDebugNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalDebugNode": "Signal Debug"
}
