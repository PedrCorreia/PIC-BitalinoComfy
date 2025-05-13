"""
Integration layer between PlotRegistry and PlotUnit for better visualization support.

This module adds methods to the PlotRegistryIntegration class to enhance the functionality 
inspired by the PygamePlotTemplate, making it work better with our visualization system.
"""

import sys
import os
import logging

# Configure logger
logger = logging.getLogger('ExtendedPlotRegistryIntegration')

def patch_plot_registry_integration(integration_class):
    """
    Add new methods to the PlotRegistryIntegration class to support enhanced visualization.
    Only adds methods if they don't already exist.
    """
    
    # Method to get connected nodes
    if not hasattr(integration_class, 'get_connected_nodes'):
        def get_connected_nodes(self):
            """Get all nodes connected to the integration"""
            return list(getattr(self, '_node_connections', {}).keys())
        
        integration_class.get_connected_nodes = get_connected_nodes
        logger.info("Added get_connected_nodes method to PlotRegistryIntegration")
    
    # Method to disconnect a node from a specific signal
    if not hasattr(integration_class, 'disconnect_node_from_signal'):
        def disconnect_node_from_signal(self, node_id, signal_id):
            """Disconnect a node from a specific signal"""
            if hasattr(self, '_node_connections') and node_id in self._node_connections:
                self._node_connections[node_id].discard(signal_id)
                logger.info(f"Disconnected node {node_id} from signal {signal_id}")
                
                # Also tell registry about disconnection if possible
                if hasattr(self.registry, 'disconnect_node_from_signal'):
                    self.registry.disconnect_node_from_signal(node_id, signal_id)
                return True
            return False
        
        integration_class.disconnect_node_from_signal = disconnect_node_from_signal
        logger.info("Added disconnect_node_from_signal method to PlotRegistryIntegration")
        
    # Advanced update method for smoother visualization
    if not hasattr(integration_class, 'update_visualizations'):
        def update_visualizations(self):
            """Force update of all visualizations"""
            if self.plot_unit:
                if hasattr(self.plot_unit, 'update'):
                    self.plot_unit.update()
                logger.info("Visualizations updated")
            
        integration_class.update_visualizations = update_visualizations
        logger.info("Added update_visualizations method to PlotRegistryIntegration")
        
# Try to import and patch the integration class
try:
    # Import from current package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.registry.plot_registry_integration import PlotRegistryIntegration
    
    # Apply patches
    patch_plot_registry_integration(PlotRegistryIntegration)
    logger.info("Successfully patched PlotRegistryIntegration")
    
except Exception as e:
    logger.error(f"Failed to patch PlotRegistryIntegration: {str(e)}")
