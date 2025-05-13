"""
Registry Initializer for PIC-2025.
This module applies necessary patches and initializes the registry system.
"""

import importlib
import logging
import inspect
import sys
import os

# Configure logger
logger = logging.getLogger("PIC2025Initializer")
handler = logging.StreamHandler()
formatter = logging.Formatter('[PIC-2025] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def patch_signal_architecture():
    """Apply necessary patches to the signal architecture"""
    try:
        # Import the patch function
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from tools.patch_registry_integration import patch_plot_registry_integration
        
        # Import the class to patch
        from src.registry.plot_registry_integration import PlotRegistryIntegration
        
        # Apply the patch
        patch_plot_registry_integration(PlotRegistryIntegration)
        
        logger.info("Registry integration successfully patched")
        return True
    except Exception as e:
        logger.error(f"Failed to patch registry integration: {str(e)}")
        return False

def ensure_update_method():
    """Ensure PlotUnit has update method"""
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from src.plot.plot_unit import PlotUnit
        
        # Check if update method exists
        if not hasattr(PlotUnit, 'update'):
            logger.warning("PlotUnit missing update method, adding it")
            
            # Define an update method
            def update(self):
                """Force an update of the visualization"""
                self.event_queue.put({'type': 'refresh', 'timestamp': __import__('time').time()})
                return True
                
            # Add the method
            setattr(PlotUnit, 'update', update)
            
        logger.info("PlotUnit update method verified")
        return True
    except Exception as e:
        logger.error(f"Failed to verify PlotUnit update method: {str(e)}")
        return False

def ensure_clear_plots_method():
    """Ensure PlotUnit has clear_plots method"""
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from src.plot.plot_unit import PlotUnit
        
        # Check if clear_plots method exists
        if not hasattr(PlotUnit, 'clear_plots'):
            logger.warning("PlotUnit missing clear_plots method, adding it")
            
            # Define a clear_plots method
            def clear_plots(self):
                """Clear all plots from the visualization"""
                with self.data_lock:
                    for key in list(self.data.keys()):
                        self.data[key] = __import__('numpy').zeros(100)
                return True
                
            # Add the method
            setattr(PlotUnit, 'clear_plots', clear_plots)
            
        logger.info("PlotUnit clear_plots method verified")
        return True
    except Exception as e:
        logger.error(f"Failed to verify PlotUnit clear_plots method: {str(e)}")
        return False

def check_categories():
    """Check if all nodes have consistent categories"""
    consistent = True
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        
        # Try to import the nodes
        signal_input_node_exists = False
        plot_unit_node_exists = False
        
        try:
            from comfy.Registry.signal_input_node import SignalInputNode
            signal_input_node_exists = True
        except ImportError:
            logger.warning("SignalInputNode not found")
        
        try:
            from comfy.Registry.plot_unit_node import PlotUnitNode
            plot_unit_node_exists = True
        except ImportError:
            logger.warning("PlotUnitNode not found")
        
        # Check categories if both nodes exist
        if signal_input_node_exists and plot_unit_node_exists:
            signal_category = getattr(SignalInputNode, 'CATEGORY', None)
            plot_category = getattr(PlotUnitNode, 'CATEGORY', None)
            
            if signal_category != plot_category:
                logger.warning(f"Inconsistent categories: SignalInputNode ({signal_category}) != PlotUnitNode ({plot_category})")
                consistent = False
            else:
                logger.info(f"Node categories are consistent: {signal_category}")
        else:
            logger.warning("Could not check categories as some nodes are missing")
            consistent = False
            
    except Exception as e:
        logger.error(f"Failed to check node categories: {str(e)}")
        consistent = False
        
    return consistent

def use_new_plot_unit_node():
    """Replace the plot_unit_node with the new version if it exists"""
    try:
        # Check if new_plot_unit_node.py exists
        new_plot_path = os.path.join(os.path.dirname(__file__), "new_plot_unit_node.py")
        plot_path = os.path.join(os.path.dirname(__file__), "plot_unit_node.py")
        
        if os.path.exists(new_plot_path):
            # Import the new class for inspection
            sys.path.append(os.path.dirname(new_plot_path))
            try:
                new_module = importlib.import_module("new_plot_unit_node")
                if hasattr(new_module, "PlotUnitNode"):
                    logger.info("Found new PlotUnitNode implementation")
                    
                    # Use the class from new file
                    from comfy.Registry.new_plot_unit_node import PlotUnitNode as NewPlotUnitNode
                    from comfy.Registry.plot_unit_node import PlotUnitNode as OldPlotUnitNode
                    
                    # Copy the new class into the old module namespace
                    import sys
                    sys.modules["comfy.Registry.plot_unit_node"].PlotUnitNode = NewPlotUnitNode
                    
                    logger.info("Successfully replaced PlotUnitNode with new implementation")
                    return True
                else:
                    logger.warning("New plot_unit_node.py doesn't contain PlotUnitNode class")
            except Exception as e:
                logger.error(f"Error importing new_plot_unit_node: {str(e)}")
        else:
            logger.info("No new plot unit node implementation found")
            
        return False
    except Exception as e:
        logger.error(f"Error replacing plot_unit_node: {str(e)}")
        return False
    
def fix_missing_get_connected_nodes():
    """Add get_connected_nodes method to integration if missing"""
    try:
        from src.registry.plot_registry_integration import PlotRegistryIntegration
        
        if not hasattr(PlotRegistryIntegration, 'get_connected_nodes'):
            def get_connected_nodes(self):
                """Get all nodes connected to the integration"""
                return list(getattr(self, '_node_connections', {}).keys())
            
            setattr(PlotRegistryIntegration, 'get_connected_nodes', get_connected_nodes)
            logger.info("Added missing get_connected_nodes method")
            
        return True
    except Exception as e:
        logger.error(f"Failed to fix missing get_connected_nodes: {str(e)}")
        return False

# Run initialization
def initialize():
    """Run all initialization tasks"""
    success = True
    
    # Apply patches
    if not patch_signal_architecture():
        success = False
    
    # Ensure methods exist
    if not ensure_update_method():
        success = False
        
    if not ensure_clear_plots_method():
        success = False
    
    # Fix missing methods
    fix_missing_get_connected_nodes()
    
    # Try to use new plot unit node
    use_new_plot_unit_node()
    
    # Check categories
    if not check_categories():
        # This is a warning, not a failure
        pass
    
    if success:
        logger.info("Registry initialization completed successfully")
    else:
        logger.warning("Registry initialization completed with warnings")
        
    return success

# Run initialization when imported
if __name__ == "__main__":
    initialize()
else:
    # When imported as a module, also initialize
    initialize()
