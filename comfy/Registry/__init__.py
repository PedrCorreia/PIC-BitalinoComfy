"""
Registry Initialization Module for PIC-2025.
This module applies necessary patches and initializes the registry system.
"""

import importlib
import logging
import inspect
import sys
import os

# Configure logger
logger = logging.getLogger("PIC2025Initialization")
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
        from comfy.Registry.signal_input_node import SignalInputNode
        from comfy.Registry.plot_unit_node import PlotUnitNode
        
        signal_category = getattr(SignalInputNode, 'CATEGORY', None)
        plot_category = getattr(PlotUnitNode, 'CATEGORY', None)
        
        if signal_category != plot_category:
            logger.warning(f"Inconsistent categories: SignalInputNode ({signal_category}) != PlotUnitNode ({plot_category})")
            consistent = False
        else:
            logger.info(f"Node categories are consistent: {signal_category}")
            
    except Exception as e:
        logger.error(f"Failed to check node categories: {str(e)}")
        consistent = False
        
    return consistent

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
    
    # Check categories
    if not check_categories():
        # This is a warning, not a failure
        pass
    
    return success

# Run initialization when imported
initialize()