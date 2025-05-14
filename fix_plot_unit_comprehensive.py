"""
PlotUnit Comprehensive Fix Script

This script addresses known issues with the PlotUnit visualization system:
1. Proper initialization sequence
2. Ensuring the proper sequence of component initialization
3. Setting up container, sidebar, and status bar correctly
4. Initializing each view
5. Bridging to the signal registry
"""

import sys
import os
import time
import logging
import traceback
import pygame

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('PlotUnitFix')

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import required components
from src.plot.plot_unit import PlotUnit
from src.plot.view_mode import ViewMode
from src.registry.plot_registry import PlotRegistry
from src.registry.plot_registry_integration import PlotRegistryIntegration

def fix_step(step_name, func, *args, **kwargs):
    """Execute a fix step with proper error handling"""
    logger.info(f"Fix step: {step_name}")
    try:
        result = func(*args, **kwargs)
        logger.info(f"✅ {step_name} - Success")
        return result
    except Exception as e:
        logger.error(f"❌ {step_name} - Failed: {str(e)}")
        traceback.print_exc()
        return None

def fix_plot_unit():
    """Comprehensive PlotUnit fix routine"""
    logger.info("Starting comprehensive PlotUnit fix")
    
    # Step 1: Get instances of all major components
    plot_unit = fix_step("Get PlotUnit instance", 
                        PlotUnit.get_instance)
    
    plot_registry = fix_step("Get PlotRegistry instance",
                            PlotRegistry.get_instance)
    
    integration = fix_step("Get PlotRegistryIntegration instance",
                          PlotRegistryIntegration.get_instance)
    
    if not plot_unit or not plot_registry or not integration:
        logger.error("Failed to get required component instances. Exiting.")
        return
    
    # Step 2: Connect the integration to PlotUnit
    fix_step("Connect PlotUnit to integration",
            integration.connect_plot_unit, plot_unit)
    
    # Step 3: Start visualization thread
    fix_step("Start visualization thread", 
           plot_unit.start)
    
    # Step 4: Wait for initialization
    logger.info("Waiting for initialization")
    time.sleep(2.0)
    
    # Step 5: Fix view mode
    fix_step("Set view mode to RAW",
           plot_unit._set_mode, ViewMode.RAW)
    
    # Step 6: Ensure views are properly initialized
    if hasattr(plot_unit, 'views'):
        for mode_name, mode in [(m.name, m) for m in ViewMode]:
            if mode in plot_unit.views:
                fix_step(f"Initialize {mode_name} view",
                       lambda m=mode: plot_unit.views[m].set_rect(
                           pygame.Rect(
                               plot_unit.sidebar_width,
                               plot_unit.status_bar_height,
                               plot_unit.width - plot_unit.sidebar_width,
                               plot_unit.height - plot_unit.status_bar_height
                           )
                       ))
    
    # Step 7: Configure PlotContainer
    if hasattr(plot_unit, 'plot_container') and plot_unit.plot_container:
        fix_step("Configure plot container dimensions",
               plot_unit.plot_container.set_window_rect,
               pygame.Rect(0, 0, plot_unit.width, plot_unit.height),
               plot_unit.sidebar_width)
        
        fix_step("Set plot container to single view mode",
               plot_unit.plot_container.set_twin_view, False)
        
        if hasattr(plot_unit, 'views') and ViewMode.RAW in plot_unit.views:
            fix_step("Add RAW view to plot container",
                   lambda: setattr(plot_unit.plot_container, 'plots', 
                                  [plot_unit.views[ViewMode.RAW]]))
            
        fix_step("Update plot container layout",
               plot_unit.plot_container.update_layout)
    
    # Step 8: Generate test signal for visualization
    fix_step("Generate test signals",
           plot_unit.load_test_signals)
    
    # Step 9: Force a render update
    fix_step("Force render update",
           plot_unit._render)
    
    # Summary
    logger.info("\nPlotUnit fixes applied successfully!")
    logger.info("The visualization should now be running with proper initialization.")
    
    # Return a handle to the objects to prevent garbage collection
    return plot_unit, plot_registry, integration

if __name__ == "__main__":
    # Run the fix
    fixed_components = fix_plot_unit()
    
    # Keep the script running to prevent window closure
    try:
        logger.info("Press Ctrl+C to exit")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting fix script")
