"""
PlotUnit Debug Script

This script provides a systematic approach to debug the PlotUnit visualization system
by initializing components one step at a time, with proper error handling.
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
logger = logging.getLogger('PlotUnitDebug')

from src.plot.plot_unit import PlotUnit
from src.plot.view_mode import ViewMode

def debug_step(step_name, func, *args, **kwargs):
    """Execute a debug step with proper error handling"""
    logger.info(f"Step: {step_name}")
    try:
        result = func(*args, **kwargs)
        logger.info(f"✅ {step_name} - Success")
        return result
    except Exception as e:
        logger.error(f"❌ {step_name} - Failed: {str(e)}")
        traceback.print_exc()
        return None

def main():
    logger.info("Starting PlotUnit debugging process")
    
    # Step 1: Initialize PlotUnit
    plot = debug_step("Get PlotUnit instance", 
                     PlotUnit.get_instance)
    
    if not plot:
        logger.error("Failed to get PlotUnit instance. Exiting.")
        return
    
    # Step 2: Start visualization in separated thread
    debug_step("Start visualization thread", 
              plot.start)
    
    # Step 3: Wait for initialization
    logger.info("Waiting for initialization")
    time.sleep(2.0)
    
    # Step 4: Check if initialization was successful
    if not debug_step("Check initialization status", 
                     lambda: plot.initialized):
        logger.error("PlotUnit failed to initialize. Continuing with fixes.")
    else:
        logger.info("PlotUnit initialized successfully")
    
    # Step 5: Fix view mode
    debug_step("Set view mode to RAW",
              lambda: setattr(plot, 'current_mode', ViewMode.RAW))
    
    # Step 6: Update sidebar and event handler
    if hasattr(plot, 'sidebar') and plot.sidebar:
        debug_step("Update sidebar mode",
                  lambda: setattr(plot.sidebar, 'current_mode', ViewMode.RAW))
    
    if hasattr(plot, 'event_handler') and plot.event_handler:
        debug_step("Update event handler mode",
                  lambda: setattr(plot.event_handler, 'current_mode', ViewMode.RAW))
                  
    # Step 7: Initialize PlotContainer
    if hasattr(plot, 'plot_container') and plot.plot_container:
        debug_step("Configure plot container",
                  lambda: setattr(plot.plot_container, 'twin_view', False))
                  
        if hasattr(plot, 'views') and ViewMode.RAW in plot.views:
            debug_step("Set plot container plots",
                      lambda: setattr(plot.plot_container, 'plots', 
                                     [plot.views[ViewMode.RAW]]))
            
            debug_step("Update plot container layout",
                      plot.plot_container.update_layout)
    
    # Step 8: Register some test signals
    debug_step("Generate test signals",
              plot.load_test_signals)
    
    # Summary
    logger.info("Debug process completed. The visualization should now be running.")
    logger.info("Check for any errors above and manually fix if needed.")
    
    # Keep running for observation
    try:
        logger.info("Press Ctrl+C to exit")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting debug script")

if __name__ == "__main__":
    main()
