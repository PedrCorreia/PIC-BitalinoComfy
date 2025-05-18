#!/usr/bin/env python
"""
Simple UI test script for PlotUnit with static plots.

This script initializes the PlotUnit with static debug plots
and allows switching between different view modes.
"""

import os
import sys
import time
import pygame
import traceback

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"Project root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize pygame
pygame.init()

# Import the ViewMode enum first
try:
    from src.plot.view_mode import ViewMode
    print("Successfully imported ViewMode")
except ImportError as e:
    print(f"ERROR: ViewMode import failed: {e}")
    print("Creating fallback ViewMode enum")
    
    from enum import Enum
    class ViewMode(Enum):
        RAW = 0
        PROCESSED = 1
        TWIN = 2
        STACKED = 3
        SETTINGS = 4

# Now try to import PlotUnit
try:
    from src.plot.plot_unit import PlotUnit
    print("Successfully imported PlotUnit")
except ImportError as e:
    print(f"ERROR: PlotUnit import failed: {e}")
    print("Cannot continue without PlotUnit")
    sys.exit(1)

# Try to import or create debug_plots
try:
    from src.plot.utils.debug_plots import initialize_test_plots, update_test_plots
    print("Successfully imported debug plot utilities")
except ImportError:
    print("WARNING: Could not import debug_plots, creating basic functions")
    
    import numpy as np
    
    def initialize_test_plots(plot_unit):
        """Fallback function to initialize plots"""
        print("Creating basic test plots")
        try:
            # Create basic signals
            t = np.linspace(0, 10, 1000)
            plot_unit.queue_data("test_signal_1", np.sin(2 * np.pi * 0.5 * t))
            plot_unit.queue_data("test_signal_2", np.sin(2 * np.pi * 0.3 * t))
            return True
        except Exception as e:
            print(f"Error creating basic test plots: {e}")
            return False
    
    def update_test_plots(plot_unit):
        """Fallback function to update plots"""
        pass

def main():
    """Main function to test PlotUnit UI with static plots."""
    print("\n=== PlotUnit UI Test with Static Plots ===\n")
    
    try:
        # Get PlotUnit instance
        plot = PlotUnit.get_instance()
        print("PlotUnit instance created")
        
        # Start visualization
        plot.start()
        print("PlotUnit visualization started")
        
        # Wait for initialization
        wait_seconds = 0
        while not plot.initialized and wait_seconds < 5:
            print(f"Waiting for initialization... ({wait_seconds+1}/5)")
            time.sleep(1.0)
            wait_seconds += 1
        
        if not plot.initialized:
            print("WARNING: PlotUnit did not initialize within timeout")
        else:
            print("PlotUnit initialized successfully")
        
        # Initialize with test plots
        print("Adding test plots...")
        success = initialize_test_plots(plot)
        if success:
            print("Test plots loaded successfully")
        else:
            print("WARNING: Failed to load test plots")
        
        # Print available tabs/views
        print("\n--- Available View Modes ---")
        for mode in ViewMode:
            print(f"View: {mode.name}")
        
        # Cycle through views automatically to demonstrate
        print("\nCycling through view modes...")
        for mode in ViewMode:
            print(f"Setting view mode to {mode.name}")
            if hasattr(plot, '_set_mode'):
                try:
                    plot._set_mode(mode)
                    time.sleep(2)  # View each mode for 2 seconds
                except Exception as e:
                    print(f"Error setting mode {mode.name}: {e}")
        
        # Set back to RAW view for interactive use
        if hasattr(plot, '_set_mode'):
            plot._set_mode(ViewMode.RAW)
        
        # Print user instructions
        print("\n=== Instructions ===")
        print("1. Look at the PlotUnit window that opened")
        print("2. Click on tabs on the left sidebar to switch views")
        print("3. All tabs should show static plots for testing")
        print("\nPress Ctrl+C to exit")
        
        # Periodically update plots for better visual effect
        try:
            update_count = 0
            while True:
                time.sleep(1)
                update_count += 1
                
                # Update plots every 5 seconds
                if update_count % 5 == 0:
                    print("Updating test plots...")
                    update_test_plots(plot)
        except KeyboardInterrupt:
            print("\nExiting...")
        
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    main()
