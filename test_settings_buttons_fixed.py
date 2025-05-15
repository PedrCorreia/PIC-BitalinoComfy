#!/usr/bin/env python
"""
PlotUnit Settings Button Test

This script tests the settings buttons in the PlotUnit system after applying the fixes.
It loads the necessary components and displays the settings view for testing.
"""

import sys
import os
import time
import pygame
import threading

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from src.plot.constants import *
from src.plot.view_mode import ViewMode
from src.plot.plot_unit import PlotUnit

def main():
    """Test the PlotUnit settings buttons."""
    print("\n=== PlotUnit Settings Button Test ===\n")
    
    # Initialize pygame
    pygame.init()
    
    # Create PlotUnit instance
    plot_unit = PlotUnit.get_instance()
    print("PlotUnit instance created")
    
    # Start visualization
    plot_unit.start()
    print("PlotUnit visualization started")
    
    # Wait for initialization
    wait_seconds = 0
    while not plot_unit.initialized and wait_seconds < 5:
        print(f"Waiting for initialization... ({wait_seconds+1}/5)")
        time.sleep(1.0)
        wait_seconds += 1
    
    if not plot_unit.initialized:
        print("PlotUnit failed to initialize within the timeout period")
        return False
    
    print("PlotUnit initialized successfully")
    
    # Switch to SETTINGS view
    print("\nSwitching to SETTINGS view...")
    plot_unit._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)
    
    # Print instructions
    print("\n=== Instructions ===")
    print("1. Look at the PlotUnit window that opened")
    print("2. Click on tabs in the left sidebar to switch views")
    print("3. Try clicking on toggle buttons and action buttons in the Settings view")
    print("4. Watch this console for debug messages")
    print("\nPress Ctrl+C to exit")
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting test program...")
    
    return True

if __name__ == "__main__":
    main()
