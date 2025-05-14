#!/usr/bin/env python
"""
PlotUnit Debug - Simple debug file to test PlotUnit visualization

This file initializes the PlotUnit system for button and tab testing.
It is located in the project root to avoid import problems.
"""

import os
import sys
import time
import pygame

# Initialize pygame
pygame.init()

print("\n=== PlotUnit Debug ===\n")

try:
    # Import necessary components using absolute paths
    from src.plot.plot_unit import PlotUnit
    from src.plot.view_mode import ViewMode
    print("Successfully imported modules")
    
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
    
    print(f"PlotUnit initialized: {plot.initialized}")
    
    # Load some test signals
    print("Loading test signals...")
    plot.load_test_signals()
    print("Test signals loaded")
    
    # Switch to SETTINGS view
    print("\nSwitching to SETTINGS view...")
    plot._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)
    
    # Print settings buttons info
    if hasattr(plot, 'views') and ViewMode.SETTINGS in plot.views:
        settings_view = plot.views[ViewMode.SETTINGS]
        
        if hasattr(settings_view, 'settings_buttons'):
            buttons = settings_view.settings_buttons
            print(f"Found {len(buttons)} buttons in settings view")
            
            for i, (rect, key) in enumerate(buttons):
                print(f"{i+1}. Button: '{key}', Position: {rect}")
        else:
            print("No settings_buttons found in settings view")
    
    # Print instructions
    print("\n=== Instructions ===")
    print("1. Look at the PlotUnit window that opened")
    print("2. Click on tabs in the left sidebar to switch views")
    print("3. Try clicking on toggle buttons and action buttons in the Settings view")
    print("4. Watch this console for debug messages")
    
    # Keep program running for testing
    print("\nPress Ctrl+C to exit")
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nExiting debug program...")
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
