#!/usr/bin/env python
"""
Test script for PlotUnit visualization with a focus on testing settings buttons.

This script initializes PlotUnit with test signals and opens the settings view
to verify that the toggle and action buttons are working correctly.
It also provides an interactive menu to test individual buttons.
"""

import os
import sys
import time
import pygame

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.plot.plot_unit import PlotUnit
from src.plot.view_mode import ViewMode

def test_settings_buttons():
    """Test PlotUnit settings buttons functionality"""
    print("\n=== PlotUnit Settings Buttons Test ===\n")
    
    # Get the singleton instance
    plot = PlotUnit.get_instance()
    
    # Start visualization
    if not plot.running:
        plot.start()
        print("PlotUnit visualization started")
    else:
        print("PlotUnit visualization already running")
    
    # Wait for initialization
    wait_time = 0
    max_wait = 5
    while not plot.initialized and wait_time < max_wait:
        print(f"Waiting for initialization... ({wait_time+1}/{max_wait})")
        time.sleep(1.0)
        wait_time += 1
      # Load some test data
    print("Loading test signals...")
    plot.load_test_signals()
    
    # Switch to settings view
    print("Switching to SETTINGS view mode...")
    plot._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)  # Wait for view to render
    
    # Access the settings view
    if hasattr(plot, 'views') and ViewMode.SETTINGS in plot.views:
        settings_view = plot.views[ViewMode.SETTINGS]
        
        # Ensure event handler has reference to settings view
        if hasattr(plot, 'event_handler'):
            event_handler = plot.event_handler
            
            # Check settings view references
            if not hasattr(event_handler, 'settings_view') or not event_handler.settings_view:
                print("WARNING: Adding settings_view reference to event_handler")
                event_handler.settings_view = settings_view
            
            # Check PlotUnit reference in settings view
            if hasattr(settings_view, 'plot_unit') and settings_view.plot_unit:
                print("Settings view has reference to PlotUnit - OK")
            else:
                print("WARNING: Adding PlotUnit reference to settings_view")
                settings_view.plot_unit = plot
            
            # Display buttons for debugging
            if hasattr(settings_view, 'settings_buttons'):
                buttons = settings_view.settings_buttons
                print(f"\nFound {len(buttons)} buttons in settings view:")
                
                for i, (rect, key) in enumerate(buttons):
                    print(f"  Button {i+1}: {key} at {rect}")
            else:
                print("ERROR: No settings buttons found!")
    else:
        print("ERROR: Could not access settings view!")
    
    print("\n=== Interactive Testing Menu ===")
    print("Options:")
    print("  1. Auto-test all buttons")
    print("  2. Enter manual testing mode")
    print("  3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Auto-test all buttons
        print("\nAuto-testing all buttons...")
        if 'buttons' in locals() and len(buttons) > 0:
            for i, (rect, key) in enumerate(buttons):
                print(f"Testing button {i+1}: {key}")
                # Simulate click at the center of the button
                event_handler._handle_click(rect.centerx, rect.centery)
                time.sleep(1)  # Wait to see effects
        print("Auto-testing complete")
    elif choice == "2":
        # Manual testing instructions
        print("\n=== Manual Testing Mode ===")
        print("INSTRUCTIONS:")
        print("1. Try clicking on toggle buttons in the settings view")
        print("2. Try clicking on the action buttons (Reset Plots, Reset Registry)")
        print("3. Check the console output for confirmation of button actions")
        print("4. Press Ctrl+C to exit when done testing")
        print("===========================")
        
        # Keep the script running to allow interaction
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nManual testing completed")
    else:
        print("\nExiting test...")
    
    print("Test completed!")
    
if __name__ == "__main__":
    try:
        test_settings_buttons()
    except KeyboardInterrupt:
        print("\nExiting settings button test")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
