#!/usr/bin/env python
"""
Minimal PlotUnit Settings Test

This is a very simple test script focused only on testing the settings buttons.
It has minimal dependencies and prints detailed debug information.
"""

import os
import sys
import time
import traceback

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import required modules
    import pygame
    from src.plot.plot_unit import PlotUnit
    from src.plot.view_mode import ViewMode
    
    print("\n=== Minimal PlotUnit Settings Test ===\n")
    
    # Get singleton instance
    plot_unit = PlotUnit.get_instance()
    
    # Start visualization if not already running
    if not plot_unit.running:
        print("Starting PlotUnit visualization...")
        plot_unit.start()
    else:
        print("PlotUnit visualization already running")
    
    # Wait for initialization
    wait_time = 0
    while not plot_unit.initialized and wait_time < 5:
        print(f"Waiting for initialization ({wait_time+1}/5)...")
        time.sleep(1.0)
        wait_time += 1
    
    if not plot_unit.initialized:
        print("WARNING: PlotUnit might not be properly initialized")
    else:
        print("PlotUnit initialized successfully!")
    
    # Switch to settings view
    print("\nSwitching to SETTINGS view...")
    plot_unit._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)
    
    # Print information about the settings view
    print("\n--- Settings View Information ---")
    if ViewMode.SETTINGS in plot_unit.views:
        settings_view = plot_unit.views[ViewMode.SETTINGS]
        
        # Check settings buttons
        if hasattr(settings_view, 'settings_buttons'):
            buttons = settings_view.settings_buttons
            print(f"Found {len(buttons)} buttons in settings view")
            
            for i, (rect, key) in enumerate(buttons):
                button_type = "Action" if key in ['reset_plots', 'reset_registry'] else "Toggle"
                print(f"Button {i+1}: {button_type} '{key}' at {rect}")
                print(f"  Click coordinates: ({rect.centerx}, {rect.centery})")
        else:
            print("ERROR: settings_view has no settings_buttons attribute")
    else:
        print("ERROR: Could not access SETTINGS view")
    
    # Fix event handler references
    print("\n--- Checking Event Handler Setup ---")
    if hasattr(plot_unit, 'event_handler'):
        event_handler = plot_unit.event_handler
        
        # Check settings_view reference
        if hasattr(event_handler, 'settings_view') and event_handler.settings_view:
            print("Event handler has reference to settings view ✓")
        else:
            print("WARNING: Event handler has no settings_view reference")
            event_handler.settings_view = settings_view
            print("Added settings_view reference to event_handler")
        
        # Check plot_unit reference in settings_view
        if hasattr(settings_view, 'plot_unit') and settings_view.plot_unit:
            print("Settings view has reference to plot_unit ✓")
        else:
            print("WARNING: Settings view has no plot_unit reference")
            settings_view.plot_unit = plot_unit
            print("Added plot_unit reference to settings_view")
    else:
        print("ERROR: plot_unit has no event_handler attribute")
    
    # Print instructions
    print("\n--- Testing Instructions ---")
    print("1. A window with the PlotUnit visualization should be open")
    print("2. The Settings view should be displayed with buttons")
    print("3. Try clicking on settings buttons and action buttons")
    print("4. When clicking on a button, watch this console for debug messages")
    print("5. If buttons work correctly, you should see toggle messages here")
    print("\nPress Ctrl+C when finished testing")
    
    # Keep running until user exits
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest complete!")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    traceback.print_exc()
