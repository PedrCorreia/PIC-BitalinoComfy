#!/usr/bin/env python
"""
Simplified test script for PlotUnit visualization with settings view.

This script helps test the functionality of the settings view buttons
by displaying debug information and switching to the Settings view.
"""

import os
import sys
import time

# Add the parent directory to the path for imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("\n=== PlotUnit Settings Buttons Test ===\n")

try:
    from src.plot.plot_unit import PlotUnit
    from src.plot.view_mode import ViewMode
    
    # Get the singleton instance
    plot = PlotUnit.get_instance()
    
    # Start visualization if not already running
    if not plot.running:
        plot.start()
        print("PlotUnit visualization started")
    else:
        print("PlotUnit visualization already running")
    
    # Wait for initialization
    wait_time = 0
    while not plot.initialized and wait_time < 5:
        print(f"Waiting for initialization... ({wait_time+1}/5)")
        time.sleep(1.0)
        wait_time += 1
    
    if not plot.initialized:
        print("WARNING: PlotUnit might not be properly initialized")
          # Load some test data
    plot.load_test_signals()
    print("Test signals loaded")
    
    # Switch to settings view
    print("\nSwitching to SETTINGS view...")
    plot._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)  # Wait for rendering
    
    # Display button information
    if hasattr(plot, 'views') and ViewMode.SETTINGS in plot.views:
        settings_view = plot.views[ViewMode.SETTINGS]
        
        print("\nSettings buttons information:")
        if hasattr(settings_view, 'settings_buttons'):
            buttons = settings_view.settings_buttons
            print(f"Found {len(buttons)} buttons in settings view")
            
            for i, (rect, key) in enumerate(buttons):
                print(f"  Button {i+1}: key={key}, rect={rect}")
                
                # For action buttons, add more debug info
                if key in ['reset_plots', 'reset_registry']:
                    print(f"    Action button: {key}")
                    print(f"    Position: x={rect.x}, y={rect.y}, w={rect.width}, h={rect.height}")
                    print(f"    Center: ({rect.centerx}, {rect.centery})")
        else:        print("ERROR: settings_view has no settings_buttons attribute")
    else:
        print("ERROR: Could not access SETTINGS view")
    
    # Test event handler connection
    print("\nChecking event handler setup...")
    if hasattr(plot, 'event_handler'):
        event_handler = plot.event_handler
        if hasattr(event_handler, 'settings_view') and event_handler.settings_view:
            print("Event handler has reference to settings view - OK")
            
            # Check if settings_view has PlotUnit reference
            if hasattr(settings_view, 'plot_unit') and settings_view.plot_unit:
                print("Settings view has reference to PlotUnit - OK")
            else:
                print("WARNING: Settings view has no reference to PlotUnit")
                # Add reference
                settings_view.plot_unit = plot
                print("Added PlotUnit reference to settings view")
        else:
            print("WARNING: Event handler has no reference to settings view")
            # Add reference
            event_handler.settings_view = settings_view
            print("Added settings view reference to event handler")
    else:
        print("ERROR: Plot unit has no event_handler attribute")
      # Instructions for manual testing
    print("\n--- Instructions for Manual Testing ---")
    print("1. Look at the PlotUnit window that should now be open")
    print("2. It should be showing the Settings view with toggles and buttons")
    print("3. Try clicking on the toggle buttons to change settings")
    print("4. Try clicking on the 'Reset Plots' or 'Reset Registry' action buttons")
    print("5. Check this console for debug messages related to clicks")
    
    print("\nKeep this script running while you test the buttons.")
    print("Press Ctrl+C when done testing.\n")
      # Keep the script running for manual testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest complete! Exiting...")
        
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    
if __name__ == "__main__":
    print("\nPlotUnit Settings Test complete.")
    print("If you want to run the test again, execute: python simple_settings_test.py")
