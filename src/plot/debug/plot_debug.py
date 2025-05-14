#!/usr/bin/env python
"""
PlotUnit Debug - Single file to test PlotUnit visualization

This standalone script initializes the PlotUnit visualization system and allows
for testing button functionality and tab switching with debug output.

Note: This script must be run from the project root directory to avoid import errors.
"""

import os
import sys
import time
import pygame

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"Project root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize pygame before imports
pygame.init()

# Now use absolute imports
try:
    from src.plot.plot_unit import PlotUnit
    from src.plot.view_mode import ViewMode
    print("Successfully imported modules")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    print("Make sure you run this script from the project root with:")
    print("  python -m src.plot.debug.plot_debug")
    sys.exit(1)
def main():
    """Main function to test PlotUnit visualization system"""
    print("\n=== PlotUnit Debug ===\n")
    
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
    plot.load_test_signals()
    print("Test signals loaded")
    
    # Print available tabs/views
    print("\n--- Available Tabs ---")
    if hasattr(plot, 'views'):
        for mode, view in plot.views.items():
            print(f"Tab: {mode.name}")
    
    # Switch to SETTINGS view
    print("\nSwitching to SETTINGS view...")
    plot._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)  # Wait for rendering
    
    # Print settings buttons for testing
    print("\n--- Settings Buttons ---")
    if hasattr(plot, 'views') and ViewMode.SETTINGS in plot.views:
        settings_view = plot.views[ViewMode.SETTINGS]
        
        if hasattr(settings_view, 'settings_buttons'):
            buttons = settings_view.settings_buttons
            print(f"Found {len(buttons)} buttons in settings view")
            
            for i, (rect, key) in enumerate(buttons):
                print(f"{i+1}. Button: '{key}', Position: {rect}, Center: ({rect.centerx}, {rect.centery})")
        else:
            print("No settings_buttons attribute in settings view")
    else:
        print("No SETTINGS view found")
    
    # Print event handler info
    print("\n--- Event Handler ---")
    if hasattr(plot, 'event_handler'):
        event_handler = plot.event_handler
        print(f"Event handler exists: {event_handler is not None}")
        
        if hasattr(event_handler, 'settings_view'):
            print(f"Settings view reference: {event_handler.settings_view is not None}")
        
        if hasattr(event_handler, 'current_mode'):
            print(f"Current mode: {event_handler.current_mode.name}")
    else:
        print("No event_handler attribute found")
    
    # Print user instructions
    print("\n=== Instructions ===")
    print("1. Look at the PlotUnit window that opened")
    print("2. Click on tabs on the left sidebar to switch views")
    print("3. Go to Settings tab and try clicking on toggle buttons")
    print("4. Try clicking on action buttons (Reset Plots, etc.)")
    print("5. Watch this console for debug messages")
    print("\nPress Ctrl+C to exit")
    
    # Keep running for interactive testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    return True

if __name__ == "__main__":
    main()
