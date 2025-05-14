#!/usr/bin/env python
"""
PlotUnit Test Application

This is a standalone test application for the PlotUnit visualization system.
It provides a clean, fixed structure for debugging button functionality and
testing the overall system without ComfyUI dependencies.

Features:
- Proper initialization of PlotUnit and its components
- Debug output for all interactions
- Easy testing of settings buttons and tab switching
- Simple structure for future enhancements

Usage:
    python plotunit_test_app.py
"""

import os
import sys
import time
import traceback

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pygame after path setup
import pygame

# Initialize pygame before imports that might use it
pygame.init()

class PlotUnitTestApp:
    """Test application for PlotUnit visualization system."""
    
    def __init__(self):
        """Initialize the test application."""
        self.running = False
        self.plot_unit = None
        self.initialized = False
        
        print("\n=== PlotUnit Test Application ===")
        print("Initializing components...\n")
        
    def setup(self):
        """Set up the PlotUnit visualization system."""
        try:
            # Import required components
            from src.plot.plot_unit import PlotUnit
            from src.plot.view_mode import ViewMode
            
            # Get PlotUnit singleton instance
            self.plot_unit = PlotUnit.get_instance()
            
            # Store references for easier debugging
            self.ViewMode = ViewMode
            
            # Print initialization message
            print("PlotUnit components imported successfully")
            return True
            
        except ImportError as e:
            print(f"ERROR: Failed to import required components: {str(e)}")
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error during setup: {str(e)}")
            traceback.print_exc()
            return False
    
    def start(self):
        """Start the PlotUnit visualization and test system."""
        if not self.setup():
            print("Setup failed. Cannot start application.")
            return False
        
        # Start PlotUnit visualization
        print("\nStarting PlotUnit visualization...")
        self.plot_unit.start()
        
        # Wait for initialization
        wait_time = 0
        max_wait = 5
        while not self.plot_unit.initialized and wait_time < max_wait:
            print(f"Waiting for initialization... ({wait_time+1}/{max_wait})")
            time.sleep(1)
            wait_time += 1
        
        if not self.plot_unit.initialized:
            print("WARNING: PlotUnit did not initialize within the timeout period")
        else:
            print("PlotUnit initialized successfully")
        
        # Load some test signals
        print("Loading test signals...")
        self.plot_unit.load_test_signals()
        print("Test signals loaded")
        
        # Print information about available views
        self._print_view_info()
        
        # Set running flag
        self.running = True
        self.initialized = True
        
        return True
    
    def _print_view_info(self):
        """Print information about available views and buttons."""
        print("\n--- Available Views ---")
        if hasattr(self.plot_unit, 'views'):
            for mode, view in self.plot_unit.views.items():
                print(f"View: {mode.name}")
                
                # Print special info for settings view
                if mode == self.ViewMode.SETTINGS and hasattr(view, 'settings_buttons'):
                    print(f"  Settings view has {len(view.settings_buttons)} buttons:")
                    for i, (rect, key) in enumerate(view.settings_buttons):
                        button_type = "Action" if key in ['reset_plots', 'reset_registry'] else "Toggle"
                        print(f"    {i+1}. {button_type} Button: '{key}', Position: {rect}")
        else:
            print("No views available")
        
        # Print event handler info
        print("\n--- Event Handler ---")
        if hasattr(self.plot_unit, 'event_handler'):
            event_handler = self.plot_unit.event_handler
            print(f"Event handler initialized: {event_handler is not None}")
            
            if hasattr(event_handler, 'settings_view'):
                print(f"Settings view reference: {event_handler.settings_view is not None}")
            else:
                print("No settings_view reference in event_handler")
                
            print(f"Current mode: {event_handler.get_current_mode().name}")
        else:
            print("No event_handler available")
    
    def switch_to_settings(self):
        """Switch to settings view."""
        if not self.initialized:
            print("Cannot switch view: PlotUnit not initialized")
            return False
        
        print("\nSwitching to SETTINGS view...")
        self.plot_unit._set_mode(self.ViewMode.SETTINGS)
        time.sleep(0.5)  # Allow time to render
        
        # Print settings buttons for click testing
        print("\n--- Settings Buttons for Testing ---")
        if self.ViewMode.SETTINGS in self.plot_unit.views:
            settings_view = self.plot_unit.views[self.ViewMode.SETTINGS]
            if hasattr(settings_view, 'settings_buttons'):
                for i, (rect, key) in enumerate(settings_view.settings_buttons):
                    button_type = "Action" if key in ['reset_plots', 'reset_registry'] else "Toggle"
                    print(f"{i+1}. {button_type}: '{key}' at position {rect}")
                    print(f"   Click at coordinates: ({rect.centerx}, {rect.centery})")
            else:
                print("No settings_buttons found in settings_view")
        else:
            print("Settings view not found")
        
        # Verify event handler setup
        self._verify_event_handler()
        
        return True
    
    def _verify_event_handler(self):
        """Verify that the event handler is properly set up for button clicks."""
        print("\n--- Event Handler Verification ---")
        if not hasattr(self.plot_unit, 'event_handler'):
            print("ERROR: No event_handler in plot_unit")
            return False
        
        event_handler = self.plot_unit.event_handler
        
        # Check settings view reference
        if not hasattr(event_handler, 'settings_view') or not event_handler.settings_view:
            print("ERROR: No settings_view reference in event_handler")
            
            # Try to fix it
            if self.ViewMode.SETTINGS in self.plot_unit.views:
                event_handler.settings_view = self.plot_unit.views[self.ViewMode.SETTINGS]
                print("FIXED: Added settings_view reference to event_handler")
            return False
        
        # Check PlotUnit reference in settings_view
        settings_view = self.plot_unit.views[self.ViewMode.SETTINGS]
        if not hasattr(settings_view, 'plot_unit') or not settings_view.plot_unit:
            print("ERROR: No plot_unit reference in settings_view")
            
            # Try to fix it
            settings_view.plot_unit = self.plot_unit
            print("FIXED: Added plot_unit reference to settings_view")
            return False
        
        print("Event handler setup looks good!")
        return True
    
    def print_instructions(self):
        """Print test instructions for the user."""
        print("\n" + "="*50)
        print("                Test Instructions")
        print("="*50)
        print("1. The PlotUnit window should be open now")
        print("2. Click on the 'Settings' tab on the left sidebar")
        print("3. Try clicking on toggle buttons (Light Mode, etc.)")
        print("4. Try clicking on action buttons (Reset Plots, etc.)")
        print("5. Watch this console for debug messages")
        print("\nPress Ctrl+C to exit the test")
        print("="*50 + "\n")
    
    def run(self):
        """Run the test application."""
        if not self.start():
            print("Failed to start application")
            return False
        
        # Switch to settings view for testing
        self.switch_to_settings()
        
        # Print test instructions
        self.print_instructions()
        
        # Keep running until user exits
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTest application stopped by user")
            self.running = False
        except Exception as e:
            print(f"\nERROR during execution: {str(e)}")
            traceback.print_exc()
            self.running = False
        
        print("\nTest application complete")
        return True


def check_event_handler_indentation():
    """Check for common indentation issues in the event handler file."""
    print("\nChecking for event handler indentation issues...")
    
    handler_path = os.path.join(project_root, 'src', 'plot', 'event_handler.py')
    if not os.path.exists(handler_path):
        print(f"Event handler file not found at: {handler_path}")
        return
    
    with open(handler_path, 'r') as f:
        lines = f.readlines()
    
    issues_found = False
    
    # Check specifically for the indentation issue in _handle_click and _toggle_setting
    for i, line in enumerate(lines):
        if "_handle_click" in line and "def " in line:
            # Check next few lines for indentation issues
            for j in range(i+1, min(i+20, len(lines))):
                if "if x < self.sidebar.width:" in lines[j]:
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    if indent != 8:  # Expected indentation
                        print(f"WARNING: Indentation issue found in _handle_click at line {j+1}")
                        print(f"Expected 8 spaces, found {indent}")
                        issues_found = True
                        
        if "_toggle_setting" in line and "def " in line:
            # Check for indentation of the method
            indent = len(line) - len(line.lstrip())
            if indent != 4:  # Expected indentation
                print(f"WARNING: Indentation issue found in _toggle_setting definition at line {i+1}")
                print(f"Expected 4 spaces, found {indent}")
                issues_found = True
    
    if not issues_found:
        print("No obvious indentation issues found in event_handler.py")
    else:
        print("Indentation issues found. Please fix them for proper functionality.")


if __name__ == "__main__":
    # Check for common issues before running
    check_event_handler_indentation()
    
    # Create and run the test app
    app = PlotUnitTestApp()
    app.run()
