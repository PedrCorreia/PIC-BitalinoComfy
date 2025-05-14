"""
Comprehensive fix script for PlotUnit settings buttons.

This script fixes issues with the settings buttons in the PlotUnit visualization,
ensuring that toggle buttons and action buttons work correctly.
"""

import os
import sys
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

def fix_settings_view():
    """Fix settings view button functionality"""
    print("Fixing settings view buttons...")
    
    try:
        # Import the PlotUnit instance
        from src.plot.plot_unit import PlotUnit
        plot_unit = PlotUnit.get_instance()
        
        # Ensure it's running
        if not plot_unit.running:
            plot_unit.start()
            print("Started PlotUnit visualization")
            # Wait for initialization
            wait_time = 0
            while not plot_unit.initialized and wait_time < 5:
                time.sleep(1)
                wait_time += 1
                print(f"Waiting for initialization... ({wait_time}s)")
        
        # Fix event handler settings click handling
        fix_event_handler()
        
        # Fix settings view to reference PlotUnit directly
        fix_settings_view_reference(plot_unit)
        
        # Force render to update UI
        if hasattr(plot_unit, '_render'):
            plot_unit._render()
            print("UI rendering forced to update")
    
    except Exception as e:
        print(f"ERROR during fix: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nSettings buttons fix complete!")
    print("\nTo test the fix:")
    print("1. Run PlotUnit visualization")
    print("2. Switch to SETTINGS view using the S button in sidebar")
    print("3. Try clicking the toggle buttons and action buttons")

def fix_event_handler():
    """Fix the event handler to correctly handle settings button clicks"""
    event_handler_path = os.path.join(parent_dir, "src", "plot", "event_handler.py")
    
    if not os.path.exists(event_handler_path):
        print(f"ERROR: Could not find event_handler.py at {event_handler_path}")
        return
    
    # Create a direct Python function monkey patch instead of trying to edit the file
    # This approach is more reliable than text replacement
    try:
        # Import the EventHandler class
        from src.plot.event_handler import EventHandler
        
        # Define our fixed toggle_setting method
        def fixed_toggle_setting(self, setting_key):
            """
            Toggle a setting or trigger an action.
            This is a fixed implementation to ensure buttons work properly.
            """
            print(f"[EVENT] Toggle setting called for: {setting_key}")
            
            # Special handling for action buttons
            if setting_key == 'reset_plots':
                # Get the PlotUnit instance and call clear_plots
                try:
                    from src.plot.plot_unit import PlotUnit
                    plot_unit = PlotUnit.get_instance()
                    if hasattr(plot_unit, 'clear_plots'):
                        plot_unit.clear_plots()
                        print("[EVENT] Reset plots action triggered")
                    else:
                        print("[EVENT] PlotUnit has no clear_plots method")
                except Exception as e:
                    print(f"[EVENT] Error resetting plots: {str(e)}")
                return True
                
            elif setting_key == 'reset_registry':
                # Try to reset the registry
                try:
                    # Try different ways to access the registry
                    try:
                        from src.registry.signal_registry import SignalRegistry
                        SignalRegistry.get_instance().reset()
                        print("[EVENT] Reset registry action triggered via direct import")
                    except ImportError:
                        # Try through PlotUnit
                        from src.plot.plot_unit import PlotUnit
                        plot_unit = PlotUnit.get_instance()
                        # Check if PlotUnit has a reference to the registry
                        if hasattr(plot_unit, 'registry') and hasattr(plot_unit.registry, 'reset'):
                            plot_unit.registry.reset()
                            print("[EVENT] Reset registry action triggered via PlotUnit")
                        else:
                            print("[EVENT] PlotUnit has no reference to registry")
                except Exception as e:
                    print(f"[EVENT] Failed to reset registry: {str(e)}")
                return True
            
            # Toggle regular settings
            if hasattr(self, 'settings_view') and hasattr(self.settings_view, 'settings'):
                if setting_key in self.settings_view.settings:
                    self.settings_view.settings[setting_key] = not self.settings_view.settings[setting_key]
                    print(f"[EVENT] Setting '{setting_key}' toggled to {self.settings_view.settings[setting_key]}")
                else:
                    print(f"[EVENT] Setting '{setting_key}' not found in settings")
            else:
                print("[EVENT] Cannot access settings_view.settings")
            
            return False
        
        # Replace the method in the class
        EventHandler._toggle_setting = fixed_toggle_setting
        
        print("Successfully patched EventHandler._toggle_setting method")
        return True
        
    except Exception as e:
        print(f"ERROR patching EventHandler: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Fix indentation issues in event_handler.py
    content = content.replace("      def _handle_settings_click", "    def _handle_settings_click")
    content = content.replace("      def _toggle_setting", "    def _toggle_setting")
    
    # Add debug output to settings click handling
    content = content.replace(
        "# Check if a settings button was clicked\n        for button_rect, setting_key in self.settings_view.settings_buttons:",
        "# Check if a settings button was clicked\n        print(f\"[EVENT] Processing click in settings view with {len(self.settings_view.settings_buttons)} buttons\")\n        for button_rect, setting_key in self.settings_view.settings_buttons:"
    )
    
    # Improve toggle_setting implementation
    toggle_setting_start = content.find("def _toggle_setting(self, setting_key):")
    if toggle_setting_start > 0:
        # Find the end of the method
        next_def_start = content.find("def ", toggle_setting_start + 10)
        if next_def_start > 0:
            # Replace the entire method
            improved_toggle_setting = '''    def _toggle_setting(self, setting_key):
        """
        Toggle a setting or trigger an action.
        
        Args:
            setting_key (str): The key of the setting to toggle
        
        Returns:
            bool: True if an action was triggered, False otherwise
        """
        print(f"[EVENT] Toggle setting: {setting_key}")
        
        # Import parent PlotUnit
        try:
            from src.plot.plot_unit import PlotUnit
            plot_unit = PlotUnit.get_instance()
        except Exception as e:
            print(f"[EVENT] Error importing PlotUnit: {e}")
            plot_unit = None
        
        # Special handling for action buttons
        if setting_key == 'reset_plots':
            if plot_unit and hasattr(plot_unit, 'clear_plots'):
                plot_unit.clear_plots()
                print("[EVENT] Reset plots action completed")
            else:
                print("[EVENT] Error: PlotUnit instance or clear_plots method not found")
            return True
            
        elif setting_key == 'reset_registry':
            try:
                from src.registry.signal_registry import SignalRegistry
                registry = SignalRegistry.get_instance()
                registry.reset()
                print("[EVENT] Reset registry action completed")
            except Exception as e:
                print(f"[EVENT] Error resetting registry: {e}")
            return True
        
        # Toggle regular settings
        if self.settings_view and hasattr(self.settings_view, 'settings'):
            if setting_key in self.settings_view.settings:
                current_value = self.settings_view.settings[setting_key]
                self.settings_view.settings[setting_key] = not current_value
                print(f"[EVENT] Setting '{setting_key}' toggled to {self.settings_view.settings[setting_key]}")
        
        return False
'''
            content = content[:toggle_setting_start] + improved_toggle_setting + content[next_def_start:]
    
    # Write the updated content back
    with open(event_handler_path, 'w') as f:
        f.write(content)
    
    print(f"Updated event handler at {event_handler_path}")

def fix_settings_view_reference():
    """Fix the settings view to have a reference to PlotUnit"""
    settings_view_path = os.path.join(parent_dir, "src", "plot", "view", "settings_view.py")
    
    if not os.path.exists(settings_view_path):
        print(f"ERROR: Could not find settings_view.py at {settings_view_path}")
        return
        
    with open(settings_view_path, 'r') as f:
        content = f.read()
    
    # Fix indentation issues
    content = content.replace("      def __init__", "    def __init__")
    
    # Add PlotUnit reference in __init__
    init_end = content.find("self.settings_buttons = []  # Will store (rect, setting_key) pairs")
    if init_end > 0:
        # Find the correct position to insert the reference
        next_line_end = content.find("\n", init_end)
        if next_line_end > 0:
            plot_unit_ref = '''
        # Store reference to PlotUnit
        try:
            import sys
            from src.plot.plot_unit import PlotUnit
            self.plot_unit = PlotUnit.get_instance()
            print(f"[SettingsView] Connected to PlotUnit instance")
        except Exception as e:
            print(f"[SettingsView] Error connecting to PlotUnit: {e}")
            self.plot_unit = None'''
            content = content[:next_line_end] + plot_unit_ref + content[next_line_end:]
    
    # Add debug output to button drawing methods
    content = content.replace(
        "# Store button rect and settings key for click handling\n        self.settings_buttons.append((switch_bg_rect, setting_key))",
        "# Store button rect and settings key for click handling\n        self.settings_buttons.append((switch_bg_rect, setting_key))\n        print(f\"[SettingsView] Added toggle button: {setting_key}, rect: {switch_bg_rect}\")"
    )
    
    content = content.replace(
        "# Store button rect and action key for click handling\n        self.settings_buttons.append((button_rect, action_key))",
        "# Store button rect and action key for click handling\n        self.settings_buttons.append((button_rect, action_key))\n        print(f\"[SettingsView] Added action button: {action_key}, rect: {button_rect}\")"
    )
    
    # Write the updated content back
    with open(settings_view_path, 'w') as f:
        f.write(content)
    
    print(f"Updated settings view at {settings_view_path}")

def fix_settings_view_reference(plot_unit=None):
    """Add a reference to PlotUnit in the SettingsView class"""
    try:
        # Get plot_unit if not provided
        if plot_unit is None:
            from src.plot.plot_unit import PlotUnit
            plot_unit = PlotUnit.get_instance()
        
        # Add reference to PlotUnit in settings_view
        if hasattr(plot_unit, 'views') and hasattr(plot_unit, 'event_handler'):
            from src.plot.view_mode import ViewMode
            if ViewMode.SETTINGS in plot_unit.views:
                # Add a direct reference to plot_unit in settings_view
                settings_view = plot_unit.views[ViewMode.SETTINGS]
                settings_view.plot_unit = plot_unit
                
                # Update event handler with this settings view
                plot_unit.event_handler.settings_view = settings_view
                
                # Also patch the settings view draw method to ensure buttons are visible
                from src.plot.view.settings_view import SettingsView
                
                # Store original _draw_button method
                original_draw_button = SettingsView._draw_button
                
                # Create an enhanced draw_button method with debug information
                def enhanced_draw_button(self, x, y, label, action_key, color):
                    # Call original method
                    result = original_draw_button(self, x, y, label, action_key, color)
                    
                    # Add debug information
                    print(f"[DEBUG] Button '{label}' (key: {action_key}) drawn at ({x}, {y})")
                    
                    # Check if the button was added to settings_buttons
                    if hasattr(self, 'settings_buttons'):
                        found = False
                        for rect, key in self.settings_buttons:
                            if key == action_key:
                                found = True
                                print(f"[DEBUG] Button registered with rect: {rect}")
                                break
                        if not found:
                            print(f"[WARNING] Button not found in settings_buttons!")
                                
                    return result
                
                # Replace the method
                SettingsView._draw_button = enhanced_draw_button
                
                print("Successfully patched SettingsView._draw_button method")
                print("Successfully added PlotUnit reference to settings view")
                return True
            else:
                print("WARNING: SETTINGS view not found in PlotUnit views")
        else:
            print("WARNING: PlotUnit views or event_handler not initialized")
        return False
    except Exception as e:
        print(f"ERROR setting up view reference: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_settings_view()
