"""
Event handler for PlotUnit visualization system

This module provides the event handling functionality for the PlotUnit visualization system.
"""

import pygame

# Import local modules
from modules.view_mode import ViewMode

class EventHandler:
    """Event handler for the PlotUnit system."""
    
    def __init__(self, sidebar, settings_view=None):
        """Initialize the event handler.
        
        Args:
            sidebar: The sidebar component
            settings_view: The settings view component (optional)
        """
        self.sidebar = sidebar
        self.settings_view = settings_view
        self.current_mode = sidebar.current_mode
        print(f"EventHandler initialized with current_mode: {self.current_mode.name}")
    
    def process_events(self):
        """Process pygame events.
        
        Returns:
            False if the application should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    x, y = event.pos
                    self._handle_click(x, y)
        
        return True
    
    def _handle_click(self, x, y):
        """Handle a mouse click.
        
        Args:
            x: The x position of the click
            y: The y position of the click
        """
        print(f"Click detected at ({x}, {y})")
        if x < self.sidebar.width:
            # Click in sidebar area
            mode_index = self.sidebar.handle_click(y)
            if mode_index is not None:
                self._update_mode(mode_index)
        elif self.current_mode == ViewMode.SETTINGS and self.settings_view:
            # Pass click to settings view
            self._handle_settings_click(x, y)
    
    def _update_mode(self, mode_index):
        """Update the current view mode.
        
        Args:
            mode_index: The index of the new mode
        """
        # Map index to ViewMode
        mode_mapping = {
            0: ViewMode.RAW,
            1: ViewMode.PROCESSED,
            2: ViewMode.TWIN,
            3: ViewMode.SETTINGS
        }
        
        if mode_index in mode_mapping:
            self.current_mode = mode_mapping[mode_index]
            self.sidebar.current_mode = self.current_mode
            print(f"Mode updated to: {self.current_mode.name}")
    
    def _handle_settings_click(self, x, y):
        """Handle a click in the settings view.
        
        Args:
            x: The x position of the click
            y: The y position of the click
        """
        if not self.settings_view:
            print("Settings view not available")
            return
        
        print(f"Settings view click at: ({x}, {y})")
        
        # Check if a settings button was clicked
        found_button = False
        for button_rect, setting_key in self.settings_view.settings_buttons:
            if button_rect.collidepoint(x, y):
                print(f"Button clicked: {setting_key}")
                self._toggle_setting(setting_key)
                found_button = True
                break
        
        if not found_button:
            print(f"No button found at position: ({x}, {y})")
            print(f"Available buttons: {len(self.settings_view.settings_buttons)}")
            
    def _toggle_setting(self, setting_key):
        """Toggle a setting or trigger an action.
        
        Args:
            setting_key: The setting key or action key
            
        Returns:
            True if an action was triggered, False otherwise
        """
        print(f"Toggling setting: {setting_key}")
        
        # Special handling for action buttons
        if setting_key == 'reset_plots':
            print("Reset plots action triggered")
            # Flash the button by temporarily changing its color
            if hasattr(self, 'settings_view') and self.settings_view:
                for rect, key in self.settings_view.settings_buttons:
                    if key == 'reset_plots':
                        # We'll just trigger the action without visual feedback in this demo
                        break
            return True
        
        elif setting_key == 'reset_registry':
            print("Reset registry action triggered")
            # Same visual feedback approach
            return True
        
        # Toggle regular settings
        if setting_key in self.settings_view.settings:
            self.settings_view.settings[setting_key] = not self.settings_view.settings[setting_key]
            print(f"Setting '{setting_key}' toggled to: {self.settings_view.settings[setting_key]}")
        
        return False
    
    def get_current_mode(self):
        """Get the current view mode.
        
        Returns:
            The current view mode
        """
        return self.current_mode
