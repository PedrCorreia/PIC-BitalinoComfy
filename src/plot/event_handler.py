"""
EventHandler module for the PlotUnit visualization system.

This module manages event processing for the PlotUnit system,
handling user input and system events.
"""

import pygame
from .constants import *
from .view_mode import ViewMode

class EventHandler:
    """
    Event handler for the PlotUnit visualization system.
    
    This class manages event processing, including mouse and keyboard events,
    and updates the application state accordingly.
    
    Attributes:
        sidebar (Sidebar): The sidebar component for navigation
        current_mode (ViewMode): The currently active view mode
        settings_view (SettingsView): The settings view for handling settings clicks
        button_controller (ButtonController): The button controller for handling button events
    """
    def __init__(self, sidebar, settings_view=None, button_controller=None):
        """
        Initialize the event handler.
        
        Args:
            sidebar (Sidebar): The sidebar component for navigation
            settings_view (SettingsView, optional): The settings view for handling settings clicks
            button_controller (ButtonController, optional): The button controller for handling button events
        """
        self.sidebar = sidebar
        self.settings_view = settings_view
        self.button_controller = button_controller
        self.current_mode = sidebar.current_mode
        
    def process_events(self):
        """
        Process pygame events and update application state.
        
        Returns:
            bool: True if the application should continue running, False if it should exit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            # Give button controller first chance to handle events
            if self.button_controller and self.button_controller.handle_events(event):
                continue
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    x, y = event.pos
                    self._handle_click(x, y)
        
        return True
    
    def _handle_click(self, x, y):
        """
        Handle mouse click events.
        
        Args:
            x (int): X coordinate of the click
            y (int): Y coordinate of the click
        """
        if x < self.sidebar.width:
            # Click in sidebar area
            mode_index = self.sidebar.handle_click(y)
            if mode_index is not None:
                self._update_mode(mode_index)
        elif self.current_mode.value == 3 and self.settings_view:  # Settings mode
            # Pass click to settings view
            self._handle_settings_click(x, y)
            
    def _update_mode(self, mode_index):
        """
        Update the current view mode.
        
        Args:
            mode_index (int): New mode index
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
    
    def _handle_settings_click(self, x, y):
        """
        Handle clicks in the settings view.
        
        Args:
            x (int): X coordinate of the click
            y (int): Y coordinate of the click
        """
        if not self.settings_view:
            return
            
        # Check if a settings button was clicked
        for button_rect, setting_key in self.settings_view.settings_buttons:
            if button_rect.collidepoint(x, y):
                self._toggle_setting(setting_key)
                break
    
    def _toggle_setting(self, setting_key):
        """
        Toggle a setting or trigger an action.
        
        Args:
            setting_key (str): The key of the setting to toggle
        
        Returns:
            bool: True if an action was triggered, False otherwise
        """
        # Special handling for action buttons
        if setting_key == 'reset_plots' or setting_key == 'reset_registry':
            return True
        
        # Toggle regular settings
        if setting_key in self.settings_view.settings:
            self.settings_view.settings[setting_key] = not self.settings_view.settings[setting_key]
        
        return False
    
    def get_current_mode(self):
        """
        Get the current view mode.
        
        Returns:
            ViewMode: The current view mode
        """
        return self.current_mode
