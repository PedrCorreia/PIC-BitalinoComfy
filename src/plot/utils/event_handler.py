"""
EventHandler module for the PlotUnit visualization system.

This module manages event processing for the PlotUnit system,
handling user input and system events.
"""

import pygame
import os
import sys
from ..constants import *
from ..view_mode import ViewMode

class EventHandler:
    """
    Event handler for the PlotUnit visualization system.
    
    This class manages event processing, including mouse and keyboard events,
    and updates the application state accordingly.
    
    Attributes:
        sidebar (Sidebar): The sidebar component for navigation
        current_mode (ViewMode): The currently active view mode
        settings_view (SettingsView): The settings view for handling settings clicks
    """
    def __init__(self, sidebar, settings_view=None, button_controller=None):
        """
        Initialize the event handler.
        
        Args:
            sidebar (Sidebar): The sidebar component for navigation
            settings_view (SettingsView, optional): The settings view for handling settings clicks
            button_controller: Kept for backward compatibility but no longer used
        """
        self.sidebar = sidebar
        self.settings_view = settings_view
        # Button controller has been removed and is no longer used
        self.current_mode = sidebar.current_mode
        
    def get_current_mode(self):
        """
        Get the current view mode.
        
        Returns:
            ViewMode: The current view mode
        """
        return self.current_mode
        
    def process_events(self):
        """
        Process pygame events and update application state.
        
        Returns:
            bool: True if the application should continue running, False if it should exit
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
            print("[EVENT] Settings view not available")
            return
        
        print(f"[EVENT] Settings view click at coordinates: ({x}, {y})")
        
        # Check if a settings button was clicked
        found_button = False
        for button_rect, setting_key in self.settings_view.settings_buttons:
            if button_rect.collidepoint(x, y):
                print(f"[EVENT] Button clicked: {setting_key}")
                self._toggle_setting(setting_key)
                found_button = True
                # No break, continue checking all buttons
        
        if not found_button:
            print(f"[EVENT] No button found at click position ({x}, {y})")
            print(f"[EVENT] Available buttons: {len(self.settings_view.settings_buttons)}")
            
    def _toggle_setting(self, setting_key):
        """
        Toggle a setting or trigger an action.
        
        Args:
            setting_key (str): The key of the setting to toggle
        
        Returns:
            bool: True if an action was triggered, False otherwise
        """
        PlotUnit = None
        try:
            from src.plot.plot_unit import PlotUnit
        except ImportError:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from plot.plot_unit import PlotUnit
            except Exception as e:
                print(f"[EVENT] Failed to import PlotUnit: {str(e)}")
                PlotUnit = None

        # Special handling for action buttons
        if setting_key == 'reset_plots':
            if PlotUnit:
                plot_unit = PlotUnit.get_instance()
                if hasattr(plot_unit, 'clear_plots'):
                    plot_unit.clear_plots()
                    print("[EVENT] Reset plots action triggered")
                    return True
            print("[EVENT] Could not reset plots (PlotUnit unavailable)")
            return True

        elif setting_key == 'reset_registry':
            try:
                from src.registry.plot_registry import PlotRegistry
                PlotRegistry.get_instance().reset()
                print("[EVENT] Reset registry action triggered (PlotRegistry)")
            except Exception as e:
                print(f"[EVENT] Failed to reset registry using PlotRegistry: {str(e)}")
            return True

        # Toggle regular settings
        if self.settings_view and setting_key in self.settings_view.settings:
            self.settings_view.settings[setting_key] = not self.settings_view.settings[setting_key]
            print(f"[EVENT] Setting '{setting_key}' toggled to {self.settings_view.settings[setting_key]}")
        return False
