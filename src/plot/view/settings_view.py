"""
SettingsView - View implementation for displaying and managing plot settings.

This module provides the settings interface for the PlotUnit system,
allowing users to adjust visualization parameters and trigger actions.
"""

import pygame
import sys
import os
from .base_view import BaseView
from ..constants import *

class SettingsView(BaseView):
    """
    View for displaying and managing plot settings.
    
    This view provides an interface for users to adjust visualization parameters
    and trigger actions like resetting plots or the registry.
    """
    
    def __init__(self, surface, data_lock, data, font, settings):
        """
        Initialize the settings view.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            data_lock (threading.Lock): Lock for thread-safe data access
            data (dict): Dictionary containing signal data
            font (pygame.font.Font): Font for text rendering
            settings (dict): Dictionary containing current settings
        """
        super().__init__(surface, data_lock, data, font)
        self.settings = settings
        self.settings_buttons = []  # Will store (rect, setting_key) pairs
        
        # Store a reference to the parent PlotUnit instance for action handling
        try:
            # Import here to avoid circular imports
            from src.plot.plot_unit import PlotUnit
            self.plot_unit = PlotUnit.get_instance()
        except ImportError:
            print("[SettingsView] Warning: Could not import PlotUnit")
            self.plot_unit = None
    def draw(self):
        """
        Draw the settings interface.
        
        This method renders the settings UI with toggles and buttons
        for adjusting visualization parameters.
        """
        # Clear the settings buttons list
        self.settings_buttons = []
        
        # Debug message to confirm draw is being called
        print(f"[SettingsView] Drawing settings interface")
        
        # Use plot_rect coordinates for positioning
        rect = self.plot_rect
        
        # Draw settings header with proper padding
        header = "Settings"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (rect.x + SECTION_MARGIN, rect.y + SECTION_MARGIN))
        
        # Calculate positions with proper spacing between sections
        first_section_y = rect.y + SECTION_MARGIN + TITLE_PADDING + self.font.get_height()
        section_height = 120  # Approximate height needed for each settings section
        
        # Draw visualization settings section
        self._draw_visualization_settings(rect.x + SECTION_MARGIN, first_section_y, rect.width - (SECTION_MARGIN * 2))
        
        # Draw appearance settings section
        self._draw_appearance_settings(rect.x + SECTION_MARGIN, first_section_y + section_height, rect.width - (SECTION_MARGIN * 2))
        
        # Draw action buttons section
        self._draw_action_buttons(rect.x + SECTION_MARGIN, first_section_y + section_height * 2, rect.width - (SECTION_MARGIN * 2))
    
    def _draw_visualization_settings(self, x, y, width):
        """
        Draw visualization-related settings.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
            width (int): Width of the panel
        """
        # Draw section header
        header = "Visualization"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        header_height = header_surface.get_height()
        self.surface.blit(header_surface, (x, y))
        
        # Calculate spacing between toggles
        toggle_height = 35
        toggle_spacing = CONTROL_MARGIN
        first_toggle_y = y + header_height + TITLE_PADDING
        
        # Draw FPS cap toggle
        self._draw_toggle(
            x, first_toggle_y,
            "FPS Cap",
            "caps_enabled",
            self.settings['caps_enabled']
        )
        
        # Draw performance mode toggle
        self._draw_toggle(
            x, first_toggle_y + toggle_height + toggle_spacing,
            "Performance Mode",
            "performance_mode",
            self.settings['performance_mode']
        )
    
    def _draw_appearance_settings(self, x, y, width):
        """
        Draw appearance-related settings.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
            width (int): Width of the panel
        """
        # Draw section header
        header = "Appearance"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw light mode toggle
        self._draw_toggle(
            x, y + 40,
            "Light Mode",
            "light_mode",
            self.settings['light_mode']
        )
    
    def _draw_action_buttons(self, x, y, width):
        """
        Draw action buttons for resetting plots and registry.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
            width (int): Width of the panel
        """
        # Draw section header
        header = "Actions"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw reset plots button
        self._draw_button(
            x, y + 40,
            "Reset Plots",
            "reset_plots",
            (180, 60, 60)  # Red button
        )
        
        # Draw reset registry button
        self._draw_button(
            x, y + 90, 
            "Reset Registry",
            "reset_registry",
            (180, 60, 60)  # Red button
        )
        
        # Connected nodes info
        nodes_text = f"Connected nodes: {self.settings['connected_nodes']}"
        text_surface = self.font.render(nodes_text, True, TEXT_COLOR)
        self.surface.blit(text_surface, (x, y + 150))
    def _draw_toggle(self, x, y, label, setting_key, value):
        """
        Draw a toggle switch for a boolean setting.
        
        Args:
            x (int): X coordinate for the toggle
            y (int): Y coordinate for the toggle
            label (str): Label for the toggle
            setting_key (str): The key in the settings dictionary
            value (bool): The current value of the setting
        """
        # Draw label with proper spacing
        label_surface = self.font.render(label, True, TEXT_COLOR)
        label_height = label_surface.get_height()
        self.surface.blit(label_surface, (x + TEXT_MARGIN, y + TEXT_MARGIN))
        
        # Calculate toggle dimensions and position with proper spacing
        switch_width = 54  # Slightly wider for better visibility
        switch_height = 26  # Slightly taller for better visibility
        label_width = 180   # Fixed width allocation for label
        switch_x = x + label_width + CONTROL_MARGIN
        switch_y = y + (label_height - switch_height) // 2  # Center vertically with label
        
        # Draw switch background with rounded corners
        bg_color = ACCENT_COLOR if value else GRID_COLOR
        switch_bg_rect = pygame.Rect(switch_x, switch_y, switch_width, switch_height)
        pygame.draw.rect(self.surface, bg_color, switch_bg_rect, border_radius=13)
        
        # Draw switch handle with proper positioning
        handle_size = 18
        handle_margin = 4
        handle_pos = switch_x + switch_width - handle_size - handle_margin if value else switch_x + handle_margin
        handle_rect = pygame.Rect(handle_pos, switch_y + handle_margin, handle_size, handle_size)
        pygame.draw.rect(self.surface, TEXT_COLOR, handle_rect, border_radius=9)
        
        # Store button rect and settings key for click handling
        self.settings_buttons.append((switch_bg_rect, setting_key))
        
        # Debug output for button positioning
        print(f"[SettingsView] Added toggle: {setting_key}, rect: {switch_bg_rect}, value: {value}")
    def _draw_button(self, x, y, label, action_key, color):
        """
        Draw an action button.
        
        Args:
            x (int): X coordinate for the button
            y (int): Y coordinate for the button
            label (str): Label for the button
            action_key (str): Key in the settings dictionary for this action
            color (tuple): RGB color tuple for the button
        """
        # Draw button
        button_width = 150
        button_height = 30
        button_rect = pygame.Rect(x, y, button_width, button_height)
        
        pygame.draw.rect(self.surface, color, button_rect, border_radius=5)
        pygame.draw.rect(self.surface, TEXT_COLOR, button_rect, 1, border_radius=5)
        
        # Draw label
        label_surface = self.font.render(label, True, TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button_rect.center)
        self.surface.blit(label_surface, label_rect)
        
        # Store button rect and action key for click handling
        self.settings_buttons.append((button_rect, action_key))
        
        # Debug output for button positioning
        print(f"[SettingsView] Added action button: {action_key}, rect: {button_rect}")
