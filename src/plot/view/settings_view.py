"""
SettingsView - View implementation for displaying and managing plot settings.

This module provides the settings interface for the PlotUnit system,
allowing users to adjust visualization parameters and trigger actions.
"""

import pygame
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
    
    def draw(self):
        """
        Draw the settings interface.
        
        This method renders the settings UI with toggles and buttons
        for adjusting visualization parameters.
        """
        # Clear the settings buttons list
        self.settings_buttons = []
        
        # Draw settings header
        header = "Settings"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (self.sidebar_width + 20, self.status_bar_height + 20))
        
        # Draw settings panels
        self._draw_performance_settings(self.sidebar_width + 20, self.status_bar_height + 60)
        self._draw_appearance_settings(self.sidebar_width + 20, self.status_bar_height + 200)
        self._draw_action_buttons(self.sidebar_width + 20, self.status_bar_height + 340)
        
    def _draw_performance_settings(self, x, y):
        """
        Draw performance-related settings.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
        """
        # Draw section header
        header = "Performance"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw FPS cap toggle
        self._draw_toggle(
            x, y + 40,
            "FPS Cap",
            "caps_enabled",
            self.settings['caps_enabled']
        )
        
        # Draw performance mode toggle
        self._draw_toggle(
            x, y + 80,
            "Performance Mode",
            "performance_mode",
            self.settings['performance_mode']
        )
    
    def _draw_appearance_settings(self, x, y):
        """
        Draw appearance-related settings.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
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
    
    def _draw_action_buttons(self, x, y):
        """
        Draw action buttons for resetting plots and registry.
        
        Args:
            x (int): X coordinate for the panel
            y (int): Y coordinate for the panel
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
        # Draw label
        label_surface = self.font.render(label, True, TEXT_COLOR)
        self.surface.blit(label_surface, (x, y))
        
        # Draw toggle switch
        switch_width = 50
        switch_height = 24
        switch_x = x + 200
        
        # Draw switch background
        bg_color = ACCENT_COLOR if value else GRID_COLOR
        switch_bg_rect = pygame.Rect(switch_x, y, switch_width, switch_height)
        pygame.draw.rect(self.surface, bg_color, switch_bg_rect, border_radius=12)
        
        # Draw switch handle
        handle_pos = switch_x + switch_width - 20 if value else switch_x + 4
        handle_rect = pygame.Rect(handle_pos, y + 4, 16, 16)
        pygame.draw.rect(self.surface, TEXT_COLOR, handle_rect, border_radius=8)
        
        # Store button rect and settings key for click handling
        self.settings_buttons.append((switch_bg_rect, setting_key))
    
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
