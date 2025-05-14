"""
Sidebar module for the PlotUnit visualization system.

This module provides the sidebar navigation component for the PlotUnit system,
allowing users to switch between different view modes.
"""

import pygame
from ..constants import *
from ..view_mode import ViewMode

class Sidebar:
    """
    Sidebar navigation component for PlotUnit.
    
    This class manages the sidebar that allows users to switch between
    different visualization modes.
    
    Attributes:
        surface (pygame.Surface): The surface to draw on
        width (int): Width of the sidebar
        height (int): Height of the sidebar
        font (pygame.font.Font): Font for text rendering
        icon_font (pygame.font.Font): Font for icon rendering
        current_mode (ViewMode): The currently active view mode
    """
    def __init__(self, surface, width, height, font, icon_font, current_mode):
        """
        Initialize the sidebar.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            width (int): Width of the sidebar
            height (int): Height of the sidebar
            font (pygame.font.Font): Font for text rendering
            icon_font (pygame.font.Font): Font for icon rendering
            current_mode (ViewMode): The currently active view mode
        """       
        self.surface = surface
        self.width = width
        self.height = height
        self.font = font
        self.icon_font = icon_font
        self.current_mode = current_mode
        
        # Button settings
        self.button_height = TAB_HEIGHT + 10  # Use TAB_HEIGHT constant plus padding
        self.button_spacing = 15  # Space between buttons
        
    def draw(self):
        """
        Draw the sidebar with navigation buttons.
        """
        # Draw sidebar background
        sidebar_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, sidebar_rect)
        
        # Draw mode buttons with proper spacing
        self._draw_mode_button(0, "R", "Raw", 0)
        self._draw_mode_button(1, "P", "Processed", 1)
        self._draw_mode_button(2, "T", "Twin View", 2)
        self._draw_mode_button(3, "S", "Settings", 3)
        
    def _draw_mode_button(self, position, icon, tooltip, mode_value):
        """
        Draw a mode selection button.
        
        Args:
            position (int): Position index in the sidebar (0-based)
            icon (str): Single character icon to display
            tooltip (str): Tooltip text for the button
            mode_value (int): The ViewMode enum value for this button
        """
        # Calculate position with proper spacing between buttons
        y = self.button_spacing + position * (self.button_height + self.button_spacing)
        
        # Check if this is the current mode
        is_active = self.current_mode.value == mode_value
        
        # Draw button background if active
        if is_active:
            button_rect = pygame.Rect(0, y, self.width, self.button_height)
            pygame.draw.rect(self.surface, ACCENT_COLOR, button_rect)
            
        # Draw button icon
        icon_surface = self.icon_font.render(icon, True, TEXT_COLOR)
        icon_rect = icon_surface.get_rect(center=(self.width // 2, y + self.button_height // 2))
        self.surface.blit(icon_surface, icon_rect)
    
    def handle_click(self, y):
        """
        Handle a click on the sidebar.
        
        Args:
            y (int): Y coordinate of the click
            
        Returns:
            int: The ViewMode enum value for the clicked button, or None if no button was clicked
        """
        # Update click handling to account for button spacing
        for i in range(4):  # We have 4 buttons (0-3)
            button_top = self.button_spacing + i * (self.button_height + self.button_spacing)
            button_bottom = button_top + self.button_height
            
            if button_top <= y <= button_bottom:
                return i
                
        return None
