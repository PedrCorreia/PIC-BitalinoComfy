"""
Sidebar module for the PlotUnit visualization system.

This module provides the sidebar navigation component for the PlotUnit system,
allowing users to switch between different view modes.
"""

import pygame
from ..constants import *
from ..view_mode import ViewMode

# Ensure TAB_ICON_FONT_SIZE is always defined
if 'TAB_ICON_FONT_SIZE' not in globals():
    TAB_ICON_FONT_SIZE = 28  # Fallback default if not imported

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
        settings (dict): Settings dictionary for additional configurations
    """
    def __init__(self, surface, width, height, font, icon_font, current_mode, settings):
        """
        Initialize the sidebar.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            width (int): Width of the sidebar
            height (int): Height of the sidebar
            font (pygame.font.Font): Font for text rendering
            icon_font (pygame.font.Font): Font for icon rendering
            current_mode (ViewMode): The currently active view mode
            settings (dict): Settings dictionary for additional configurations
        """       
        self.surface = surface
        self.width = width
        self.height = height
        self.font = font
        self.icon_font = icon_font
        self.current_mode = current_mode
        self.settings = settings  # Add settings reference
        # Button settings
        self.button_height = TAB_HEIGHT + 10  # Use TAB_HEIGHT constant plus padding
        self.button_spacing = 15  # Space between buttons
        
    def draw(self):
        """
        Draw the sidebar with navigation buttons.
        """
        # Draw sidebar background, accounting for status bar at top
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        sidebar_rect = pygame.Rect(0, status_bar_offset, self.width, self.height - status_bar_offset)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, sidebar_rect)
        
        # Draw mode buttons with proper spacing
        self._draw_mode_button(0, "R", "Raw", ViewMode.RAW)
        self._draw_mode_button(1, "P", "Processed", ViewMode.PROCESSED)
        self._draw_mode_button(2, "T", "Twin View", ViewMode.TWIN)
        self._draw_mode_button(3, "S", "Settings", ViewMode.SETTINGS)
        # Draw status indicator dot below the S button
        self._draw_status_dot_below_settings()

    def _draw_mode_button(self, position, icon, tooltip, mode_value):
        """
        Draw a mode selection button.
        
        Args:
            position (int): Position index in the sidebar (0-based)
            icon (str): Single character icon to display
            tooltip (str): Tooltip text for the button
            mode_value (ViewMode): The ViewMode enum value for this button
        """
        # Calculate position with proper spacing between buttons, accounting for status bar
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        y = status_bar_offset + self.button_spacing + position * (self.button_height + self.button_spacing)
        
        # Check if this is the current mode
        is_active = self.current_mode == mode_value
        
        # Draw button background if active
        if is_active:
            button_rect = pygame.Rect(0, y, self.width, self.button_height)
            pygame.draw.rect(self.surface, ACCENT_COLOR, button_rect)
            
        # Draw button icon with larger font size for better visibility
        larger_icon_font = pygame.font.SysFont(None, TAB_ICON_FONT_SIZE)  # Use larger font size constant
        icon_surface = larger_icon_font.render(icon, True, TEXT_COLOR)
        icon_rect = icon_surface.get_rect(center=(self.width // 2, y + self.button_height // 2))
        self.surface.blit(icon_surface, icon_rect)

    def _draw_status_dot_below_settings(self):
        """
        Draw status indicator dots below the Settings button:
        - Yellow for performance mode on
        - Green for caps on
        - Gray (placeholder) for registry connection (future: blinking)
        """
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        s_button_top = status_bar_offset + self.button_spacing + 3 * (self.button_height + self.button_spacing)
        s_button_bottom = s_button_top + self.button_height
        dot_radius = 7
        dot_x = self.width // 2

        # Vertical stacking of dots
        dot_y_start = s_button_bottom + 18
        dot_spacing = 22  # px between dots

        dot_index = 0

        # Performance mode dot (yellow)
        if self.settings.get('performance_mode', False):
            pygame.draw.circle(self.surface, (220, 220, 0), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
            dot_index += 1

        # Caps enabled dot (green)
        if self.settings.get('caps_enabled', False):
            pygame.draw.circle(self.surface, (0, 220, 0), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
            dot_index += 1

        # Registry connection dot (gray, placeholder for blinking)
        # In the future, replace this with blinking logic if connected
        # Example: if self.settings.get('connected_to_registry', False):
        pygame.draw.circle(self.surface, (120, 120, 120), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
        # To implement blinking, toggle visibility based on time or state

    def handle_click(self, y):
        """
        Handle a click on the sidebar.
        
        Args:
            y (int): Y coordinate of the click
            
        Returns:
            int: The ViewMode enum value for the clicked button, or None if no button was clicked
        """
        # Account for status bar position when processing clicks
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        
        # Update click handling to account for button spacing and status bar
        for i in range(4):  # We have 4 buttons (0-3)
            button_top = status_bar_offset + self.button_spacing + i * (self.button_height + self.button_spacing)
            button_bottom = button_top + self.button_height
            
            if button_top <= y <= button_bottom:
                return i
                
        return None

    def update_dynamic_state(self, current_mode, settings):
        """
        Update the sidebar's dynamic state (current mode and settings) from the main loop.
        
        Args:
            current_mode (ViewMode): The currently active view mode
            settings (dict): The latest settings dictionary
        """
        self.current_mode = current_mode
        self.settings = settings.copy() if isinstance(settings, dict) else settings
