"""
Tooltip components for the PlotUnit visualization system.

This module provides tooltip components for use in the PlotUnit UI system,
adding hover text information to UI elements.
"""

import pygame
from ..constants import *

class Tooltip:
    """
    Tooltip component for displaying additional information on hover.
    
    This class provides a tooltip that can be displayed when hovering
    over UI elements, providing additional context or information.
    
    Attributes:
        text (str): Text to display in the tooltip
        font (pygame.font.Font): Font for rendering the tooltip text
        padding (int): Padding around the tooltip text
        max_width (int): Maximum width of the tooltip
        background_color (tuple): RGB color tuple for the tooltip background
        text_color (tuple): RGB color tuple for the tooltip text
        visible (bool): Whether the tooltip is currently visible
    """
    
    def __init__(self, text, font=None, padding=5, max_width=300):
        """
        Initialize the tooltip.
        
        Args:
            text (str): Text to display in the tooltip
            font (pygame.font.Font, optional): Font for rendering the tooltip text
            padding (int, optional): Padding around the tooltip text
            max_width (int, optional): Maximum width of the tooltip
        """
        self.text = text
        self.font = font
        self.padding = padding
        self.max_width = max_width
        self.background_color = TOOLTIP_BG_COLOR
        self.text_color = TOOLTIP_TEXT_COLOR
        self.visible = False
    
    def draw(self, surface, position, font=None):
        """
        Draw the tooltip on the surface at the given position.
        
        Args:
            surface (pygame.Surface): Surface to draw on
            position (tuple): (x, y) position to draw the tooltip
            font (pygame.font.Font, optional): Font for rendering the tooltip text
        """
        if not self.visible:
            return
            
        # Use provided font or existing font or fallback
        if font:
            self.font = font
        elif not self.font:
            self.font = pygame.font.SysFont(DEFAULT_FONT, TOOLTIP_FONT_SIZE)
        
        # Split text to fit within max_width
        lines = self._wrap_text()
        
        # Calculate tooltip dimensions
        line_surfaces = [self.font.render(line, True, self.text_color) for line in lines]
        line_heights = [surf.get_height() for surf in line_surfaces]
        tooltip_width = max([surf.get_width() for surf in line_surfaces]) + self.padding * 2
        tooltip_height = sum(line_heights) + self.padding * 2
        
        # Ensure tooltip stays within screen bounds
        x, y = position
        if x + tooltip_width > surface.get_width():
            x = surface.get_width() - tooltip_width
        if y + tooltip_height > surface.get_height():
            y = position[1] - tooltip_height
        
        # Draw tooltip background
        tooltip_rect = pygame.Rect(x, y, tooltip_width, tooltip_height)
        pygame.draw.rect(surface, self.background_color, tooltip_rect, border_radius=5)
        pygame.draw.rect(surface, TOOLTIP_BORDER_COLOR, tooltip_rect, 1, border_radius=5)
        
        # Draw tooltip text
        current_y = y + self.padding
        for line_surface in line_surfaces:
            surface.blit(line_surface, (x + self.padding, current_y))
            current_y += line_surface.get_height()
    
    def _wrap_text(self):
        """
        Wrap text to fit within max_width.
        
        Returns:
            list: List of wrapped text lines
        """
        if not self.font:
            return [self.text]
            
        words = self.text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_width = self.font.size(test_line)[0]
            
            if test_width <= self.max_width:
                current_line.append(word)
            else:
                if not current_line:
                    # If the word itself is too long, just add it
                    lines.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def show(self):
        """
        Show the tooltip.
        """
        self.visible = True
    
    def hide(self):
        """
        Hide the tooltip.
        """
        self.visible = False
