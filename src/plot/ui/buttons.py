"""
Button components for the PlotUnit visualization system.

This module provides button components for use in the PlotUnit UI system,
including hover effects and standardized styling.
"""

import pygame
from ..constants import *

class Button:
    """
    Base button class for the PlotUnit UI system.
    
    This class provides common functionality for buttons, including
    hover effects, rendering, and click handling.
    
    Attributes:
        rect (pygame.Rect): Rectangle defining the button area
        text (str): Text to display on the button
        color (tuple): RGB color tuple for the button
        hover_color (tuple): RGB color tuple for the button when hovered
        action (callable): Function to call when the button is clicked
        is_hovered (bool): Flag indicating if the button is being hovered
        tooltip (Tooltip): Optional tooltip to display when button is hovered
    """
    
    def __init__(self, x, y, width, height, text, color, action=None, tooltip_text=None):
        """
        Initialize the button.
        
        Args:
            x (int): X coordinate of the button top-left corner
            y (int): Y coordinate of the button top-left corner
            width (int): Width of the button
            height (int): Height of the button
            text (str): Text to display on the button
            color (tuple): RGB color tuple for the button
            action (callable, optional): Function to call when the button is clicked
            tooltip_text (str, optional): Text to display in the tooltip when hovered
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = self._calculate_hover_color(color)
        self.active_color = color
        self.action = action
        self.is_hovered = False
        self.tooltip = None
        
        # Create tooltip if tooltip_text is provided
        if tooltip_text:
            from .tooltip import Tooltip
            self.tooltip = Tooltip(tooltip_text)
    def _calculate_hover_color(self, color):
        """
        Calculate a lighter color for hover effect.
        
        Args:
            color (tuple): RGB color tuple
            
        Returns:
            tuple: Lighter RGB color tuple
        """
        return (
            min(color[0] + 30, 255),
            min(color[1] + 30, 255),
            min(color[2] + 30, 255)
        )
        
    def draw(self, surface, font=None):
        """
        Draw the button on the surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
            font (pygame.font.Font, optional): Font to use for text rendering
        """
        # Choose color based on hover state
        current_color = self.hover_color if self.is_hovered else self.active_color
        
        # Draw button rectangle with rounded corners
        pygame.draw.rect(surface, current_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 1, border_radius=5)  # Button border
        
        # Draw text
        if font is None:
            font = pygame.font.SysFont(DEFAULT_FONT, DEFAULT_FONT_SIZE)
            
        text_surface = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)       
        surface.blit(text_surface, text_rect)
        
        # Draw tooltip if hovered and tooltip exists
        if self.is_hovered and self.tooltip:
            mouse_pos = pygame.mouse.get_pos()
            # Position tooltip slightly below and to the right of mouse
            tooltip_pos = (mouse_pos[0] + 15, mouse_pos[1] + 15)
            self.tooltip.show()
            self.tooltip.draw(surface, tooltip_pos, font)
    
    def check_hover(self, mouse_pos):
        """
        Check if the mouse is hovering over the button.
        
        Args:
            mouse_pos (tuple): (x, y) position of the mouse
            
        Returns:
            bool: True if the button is being hovered, False otherwise
        """
        was_hovered = self.is_hovered
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        # Show or hide tooltip based on hover state
        if self.tooltip:
            if self.is_hovered:
                self.tooltip.show()
            else:
                self.tooltip.hide()
                
        return was_hovered != self.is_hovered  # Return True if hover state changed
    
    def handle_click(self):
        """
        Handle a click on the button.
        
        Returns:
            bool: True if the button has an action and it was called, False otherwise
        """
        if self.action:
            self.action()
            return True
        return False


class ResetButton(Button):
    """
    Special button for reset operations.
    
    This button is styled specifically for reset operations, with red coloring
    to indicate destructive actions.
    """
    
    def __init__(self, x, y, width, height, text, action=None):
        """
        Initialize the reset button.
        
        Args:
            x (int): X coordinate of the button top-left corner
            y (int): Y coordinate of the button top-left corner
            width (int): Width of the button
            height (int): Height of the button
            text (str): Text to display on the button
            action (callable, optional): Function to call when the button is clicked
        """
        # Use red color for reset buttons
        super().__init__(x, y, width, height, text, (180, 60, 60), action)


class ToggleButton(Button):
    """
    Toggle button with on/off state.
    
    This button toggles between two states, with different colors for each.
    
    Attributes:
        is_on (bool): Current state of the toggle
        on_color (tuple): RGB color tuple for the ON state
        off_color (tuple): RGB color tuple for the OFF state
    """
    
    def __init__(self, x, y, width, height, text, is_on=False, action=None):
        """
        Initialize the toggle button.
        
        Args:
            x (int): X coordinate of the button top-left corner
            y (int): Y coordinate of the button top-left corner
            width (int): Width of the button
            height (int): Height of the button
            text (str): Text to display on the button
            is_on (bool, optional): Initial state of the toggle
            action (callable, optional): Function to call when the button is clicked
        """
        self.is_on = is_on
        self.on_color = ACCENT_COLOR
        self.off_color = GRID_COLOR
        
        # Initialize with appropriate color based on state
        color = self.on_color if is_on else self.off_color
        super().__init__(x, y, width, height, text, color, action)
    
    def toggle(self):
        """
        Toggle the button state.
        
        Returns:
            bool: New state of the toggle
        """       
        self.is_on = not self.is_on
        self.active_color = self.on_color if self.is_on else self.off_color
        self.hover_color = self._calculate_hover_color(self.active_color)
        
        if self.action:
            self.action(self.is_on)
            
        return self.is_on
    
    def handle_click(self):
        """
        Handle a click on the toggle button.
        
        Returns:
            bool: Always True to indicate the click was handled
        """
        self.toggle()
        return True
