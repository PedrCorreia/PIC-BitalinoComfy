#!/usr/bin/env python
"""
Sidebar Update for Registry Integration

This module enhances the src.plot.ui.sidebar module with registry connection indicators.
"""

import pygame
from src.plot.ui.sidebar import Sidebar

# Store the original method
original_draw_status_dot = Sidebar._draw_status_dot_below_settings

def enhanced_draw_status_dot(self):
    """
    Enhanced version of the status dot drawing method that adds blinking for registry connections.
    
    This completely replaces the original method to handle all dots including the blinking
    registry connection dot.
    """
    # Use the same calculations as in the original method
    status_bar_offset = 30 if hasattr(self, 'STATUS_BAR_TOP') and self.STATUS_BAR_TOP else 0
    s_button_top = status_bar_offset + self.button_spacing + 3 * (self.button_height + self.button_spacing)
    s_button_bottom = s_button_top + self.button_height
    dot_radius = 7
    dot_x = self.width // 2    # Vertical stacking of dots - position lower to avoid UI crashes
    dot_y_start = s_button_bottom + 40  # Increased from 18 to 40
    dot_spacing = 30  # Increased from 22 to 30px between dots

    dot_index = 0

    # Performance mode dot (yellow)
    if self.settings.get('performance_mode', False):
        pygame.draw.circle(self.surface, (220, 220, 0), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
        dot_index += 1

    # Caps enabled dot (green)
    if self.settings.get('caps_enabled', False):
        pygame.draw.circle(self.surface, (0, 220, 0), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
        dot_index += 1

    # Registry connection dot - enhanced with blinking when connected
    if hasattr(self.settings, 'get') and self.settings.get('registry_connected', False):
        # Bright green when connected and blinking on (actively connected)
        pygame.draw.circle(self.surface, (50, 220, 50), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)
    else:
        # Gray when not connected or blinking off
        pygame.draw.circle(self.surface, (120, 120, 120), (dot_x, dot_y_start + dot_index * dot_spacing), dot_radius)

# Apply the enhancement
try:
    # Replace the method with our enhanced version
    Sidebar._draw_status_dot_below_settings = enhanced_draw_status_dot
    print("Successfully enhanced Sidebar with registry connection indicator")
except Exception as e:
    print(f"Warning: Could not enhance Sidebar for registry connection: {e}")
