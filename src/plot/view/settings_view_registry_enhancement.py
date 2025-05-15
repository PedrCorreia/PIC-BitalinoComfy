#!/usr/bin/env python
"""
Settings View Enhancement for Registry Integration

This module enhances the SettingsView class with additional registry information.
"""

import pygame
from src.plot.view.settings_view import SettingsView

# Store the original method
original_draw_action_buttons = SettingsView._draw_action_buttons

def enhanced_draw_action_buttons(self, x, y, width):
    """
    Enhanced version of the action buttons method that adds registry connection info.
    """
    # Call the original method first
    original_draw_action_buttons(self, x, y, width)
    
    # Add registry information section
    header_y = y + 150  # Position after the existing Connected nodes info
    header = "Registry Connection"
    header_surface = self.font.render(header, True, (220, 220, 220))
    self.surface.blit(header_surface, (x, header_y))
    
    # Signal count information
    signals_count = self.settings.get('connected_nodes', 0)
    signal_color = (0, 220, 0) if signals_count > 0 else (220, 220, 220)
    signals_text = f"Signals in registry: {signals_count}"
    signals_surface = self.font.render(signals_text, True, signal_color)
    self.surface.blit(signals_surface, (x + 10, header_y + 30))
    
    # Connection status
    registry_connected = self.settings.get('registry_connected', False)
    status_text = "Status: Connected" if registry_connected else "Status: Monitoring"
    status_color = (0, 220, 0) if registry_connected else (220, 180, 0)
    status_surface = self.font.render(status_text, True, status_color)
    self.surface.blit(status_surface, (x + 10, header_y + 60))
    
    # Additional help text
    help_text = "The blinking dot shows data flow activity"
    help_surface = self.font.render(help_text, True, (180, 180, 180))
    self.surface.blit(help_surface, (x + 10, header_y + 90))

# Apply the enhancement
try:
    # Replace the method with our enhanced version
    SettingsView._draw_action_buttons = enhanced_draw_action_buttons
    print("Successfully enhanced SettingsView with registry information")
except Exception as e:
    print(f"Warning: Could not enhance SettingsView for registry information: {e}")
