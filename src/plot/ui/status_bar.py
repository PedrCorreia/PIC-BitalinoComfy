"""
StatusBar module for the PlotUnit visualization system.

This module provides the status bar component for the PlotUnit system,
displaying performance metrics and other status information.
"""

import pygame
import time
from ..constants import *

class StatusBar:
    """
    Status bar component for PlotUnit.
    
    This class manages the status bar that displays performance metrics
    and other status information.
    
    Attributes:
        surface (pygame.Surface): The surface to draw on
        width (int): Width of the status bar
        height (int): Height of the status bar
        font (pygame.font.Font): Font for text rendering
        start_time (float): Timestamp when the visualization started
    """
    
    def __init__(self, surface, height, font, start_time=None):
        """
        Initialize the status bar.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            height (int): Height of the status bar
            font (pygame.font.Font): Font for text rendering
            start_time (float): Timestamp when the visualization started
        """
        self.surface = surface
        self.width = self.surface.get_width()
        self.height = height
        self.font = font
        self.start_time = start_time

    def set_start_time(self, start_time):
        """
        Dynamically set or update the start_time attribute.

        Args:
            start_time (float): The timestamp to set as the start time.
        """
        self.start_time = start_time

    def draw(self, fps, latency, signal_times=None):
        """
        Draw the status bar with performance metrics, using sidebar and window dimensions for precise layout.
        """
        left = SIDEBAR_WIDTH + ELEMENT_PADDING
        right = self.width - ELEMENT_PADDING
        y_center = (self.height) // 2
        x = left

        # Draw status bar background
        status_bar_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, status_bar_rect)

        # Runtime (left-aligned)
        if self.start_time is not None:
            runtime = time.time() - self.start_time
            runtime_text = f"Runtime: {int(runtime // 60):02d}:{int(runtime % 60):02d}"
        else:
            runtime_text = "Runtime: N/A"
        runtime_surface = self.font.render(runtime_text, True, TEXT_COLOR)
        self.surface.blit(runtime_surface, (x, y_center - runtime_surface.get_height() // 2))
        x += runtime_surface.get_width() + SECTION_MARGIN

        # NOTE: Latency display removed per user request

        # Signal info (centered if available)
        if signal_times and len(signal_times) > 0:
            signal_count_text = f"Signals: {len(signal_times)}"
            signal_count_surface = self.font.render(signal_count_text, True, TEXT_COLOR)
            self.surface.blit(signal_count_surface, (x, y_center - signal_count_surface.get_height() // 2))
            x += signal_count_surface.get_width() + SECTION_MARGIN

            latest_time = max(signal_times.values()) if signal_times else time.time()
            time_since_update = time.time() - latest_time
            if time_since_update < 1.0:
                update_text = "Last update: Just now"
                update_color = OK_COLOR
            elif time_since_update < 5.0:
                update_text = f"Last update: {time_since_update:.1f}s ago"
                update_color = OK_COLOR
            elif time_since_update < 30.0:
                update_text = f"Last update: {time_since_update:.1f}s ago"
                update_color = WARNING_COLOR
            else:
                update_text = f"Last update: {int(time_since_update)}s ago"
                update_color = ERROR_COLOR
            update_surface = self.font.render(update_text, True, update_color)
            self.surface.blit(update_surface, (x, y_center - update_surface.get_height() // 2))
            x += update_surface.get_width() + SECTION_MARGIN

        # FPS (right-aligned)
        fps_text = f"FPS: {fps:.1f}"
        fps_surface = self.font.render(fps_text, True, TEXT_COLOR)
        fps_x = right - fps_surface.get_width()
        self.surface.blit(fps_surface, (fps_x, y_center - fps_surface.get_height() // 2))
