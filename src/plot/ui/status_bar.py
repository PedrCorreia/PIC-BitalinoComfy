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
    
    def __init__(self, surface, width, height, font, start_time):
        """
        Initialize the status bar.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            width (int): Width of the status bar
            height (int): Height of the status bar
            font (pygame.font.Font): Font for text rendering
            start_time (float): Timestamp when the visualization started
        """
        self.surface = surface
        self.width = width
        self.height = height
        self.font = font
        self.start_time = start_time
        
    def draw(self, fps, latency, signal_times=None):
        """
        Draw the status bar with performance metrics.
        
        Args:
            fps (float): Current frames per second
            latency (float): Current signal latency in seconds
            signal_times (dict, optional): Dictionary of signal timestamps
        """
        # Draw status bar background
        status_bar_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, status_bar_rect)
        
        # Draw runtime
        runtime = time.time() - self.start_time
        runtime_text = f"Runtime: {int(runtime // 60):02d}:{int(runtime % 60):02d}"
        runtime_surface = self.font.render(runtime_text, True, TEXT_COLOR)
        self.surface.blit(runtime_surface, (10, (self.height - runtime_surface.get_height()) // 2))
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        fps_surface = self.font.render(fps_text, True, TEXT_COLOR)
        self.surface.blit(fps_surface, (170, (self.height - fps_surface.get_height()) // 2))
        
        # Draw latency with color-coding
        if latency < LOW_LATENCY_THRESHOLD:
            latency_color = OK_COLOR
        elif latency < HIGH_LATENCY_THRESHOLD:
            latency_color = WARNING_COLOR
        else:
            latency_color = ERROR_COLOR
        
        latency_ms = latency * 1000  # Convert to milliseconds
        latency_text = f"Latency: {latency_ms:.1f} ms"
        latency_surface = self.font.render(latency_text, True, latency_color)
        self.surface.blit(latency_surface, (300, (self.height - latency_surface.get_height()) // 2))
        
        # Draw signal info if available
        if signal_times and len(signal_times) > 0:
            # Count how many signals are being tracked
            signal_count_text = f"Signals: {len(signal_times)}"
            signal_count_surface = self.font.render(signal_count_text, True, TEXT_COLOR)
            self.surface.blit(signal_count_surface, (480, (self.height - signal_count_surface.get_height()) // 2))
            
            # Show time since last signal update
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
            self.surface.blit(update_surface, (600, (self.height - update_surface.get_height()) // 2))
