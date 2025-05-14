"""
BaseView - Abstract base class for all visualization views.

This module defines the base class that all view implementations must inherit from,
ensuring a consistent interface across different visualization modes.
"""

from abc import ABC, abstractmethod
import pygame
import numpy as np
from ..constants import *

class BaseView(ABC):
    """
    Abstract base class for PlotUnit views.
    
    All visualization views must inherit from this class and implement
    the required methods to ensure consistent behavior.
    
    Attributes:
        surface (pygame.Surface): The surface to draw on
        data_lock (threading.Lock): Lock for thread-safe data access
        data (dict): Dictionary containing signal data
        font (pygame.font.Font): Font for text rendering
    """
    
    def __init__(self, surface, data_lock, data, font):
        """
        Initialize the base view.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            data_lock (threading.Lock): Lock for thread-safe data access
            data (dict): Dictionary containing signal data
            font (pygame.font.Font): Font for text rendering
        """
        self.surface = surface
        self.data_lock = data_lock
        self.data = data
        self.font = font
        self.sidebar_width = SIDEBAR_WIDTH
        self.status_bar_height = STATUS_BAR_HEIGHT
        self.plot_width = self.surface.get_width() - self.sidebar_width
        self.plot_height = self.surface.get_height() - self.status_bar_height
    
    @abstractmethod
    def draw(self):
        """
        Draw the view on the surface.
        
        This method must be implemented by all subclasses to render
        their specific visualization content.
        """
        pass
    
    def _draw_grid(self, x, y, width, height):
        """
        Draw a grid background for signal plots.
        
        Args:
            x (int): X coordinate of the grid top-left corner
            y (int): Y coordinate of the grid top-left corner
            width (int): Width of the grid
            height (int): Height of the grid
        """
        # Draw background
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.surface, BACKGROUND_COLOR, rect)
        
        # Draw grid lines
        for i in range(0, width, 50):
            line_x = x + i
            pygame.draw.line(self.surface, GRID_COLOR, (line_x, y), (line_x, y + height))
        
        for i in range(0, height, 50):
            line_y = y + i
            pygame.draw.line(self.surface, GRID_COLOR, (x, line_y), (x + width, line_y))
    
    def _draw_signal_panel(self, data, title, color, x, y, width, height=None):
        """
        Draw a signal in a panel with title and grid.
        
        Args:
            data (numpy.ndarray): Signal data to plot
            title (str): Title to display above the signal
            color (tuple): RGB color tuple for the signal
            x (int): X coordinate of the panel top-left corner
            y (int): Y coordinate of the panel top-left corner
            width (int): Width of the panel
            height (int, optional): Height of the panel. If None, uses available height.
        """
        if height is None:
            height = self.plot_height - y
        
        # Draw the background grid
        self._draw_grid(x, y + 20, width, height - 20)  # Leave space for title
        
        # Draw the title
        title_surface = self.font.render(title, True, TEXT_COLOR)
        self.surface.blit(title_surface, (x + 5, y))
        
        # Draw signal if we have data
        if data is not None and len(data) > 1:
            # Scale the data to fit in the panel
            plot_height = height - 40  # Leave space for title and margins
            data_min = np.min(data)
            data_max = np.max(data)
            
            if data_max == data_min:  # Prevent division by zero
                data_max = data_min + 1
                
            scale = plot_height / (data_max - data_min)
            offset = y + 20 + plot_height / 2
            
            # Draw the signal line
            points = []
            num_points = min(len(data), width)
            step = max(1, len(data) // num_points)
            
            for i in range(0, num_points):
                idx = min(int(i * step), len(data) - 1)
                val = data[idx]
                point_x = x + i
                point_y = offset - (val - data_min) * scale * 0.8  # 0.8 to leave margins
                points.append((point_x, point_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 1)
    
    def get_available_signals(self, filter_processed=False, max_signals=3):
        """
        Get available signals from the data dictionary.
        
        Args:
            filter_processed (bool): Whether to filter processed signals
            max_signals (int): Maximum number of signals to return
            
        Returns:
            list: List of signal IDs
        """
        signals = []
        
        with self.data_lock:
            for signal_id in self.data:
                # Skip special keys
                if signal_id in ['filtered', 'processed', 'raw']:
                    continue
                    
                # Filter processed signals if requested
                if filter_processed and '_processed' in signal_id:
                    continue
                
                signals.append(signal_id)
                if len(signals) >= max_signals:
                    break
                    
        return signals
