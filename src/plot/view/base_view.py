"""
BaseView - Abstract base class for all visualization views.

This module defines the base class that all view implementations must inherit from,
ensuring a consistent interface across different visualization modes.
"""

from abc import ABC, abstractmethod
import pygame
import numpy as np
from ..constants import *
from ..ui.plot_container import PlotContainer  # Import PlotContainer to access grid drawing

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
        
        # Calculate plot area dimensions
        width = self.surface.get_width() if self.surface else 0
        height = self.surface.get_height() if self.surface else 0
        self.plot_width = width - self.sidebar_width
        self.plot_height = height - self.status_bar_height
        
        # Initialize default plot rectangle (area excluding sidebar and status bar)
        self.plot_rect = pygame.Rect(
            self.sidebar_width, self.status_bar_height,
            self.plot_width, self.plot_height
        )
        
        # Twin view rectangles (left and right sides)
        self.plot_rect_left = None
        self.plot_rect_right = None
    
    def set_rect(self, rect):
        """
        Set the rectangle for this view.
        
        Args:
            rect (pygame.Rect): The rectangle defining the view's drawing area
        """
        self.plot_rect = rect
        # Update dimensions based on the new rectangle
        self.plot_width = rect.width
        self.plot_height = rect.height
        # Reset twin view rectangles
        self.plot_rect_left = None
        self.plot_rect_right = None
    
    def set_rects(self, rects):
        """
        Set left and right rectangles for twin view mode.
        
        Args:
            rects (list): List containing [left_rect, right_rect] for twin view
        """
        if isinstance(rects, list) and len(rects) >= 2:
            self.plot_rect_left = rects[0]
            self.plot_rect_right = rects[1]
        else:
            # If not properly specified, use the first rect or create default rects
            rect = rects[0] if isinstance(rects, list) and len(rects) > 0 else self.plot_rect
            self.set_rect(rect)
    
    @abstractmethod
    def draw(self):
        """
        Draw the view on the surface.
        
        This method must be implemented by all subclasses to render
        their specific visualization content.
        """
        pass
    
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
        
        # Calculate title height with padding
        title_height = TITLE_PADDING + self.font.get_height()
        
        # Draw the background grid with proper title spacing - ensure it fills exactly the available space
        available_height = height - title_height
        grid_height = (available_height // 50) * 50  # Make sure it's a multiple of grid size (50)
        if grid_height < available_height:
            grid_height = available_height  # If too small, use full available height
            
        available_width = width
        grid_width = (available_width // 50) * 50  # Make sure it's a multiple of grid size (50)
        if grid_width < available_width:
            grid_width = available_width  # If too small, use full available width
            
        # Get colors based on light/dark mode
        bg_color = BACKGROUND_COLOR
        grid_color = GRID_COLOR
        
        # Check if we're in light mode
        if hasattr(self, 'data') and hasattr(self,'data_lock'):
            with self.data_lock:
                if hasattr(self, 'settings') and self.settings.get('light_mode', False):
                    bg_color = (240, 240, 240)  # Light background
                    grid_color = (190, 190, 190)  # Light grid
                elif hasattr(self, 'data') and isinstance(self.data, dict) and 'settings' in self.data:
                    # Try to get settings from data if available
                    if self.data.get('settings', {}).get('light_mode', False):
                        bg_color = (240, 240, 240)  # Light background
                        grid_color = (190, 190, 190)  # Light grid
        
        # Use the container's grid drawing method
        container = PlotContainer(pygame.Rect(0, 0, 0, 0), 0)  # Temporary container
        container.draw_grid(self.surface, x, y + title_height, grid_width, grid_height, bg_color, grid_color)
        
        # Draw the title with proper margin
        title_color = TEXT_COLOR
        if hasattr(self, 'surface') and hasattr(self.surface, 'get_at'):
            bg_color = self.surface.get_at((x, y))
            # If bg_color is light (in light mode), use dark text
            if sum(bg_color[:3]) > 600:  # Simple brightness threshold
                title_color = (20, 20, 20)  # Dark text for light background
                
        title_surface = self.font.render(title, True, title_color)
        self.surface.blit(title_surface, (x + TEXT_MARGIN, y + TEXT_MARGIN))
          
        # Draw signal if we have data
        if data is not None and len(data) > 1:
            # Calculate margins and spacing
            title_height = TITLE_PADDING + self.font.get_height()
            margin_top = ELEMENT_PADDING
            margin_bottom = ELEMENT_PADDING
            margin_sides = ELEMENT_PADDING
            
            # Scale the data to fit in the panel with proper margins
            plot_height = grid_height - margin_top - margin_bottom
            plot_width = grid_width - (2 * margin_sides)
            
            data_min = np.min(data)
            data_max = np.max(data)
            
            if data_max == data_min:  # Prevent division by zero
                data_max = data_min + 1
                
            scale = plot_height / (data_max - data_min)
            offset = y + title_height + margin_top + (plot_height / 2)
            
            # Draw the signal line - ensure it fits exactly in the grid
            points = []
            num_points = min(len(data), plot_width)
            step = max(1, len(data) // num_points)
            
            for i in range(0, min(int(plot_width), num_points)):
                idx = min(int(i * step), len(data) - 1)
                val = data[idx]
                point_x = x + margin_sides + i
                # Constrain the point to stay within the grid
                point_y = offset - (val - data_min) * scale * 0.8  # 0.8 to leave space at edges
                point_y = max(y + title_height + margin_top, min(point_y, y + title_height + plot_height - margin_bottom))
                points.append((point_x, point_y))
                
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)  # Increased line width to 2
    
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
                
                    
        return signals
