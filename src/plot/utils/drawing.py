"""
Drawing utilities for the PlotUnit visualization system.

This module provides drawing utilities for the PlotUnit system,
simplifying common drawing operations with PyGame.
"""

import pygame
import numpy as np

def draw_grid(surface, x, y, width, height, background_color, grid_color, grid_spacing=50):
    """
    Draw a grid background for signal plots.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        x (int): X coordinate of the grid top-left corner
        y (int): Y coordinate of the grid top-left corner
        width (int): Width of the grid
        height (int): Height of the grid
        background_color (tuple): RGB color tuple for the background
        grid_color (tuple): RGB color tuple for the grid lines
        grid_spacing (int, optional): Spacing between grid lines
    """
    # Draw background
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, background_color, rect)
    
    # Draw grid lines
    for i in range(0, width, grid_spacing):
        line_x = x + i
        pygame.draw.line(surface, grid_color, (line_x, y), (line_x, y + height))
    
    for i in range(0, height, grid_spacing):
        line_y = y + i
        pygame.draw.line(surface, grid_color, (x, line_y), (x + width, line_y))

def draw_signal(surface, data, x, y, width, height, color, line_width=1, smart_downsample=True):
    """
    Draw a signal line on a surface.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        data (numpy.ndarray): Signal data to draw
        x (int): X coordinate of the plot area
        y (int): Y coordinate of the plot area
        width (int): Width of the plot area
        height (int): Height of the plot area
        color (tuple): RGB color tuple for the signal
        line_width (int, optional): Width of the signal line
        smart_downsample (bool, optional): Whether to use smart downsampling
    """
    if data is None or len(data) < 2:
        return
        
    # Scale the data to fit in the plot area
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Prevent division by zero
    if data_max == data_min:
        data_max = data_min + 1
        
    scale = height / (data_max - data_min)
    offset = y + height / 2
    
    # Draw the signal line
    points = []
    num_points = min(len(data), width)
    
    # Use appropriate sampling based on data size
    if smart_downsample and len(data) > width * 2:
        # Smart downsampling for large data
        samples_per_pixel = len(data) // width
        
        for i in range(width):
            start_idx = i * samples_per_pixel
            end_idx = min(start_idx + samples_per_pixel, len(data))
            
            if start_idx >= len(data):
                break
                
            # Use min/max for better visual representation
            segment = data[start_idx:end_idx]
            if len(segment) > 0:
                # For each pixel, plot the min and max values
                min_val = np.min(segment)
                max_val = np.max(segment)
                
                min_y = offset - (min_val - data_min) * scale * 0.8  # 0.8 to leave margins
                max_y = offset - (max_val - data_min) * scale * 0.8
                
                # Add both min and max points
                points.append((x + i, min_y))
                points.append((x + i, max_y))
    else:
        # Simple downsampling for smaller data
        step = max(1, len(data) // num_points)
        
        for i in range(num_points):
            idx = min(int(i * step), len(data) - 1)
            val = data[idx]
            point_x = x + i
            point_y = offset - (val - data_min) * scale * 0.8  # 0.8 to leave margins
            points.append((point_x, point_y))
    
    # Draw the line segments
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, line_width)

def draw_text(surface, text, position, font, color, background=None, align="left"):
    """
    Draw text on a surface with alignment options.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        text (str): Text to draw
        position (tuple): (x, y) coordinates for text placement
        font (pygame.font.Font): Font to use for rendering
        color (tuple): RGB color tuple for the text
        background (tuple, optional): RGB color tuple for text background
        align (str, optional): Text alignment ("left", "center", "right")
    """
    text_surface = font.render(text, True, color, background)
    text_rect = text_surface.get_rect()
    
    # Set position based on alignment
    if align == "center":
        text_rect.center = position
    elif align == "right":
        text_rect.right = position[0]
        text_rect.top = position[1]
    else:  # left alignment
        text_rect.topleft = position
        
    surface.blit(text_surface, text_rect)
    
    return text_rect
