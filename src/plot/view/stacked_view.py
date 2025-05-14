"""
StackedView - View implementation for vertically stacked visualization of signals.

This module provides specialized visualization for stacked signal views with proper
grid sizing and alignment, especially for differential views between raw and processed signals.
"""

import pygame
import numpy as np
from .base_view import BaseView
from ..constants import *

class StackedView(BaseView):
    """
    View for displaying signals in a vertical stack with proper grid alignment.
    
    This view is specialized for visualizing multiple signals in a stacked format,
    with support for differential views and proper grid alignment.
    
    Attributes:
        normalize_stacks (bool): Whether to normalize signal scales across stacks
        grid_enabled (bool): Whether to enable grid lines
        stack_mode (str): Mode for stacking ('free' or 'aligned')
    """
    
    def __init__(self, surface, data_lock, data, font):
        """Initialize the stacked view with default stack configuration."""
        super().__init__(surface, data_lock, data, font)
        self.normalize_stacks = True
        self.grid_enabled = True
        self.stack_mode = 'aligned'  # 'aligned' or 'free'
        self.stack_groups = {}  # Dictionary mapping signal IDs to stack groups
        self.signal_params = {}  # Dictionary for signal visualization parameters
        self.differential_views = {}  # Dictionary for differential view configurations
    
    def set_stack_mode(self, mode):
        """Set the stack mode (aligned or free)."""
        if mode in ['aligned', 'free']:
            self.stack_mode = mode
    
    def set_normalize_stacks(self, normalize):
        """Enable or disable normalization across stacked plots."""
        self.normalize_stacks = normalize
    
    def set_grid_enabled(self, enabled):
        """Enable or disable grid lines."""
        self.grid_enabled = enabled
    
    def set_signal_params(self, signal_id, color=None, label=None, stack_group=None):
        """
        Set visualization parameters for a specific signal.
        
        Args:
            signal_id (str): ID of the signal to configure
            color (tuple): RGB color tuple for the signal
            label (str): Label to display for the signal
            stack_group (str): Group for aligning signals in stacks
        """
        if signal_id not in self.signal_params:
            self.signal_params[signal_id] = {}
        
        if color is not None:
            self.signal_params[signal_id]['color'] = color
        
        if label is not None:
            self.signal_params[signal_id]['label'] = label
        
        if stack_group is not None:
            self.signal_params[signal_id]['stack_group'] = stack_group
            # Also update the stack_groups dictionary for quick lookup
            self.stack_groups[signal_id] = stack_group
    
    def enable_differential_view(self, base_signal_id, compare_signal_id, label=None, stack_position=None):
        """
        Enable a differential view between two signals.
        
        Args:
            base_signal_id (str): ID of the base signal
            compare_signal_id (str): ID of the signal to compare against
            label (str): Label for the differential view
            stack_position (int): Position in the stack (0-based)
        """
        view_id = f"diff_{base_signal_id}_{compare_signal_id}"
        
        self.differential_views[view_id] = {
            'base_signal': base_signal_id,
            'compare_signal': compare_signal_id,
            'label': label or f"Diff: {base_signal_id} vs {compare_signal_id}",
            'stack_position': stack_position
        }
    
    def draw(self):
        """
        Draw stacked signals with proper grid alignment.
        
        This method handles the rendering of signals in a vertical stack
        with proper grid alignment and optional differential views.
        """
        # Identify which signals to display
        signals_to_display = []
        
        with self.data_lock:
            # First add any configured differential views
            for view_id, config in self.differential_views.items():
                base_id = config['base_signal']
                compare_id = config['compare_signal']
                
                if base_id in self.data and compare_id in self.data:
                    signals_to_display.append({
                        'id': view_id,
                        'type': 'differential',
                        'base_id': base_id, 
                        'compare_id': compare_id,
                        'label': config['label'],
                        'position': config.get('stack_position')
                    })
            
            # Then add regular signals
            for signal_id in self.data:
                # Skip signals that are already part of differential views
                if any(signal_id in [d['base_id'], d['compare_id']] for d in signals_to_display if d['type'] == 'differential'):
                    continue
                
                params = self.signal_params.get(signal_id, {})
                signals_to_display.append({
                    'id': signal_id,
                    'type': 'regular',
                    'label': params.get('label', signal_id),
                    'color': params.get('color'),
                    'stack_group': params.get('stack_group')
                })
        
        # If no signals to display
        if not signals_to_display:
            self._draw_no_signals_message()
            return
        
        # Sort signals by stack position if specified
        signals_to_display.sort(key=lambda s: (s.get('position', float('inf')), s['id']))
        
        # Calculate layout
        rect = self.plot_rect
        signal_count = len(signals_to_display)
        stack_height = (rect.height - ((signal_count + 1) * PLOT_PADDING)) // signal_count
        
        # Determine global min/max for aligned mode if needed
        global_min, global_max = 0, 1  # Default range
        if self.stack_mode == 'aligned' and self.normalize_stacks:
            all_values = []
            
            with self.data_lock:
                for signal in signals_to_display:
                    if signal['type'] == 'regular':
                        if signal['id'] in self.data:
                            signal_data = np.copy(self.data[signal['id']])
                            if len(signal_data) > 0:
                                all_values.extend(signal_data)
                    elif signal['type'] == 'differential':
                        base_id = signal['base_id']
                        compare_id = signal['compare_id']
                        if base_id in self.data and compare_id in self.data:
                            base_data = np.copy(self.data[base_id])
                            compare_data = np.copy(self.data[compare_id])
                            min_len = min(len(base_data), len(compare_data))
                            if min_len > 0:
                                diff_data = base_data[:min_len] - compare_data[:min_len]
                                all_values.extend(diff_data)
            
            if all_values:
                global_min = np.min(all_values)
                global_max = np.max(all_values)
                # Ensure we don't have a zero range
                if global_max == global_min:
                    global_max = global_min + 1
        
        # Draw each signal in the stack
        for i, signal in enumerate(signals_to_display):
            y_position = rect.y + PLOT_PADDING + (i * (stack_height + PLOT_PADDING))
            
            # Get appropriate color based on signal type
            color = (180, 180, 180)  # Default color
            if signal['type'] == 'regular':
                signal_id = signal['id']
                if 'color' in signal and signal['color']:
                    color = signal['color']
                elif 'ecg' in signal_id.lower():
                    color = ECG_SIGNAL_COLOR
                elif 'eda' in signal_id.lower():
                    color = EDA_SIGNAL_COLOR
                elif '_processed' in signal_id:
                    color = PROCESSED_SIGNAL_COLOR
                else:
                    color = RAW_SIGNAL_COLOR
            
            if signal['type'] == 'regular':
                with self.data_lock:
                    if signal['id'] in self.data:
                        signal_data = np.copy(self.data[signal['id']])
                        
                        # Draw signal panel with grid alignment if enabled
                        self._draw_aligned_signal_panel(
                            signal_data, signal['label'], color,
                            rect.x + PLOT_PADDING, y_position,
                            rect.width - (2 * PLOT_PADDING), stack_height,
                            global_min, global_max
                        )
                    else:
                        # Draw empty panel with label if signal doesn't exist
                        self._draw_aligned_signal_panel(
                            np.zeros(100), signal['label'], color,
                            rect.x + PLOT_PADDING, y_position,
                            rect.width - (2 * PLOT_PADDING), stack_height,
                            0, 1  # Default range
                        )
            
            elif signal['type'] == 'differential':
                with self.data_lock:
                    base_id = signal['base_id']
                    compare_id = signal['compare_id']
                    
                    if base_id in self.data and compare_id in self.data:
                        base_data = np.copy(self.data[base_id])
                        compare_data = np.copy(self.data[compare_id])
                        
                        # Ensure we're comparing same-length arrays
                        min_len = min(len(base_data), len(compare_data))
                        if min_len > 0:
                            # Calculate difference signal
                            diff_data = base_data[:min_len] - compare_data[:min_len]
                            
                            # Draw the differential signal
                            self._draw_aligned_signal_panel(
                                diff_data, signal['label'], (200, 100, 200),
                                rect.x + PLOT_PADDING, y_position,
                                rect.width - (2 * PLOT_PADDING), stack_height,
                                global_min, global_max
                            )
                        else:
                            # Draw empty panel if one of the signals is empty
                            self._draw_aligned_signal_panel(
                                np.zeros(100), signal['label'], (200, 100, 200),
                                rect.x + PLOT_PADDING, y_position,
                                rect.width - (2 * PLOT_PADDING), stack_height,
                                0, 1  # Default range
                            )
    
    def _draw_aligned_signal_panel(self, data, title, color, x, y, width, height, global_min=None, global_max=None):
        """
        Draw a signal in a panel with title and grid, with optional alignment to global min/max.
        
        Args:
            data (numpy.ndarray): Signal data to plot
            title (str): Title to display above the signal
            color (tuple): RGB color tuple for the signal
            x (int): X coordinate of the panel top-left corner
            y (int): Y coordinate of the panel top-left corner
            width (int): Width of the panel
            height (int): Height of the panel
            global_min (float): Global minimum for aligned scaling
            global_max (float): Global maximum for aligned scaling
        """
        # Calculate title height with padding
        title_height = TITLE_PADDING + self.font.get_height()
        
        # Draw the background grid with proper title spacing if grid is enabled
        if self.grid_enabled:
            self._draw_grid(x, y + title_height, width, height - title_height)
        else:
            # Just draw a background rectangle
            rect = pygame.Rect(x, y + title_height, width, height - title_height)
            pygame.draw.rect(self.surface, BACKGROUND_COLOR, rect)
        
        # Draw the title with proper margin
        title_surface = self.font.render(title, True, TEXT_COLOR)
        self.surface.blit(title_surface, (x + TEXT_MARGIN, y + TEXT_MARGIN))
        
        # Draw signal if we have data
        if data is not None and len(data) > 1:
            # Calculate margins and spacing
            margin_top = ELEMENT_PADDING
            margin_bottom = ELEMENT_PADDING
            margin_sides = ELEMENT_PADDING
            
            # Scale the data to fit in the panel with proper margins
            plot_height = height - (title_height + margin_top + margin_bottom)
            plot_width = width - (2 * margin_sides)
            
            # Determine min/max for scaling based on alignment mode
            if self.stack_mode == 'aligned' and self.normalize_stacks and global_min is not None and global_max is not None:
                data_min = global_min
                data_max = global_max
            else:
                data_min = np.min(data)
                data_max = np.max(data)
            
            if data_max == data_min:  # Prevent division by zero
                data_max = data_min + 1
                
            scale = plot_height / (data_max - data_min)
            offset = y + title_height + margin_top + (plot_height / 2)
            
            # Draw the signal line
            points = []
            num_points = min(len(data), plot_width)
            step = max(1, len(data) // num_points)
            
            for i in range(0, num_points):
                idx = min(int(i * step), len(data) - 1)
                val = data[idx]
                point_x = x + margin_sides + i
                point_y = offset - (val - data_min) * scale * 0.8  # 0.8 to leave space at edges
                points.append((point_x, point_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)  # Increased line width for better visibility
    
    def _draw_no_signals_message(self):
        """Draw a message indicating there are no signals to display."""
        message = "No signals available for stacked view"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        rect = self.plot_rect
        text_rect = text_surface.get_rect(center=(
            rect.x + rect.width // 2,
            rect.y + rect.height // 2
        ))
        self.surface.blit(text_surface, text_rect)
    
    def synchronize_grid_scales(self):
        """Force synchronization of grid scales."""
        # This method is called periodically to ensure grid scales remain aligned
        pass

    def _draw_grid(self, x, y, width, height, bg_color=None, grid_color=None):
        """
        Draw a grid background for signal panels in StackedView.
        Uses the same logic as PlotContainer.draw_grid for consistency.
        """
        from ..utils.drawing import draw_grid
        # Use default colors if not provided
        bg_color = bg_color if bg_color is not None else BACKGROUND_COLOR
        grid_color = grid_color if grid_color is not None else GRID_COLOR
        draw_grid(self.surface, x, y, width, height, bg_color, grid_color, grid_spacing=50)
