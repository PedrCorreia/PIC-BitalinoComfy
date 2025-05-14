"""
ProcessedView - View implementation for displaying processed signals.

This module provides the visualization of processed signals in the PlotUnit system,
optimized for displaying data that has been filtered or otherwise processed.
"""

import pygame
import numpy as np
from .base_view import BaseView
from ..constants import *

class ProcessedView(BaseView):
    """
    View for displaying processed signals.
    
    This view is specialized for visualizing processed signals after
    filtering, transformation, or other processing steps, with
    appropriate labeling and color-coding.
    """
    def draw(self):
        """
        Draw the processed signals visualization.
        
        This method handles the rendering of processed signals on the surface,
        displaying up to 3 signals simultaneously with appropriate labeling.
        """
        # Get only processed signals
        processed_signals = []
        
        with self.data_lock:
            # First try to find explicitly processed signals
            for signal_id in self.data:
                if '_processed' in signal_id or '_filtered' in signal_id:
                    processed_signals.append(signal_id)
                elif signal_id in ['filtered', 'processed']:
                    processed_signals.append(signal_id)
            
            # Limit to 3 signals
            processed_signals = processed_signals[:3]
        
        # Handle case with no signals to display
        if not processed_signals:
            self._draw_no_signals_message()
            return
        
        # Use the plot_rect provided by PlotContainer
        rect = self.plot_rect
        
        # Draw signals with equal heights and proper spacing
        signal_count = len(processed_signals)
        total_padding = (signal_count + 1) * PLOT_PADDING
        panel_height = (rect.height - total_padding) // max(1, signal_count)
        
        for i, signal_id in enumerate(processed_signals):
            with self.data_lock:
                if signal_id in self.data:
                    data = np.copy(self.data[signal_id])
                    
                    # Choose title based on signal type
                    title = f"Processed: {signal_id}"
                    
                    # Draw the signal panel with proper padding using the rect coordinates
                    y_position = rect.y + PLOT_PADDING + (i * (panel_height + PLOT_PADDING))
                    self._draw_signal_panel(
                        data, title, PROCESSED_SIGNAL_COLOR,
                        rect.x + PLOT_PADDING, y_position,
                        rect.width - (PLOT_PADDING * 2), panel_height
                    )
    
    def _draw_no_signals_message(self):
        """
        Draw a message indicating there are no signals to display.
        """
        message = "No processed signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        # Use plot_rect for positioning
        rect = self.plot_rect
        text_rect = text_surface.get_rect(center=(
            rect.x + rect.width // 2,
            rect.y + rect.height // 2
        ))
        self.surface.blit(text_surface, text_rect)
