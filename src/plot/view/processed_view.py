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
        # Get available processed signals
        processed_signals = []
        
        with self.data_lock:
            # First try to find explicitly processed signals
            for signal_id in self.data:
                if '_processed' in signal_id or '_filtered' in signal_id:
                    processed_signals.append(signal_id)
            
            # If no processed signals found, check for the generic 'filtered' signal
            if not processed_signals and 'filtered' in self.data:
                processed_signals = ['filtered']
            
            # Limit to 3 signals
            processed_signals = processed_signals[:3]
        
        # Handle case with no signals to display
        if not processed_signals:
            self._draw_no_signals_message()
            return
              # Draw signals with proper spacing
        signal_count = min(len(processed_signals), 3)
        panel_height = (self.plot_height - ((signal_count + 1) * PLOT_PADDING)) // signal_count
        
        for i, signal_id in enumerate(processed_signals):
            with self.data_lock:
                if signal_id in self.data:
                    data = np.copy(self.data[signal_id])
                    
                    # Choose appropriate color and title based on signal type
                    if 'ecg' in signal_id.lower():
                        color = ECG_SIGNAL_COLOR
                        title = f"ECG Processed: {signal_id}"
                    elif 'eda' in signal_id.lower():
                        color = EDA_SIGNAL_COLOR
                        title = f"EDA Processed: {signal_id}"
                    else:
                        color = PROCESSED_SIGNAL_COLOR
                        title = f"Processed: {signal_id}"
                    
                    # Draw the signal panel with proper padding
                    y_position = self.status_bar_height + PLOT_PADDING + (i * (panel_height + PLOT_PADDING))
                    self._draw_signal_panel(
                        data, title, color, 
                        self.sidebar_width + PLOT_PADDING, y_position,
                        self.plot_width - (PLOT_PADDING * 2), panel_height
                    )
    
    def _draw_no_signals_message(self):
        """
        Draw a message indicating there are no signals to display.
        """
        message = "No processed signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(
            self.sidebar_width + self.plot_width // 2,
            self.status_bar_height + self.plot_height // 2
        ))
        self.surface.blit(text_surface, text_rect)
