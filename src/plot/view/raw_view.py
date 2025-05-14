"""
RawView - View implementation for displaying raw signals.

This module provides the visualization of raw signals in the PlotUnit system,
optimized for displaying unprocessed data from various sources.
"""

import pygame
import numpy as np
from .base_view import BaseView
from ..constants import *

class RawView(BaseView):
    """
    View for displaying raw signals.
    
    This view is specialized for visualizing raw, unprocessed signals
    from various sources with appropriate labeling and color-coding.
    """
    
    def draw(self):
        """
        Draw the raw signals visualization.
        
        This method handles the rendering of raw signals on the surface,
        displaying up to 3 signals simultaneously with appropriate labeling.
        """
        # Get available raw signals
        raw_signals = self.get_available_signals(filter_processed=True)
        
        # Fall back to default 'raw' signal if no specific signals found
        if not raw_signals and 'raw' in self.data:
            raw_signals = ['raw']
            
        # Handle case with no signals to display
        if not raw_signals:
            self._draw_no_signals_message()
            return
              # Draw signals with proper spacing
        signal_count = min(len(raw_signals), 3)
        panel_height = (self.plot_height - ((signal_count + 1) * PLOT_PADDING)) // signal_count
        
        for i, signal_id in enumerate(raw_signals[:3]):  # Limit to 3 signals
            with self.data_lock:
                if signal_id in self.data:
                    data = np.copy(self.data[signal_id])
                    
                    # Choose appropriate color and title based on signal type
                    if 'ecg' in signal_id.lower():
                        color = ECG_SIGNAL_COLOR
                        title = f"ECG Raw: {signal_id}"
                    elif 'eda' in signal_id.lower():
                        color = EDA_SIGNAL_COLOR
                        title = f"EDA Raw: {signal_id}"
                    else:
                        color = RAW_SIGNAL_COLOR
                        title = f"Raw Signal: {signal_id}"
                    
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
        message = "No raw signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(
            self.sidebar_width + self.plot_width // 2,
            self.status_bar_height + self.plot_height // 2
        ))
        self.surface.blit(text_surface, text_rect)
