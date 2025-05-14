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
        # Get only raw signals - filter out processed signals
        raw_signals = []
        
        with self.data_lock:
            for signal_id in self.data:
                # Only include raw signals (exclude processed)
                if ('_processed' not in signal_id and 
                    'filtered' != signal_id and 
                    'processed' != signal_id):
                    raw_signals.append(signal_id)
                
                # Limit to 3 signals
                if len(raw_signals) >= 3:
                    break
                    
        # Fall back to default 'raw' signal if no specific signals found
        if not raw_signals and 'raw' in self.data:
            raw_signals = ['raw']
            
        # Handle case with no signals to display
        if not raw_signals:
            self._draw_no_signals_message()
            return
        
        # Use the plot_rect provided by PlotContainer
        rect = self.plot_rect
        
        # Draw signals with equal heights and proper spacing
        signal_count = len(raw_signals)
        total_padding = (signal_count + 1) * PLOT_PADDING
        panel_height = (rect.height - total_padding) // max(1, signal_count)
        
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
                    
                    # Draw the signal panel with proper padding using the rect coordinates
                    y_position = rect.y + PLOT_PADDING + (i * (panel_height + PLOT_PADDING))
                    self._draw_signal_panel(
                        data, title, color, 
                        rect.x + PLOT_PADDING, y_position,
                        rect.width - (PLOT_PADDING * 2), panel_height
                    )
    
    def _draw_no_signals_message(self):
        """
        Draw a message indicating there are no signals to display.
        """
        message = "No raw signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        # Use plot_rect for positioning
        rect = self.plot_rect
        text_rect = text_surface.get_rect(center=(
            rect.x + rect.width // 2,
            rect.y + rect.height // 2
        ))
        self.surface.blit(text_surface, text_rect)
