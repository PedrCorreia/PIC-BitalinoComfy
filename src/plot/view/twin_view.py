"""
TwinView - View implementation for side-by-side visualization of raw and processed signals.

This module provides the visualization of paired raw and processed signals side by side
in the PlotUnit system, enabling direct comparison of signals before and after processing.
"""

import pygame
import numpy as np
from .base_view import BaseView
from ..constants import *

class TwinView(BaseView):
    """
    View for displaying raw and processed signals side by side.
    
    This view is specialized for visualizing paired raw and processed signals,
    enabling direct comparison of signals before and after processing.
    """
    
    def draw(self):
        """
        Draw raw and processed signals side by side.
        
        This method handles the rendering of paired signals on the surface,
        displaying up to 3 pairs of signals simultaneously with appropriate labeling.
        """
        # Find signal pairs (raw and processed versions)
        raw_signals = []
        processed_signals = []
        
        with self.data_lock:
            # Find signal pairs
            for signal_id in self.data:
                if signal_id not in ['filtered', 'processed', 'raw'] and '_processed' not in signal_id:
                    # Check if we have a processed version
                    processed_id = f"{signal_id}_processed"
                    if processed_id in self.data:
                        raw_signals.append(signal_id)
                        processed_signals.append(processed_id)
                
                if len(raw_signals) >= 3:  # Limit to 3 signal pairs
                    break
        
        # Fall back to default signals if no paired signals found
        if not raw_signals:
            with self.data_lock:
                if 'raw' in self.data and ('filtered' in self.data or 'processed' in self.data):
                    raw_signals = ['raw']
                    processed_signals = ['filtered' if 'filtered' in self.data else 'processed']
        
        # Handle case with no signals to display
        if not raw_signals:
            self._draw_no_signals_message()
            return
              # Calculate layout with improved spacing
        available_width = self.plot_width - PLOT_PADDING * 3  # Account for padding between and on sides
        half_width = available_width // 2
        
        # Determine height based on number of signals
        signal_count = len(raw_signals)
        pair_height = (self.plot_height - ((signal_count + 1) * PLOT_PADDING)) // signal_count
        
        # Draw each signal pair
        for i, (raw_id, processed_id) in enumerate(zip(raw_signals, processed_signals)):
            with self.data_lock:
                raw_data = np.copy(self.data[raw_id]) if raw_id in self.data else np.zeros(100)
                processed_data = np.copy(self.data[processed_id]) if processed_id in self.data else np.zeros(100)
                
                # Choose appropriate colors and titles
                if 'ecg' in raw_id.lower():
                    raw_color = ECG_SIGNAL_COLOR
                    processed_color = ECG_SIGNAL_COLOR
                elif 'eda' in raw_id.lower():
                    raw_color = EDA_SIGNAL_COLOR
                    processed_color = EDA_SIGNAL_COLOR
                else:
                    raw_color = RAW_SIGNAL_COLOR
                    processed_color = PROCESSED_SIGNAL_COLOR
                  # Calculate position with proper spacing
                y_position = self.status_bar_height + PLOT_PADDING + (i * (pair_height + PLOT_PADDING))
                
                # Draw raw signal on left
                self._draw_signal_panel(
                    raw_data, f"Raw: {raw_id}", raw_color,
                    self.sidebar_width + PLOT_PADDING, y_position,
                    half_width - TWIN_VIEW_SEPARATOR, pair_height
                )
                
                # Draw processed signal on right with proper spacing
                self._draw_signal_panel(
                    processed_data, f"Processed: {processed_id}", processed_color,
                    self.sidebar_width + half_width + (PLOT_PADDING * 2), y_position, 
                    half_width - TWIN_VIEW_SEPARATOR, pair_height
                )
    
    def _draw_no_signals_message(self):
        """
        Draw a message indicating there are no signal pairs to display.
        """
        message = "No paired signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(
            self.sidebar_width + self.plot_width // 2,
            self.status_bar_height + self.plot_height // 2
        ))
        self.surface.blit(text_surface, text_rect)
