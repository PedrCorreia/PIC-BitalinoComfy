"""
SignalView - Example view implementation demonstrating usage of the new signal processing components.

This module provides an example of how to use the new signal processing components
in a view implementation.
"""

import pygame
import numpy as np
from .base_view import BaseView
from ..utils import SignalHistoryManager, SignalProcessingAdapter
from ..constants import *

class SignalView(BaseView):
    """
    Example signal view demonstrating usage of the new signal processing components.
    
    This view shows how to use the SignalHistoryManager and SignalProcessingAdapter
    to process and visualize signals.
    """
    
    def __init__(self, surface, data_lock, data, font):
        """
        Initialize the signal view.
        
        Args:
            surface (pygame.Surface): The surface to draw on
            data_lock (threading.Lock): Lock for thread-safe data access
            data (dict): Dictionary containing signal data
            font (pygame.font.Font): Font for text rendering
        """
        super().__init__(surface, data_lock, data, font)
        self.history_manager = SignalHistoryManager()
        
    def draw(self):
        """
        Draw the signal visualization.
        
        This method demonstrates how to use the SignalHistoryManager and 
        SignalProcessingAdapter for signal visualization.
        """
        # Get available signals
        signals = self.get_available_signals(max_signals=2)
        
        # Handle case with no signals to display
        if not signals:
            self._draw_no_signals_message()
            return
            
        # Process each signal
        for i, signal_id in enumerate(signals):
            with self.data_lock:
                if signal_id in self.data:
                    # Get data and update history
                    data = np.copy(self.data[signal_id])
                    self.history_manager.update_history(signal_id, data)
                    
                    # Get signal history
                    history = self.history_manager.get_history(signal_id)
                    
                    # Process the signal using SignalProcessingAdapter
                    processed = SignalProcessingAdapter.process_signal(
                        history,
                        processing_type='moving_average',
                        window_size=10
                    )
                    
                    # Get signal stats
                    stats = SignalProcessingAdapter.analyze_signal(history, 'stats')
                    
                    # Draw the raw signal
                    panel_height = self.plot_height // (len(signals) * 2)
                    y_position = self.status_bar_height + (i * panel_height * 2)
                    
                    self._draw_signal_panel(
                        history, 
                        f"Raw Signal: {signal_id} (Mean: {stats['mean']:.2f})", 
                        RAW_SIGNAL_COLOR,
                        self.sidebar_width, 
                        y_position,
                        self.plot_width,
                        panel_height
                    )
                    
                    # Draw the processed signal
                    self._draw_signal_panel(
                        processed, 
                        f"Processed: {signal_id} (MA Filter)", 
                        PROCESSED_SIGNAL_COLOR,
                        self.sidebar_width, 
                        y_position + panel_height,
                        self.plot_width,
                        panel_height
                    )
    
    def _draw_no_signals_message(self):
        """
        Draw a message indicating there are no signals to display.
        """
        message = "No signals available"
        text_surface = self.font.render(message, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(
            self.sidebar_width + self.plot_width // 2,
            self.status_bar_height + self.plot_height // 2
        ))
        self.surface.blit(text_surface, text_rect)
