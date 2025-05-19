"""
Utility modules for the PlotUnit visualization system.

This module provides utility functions for the PlotUnit system,
including drawing, data conversion, signal history, signal processing, and timing utilities.
"""

from .drawing import draw_grid, draw_signal, draw_text
from .time_utils import get_timestamp, format_time, format_time_since
from .signal_history import SignalHistoryManager
from .signal_processing_adapter import SignalProcessingAdapter
from .signal_generator import generate_test_signals, update_test_signals


# Try to import debug_plots functions
try:

    __all__ = [
        'draw_grid',
        'draw_signal',
        'draw_text',
        'get_timestamp',
        'format_time',
        'format_time_since',
        'SignalHistoryManager',
        'SignalProcessingAdapter',
        'generate_test_signals',
        'update_test_signals',

        
    ]
except ImportError:
    __all__ = [
        'draw_grid',
        'draw_signal',
        'draw_text',
        'get_timestamp',
        'format_time',
        'format_time_since',
        'SignalHistoryManager',
        'SignalProcessingAdapter',
        'generate_test_signals',
        'update_test_signals',
        
    ]
