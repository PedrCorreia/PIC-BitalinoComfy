"""
Utility modules for the PlotUnit visualization system.

This module provides utility functions for the PlotUnit system,
including drawing, data conversion, signal history, signal processing, and timing utilities.
"""

from .drawing import draw_grid, draw_signal, draw_text
from .data_converter import convert_to_numpy, normalize_signal, resample_signal
from .time_utils import get_timestamp, format_time, format_time_since
from .signal_history import SignalHistoryManager
from .signal_processing_adapter import SignalProcessingAdapter
from .signal_generator import generate_test_signals, update_test_signals

# Try to import debug_plots functions
try:
    from .debug_plots import generate_debug_data, initialize_test_plots, update_test_plots
    __all__ = [
        'draw_grid',
        'draw_signal',
        'draw_text',
        'convert_to_numpy',
        'normalize_signal',
        'resample_signal',
        'get_timestamp',
        'format_time',
        'format_time_since',
        'SignalHistoryManager',
        'SignalProcessingAdapter',
        'generate_test_signals',
        'update_test_signals',
        'generate_debug_data',
        'initialize_test_plots',
        'update_test_plots',
    ]
except ImportError:
    # If import fails, just expose the basic functions
    __all__ = [
        'draw_grid',
        'draw_signal',
        'draw_text',
        'convert_to_numpy',
        'normalize_signal',
        'resample_signal',
        'get_timestamp',
        'format_time',
        'format_time_since',
        'SignalHistoryManager',
        'SignalProcessingAdapter',
        'generate_test_signals',
        'update_test_signals',
    ]
