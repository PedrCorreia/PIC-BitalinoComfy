"""
View mode definitions for the PlotUnit visualization system.

This module defines the view modes used by the PlotUnit visualization system.
It's kept separate to avoid circular imports.
"""

from enum import Enum

class ViewMode(Enum):
    """
    Enum defining the available visualization modes.
    
    Values:
        RAW: Display raw signals
        PROCESSED: Display processed signals
        TWIN: Display raw and processed signals side by side
        STACKED: Display signals in a vertically stacked layout
        SETTINGS: Display settings panel
    """
    RAW = 0
    PROCESSED = 1
    TWIN = 2
    STACKED = 3
    SETTINGS = 4
