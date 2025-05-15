"""
ViewMode enum module for PlotUnit visualization system

This module provides the view mode enumeration used by the PlotUnit visualization system.
"""

from enum import Enum

class ViewMode(Enum):
    """Enum representing the different view modes in the PlotUnit visualization system."""
    RAW = 0
    PROCESSED = 1
    TWIN = 2
    SETTINGS = 3

# Define view mode constants that match the enum values for convenience
VIEW_MODE_RAW = ViewMode.RAW.value
VIEW_MODE_PROCESSED = ViewMode.PROCESSED.value  
VIEW_MODE_TWIN = ViewMode.TWIN.value
VIEW_MODE_SETTINGS = ViewMode.SETTINGS.value
