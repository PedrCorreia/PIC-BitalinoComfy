"""
Constants module for PlotUnit visualization system

This module defines constants used throughout the PlotUnit visualization system.
"""

# Window dimensions and layout
WINDOW_WIDTH = 530
WINDOW_HEIGHT = 780
SIDEBAR_WIDTH = 50
STATUS_BAR_HEIGHT = 30
PLOT_PADDING = 15
TAB_HEIGHT = 30
CONTROL_PANEL_HEIGHT = 40
TWIN_VIEW_SEPARATOR = 10
STATUS_BAR_TOP = True
BUTTON_MARGIN = 8
SECTION_MARGIN = 20
TITLE_PADDING = 12
TEXT_MARGIN = 5
CONTROL_MARGIN = 10
ELEMENT_PADDING = 8

# Colors
BACKGROUND_COLOR = (14, 14, 14)
BUTTON_COLOR = (24, 24, 24)
BUTTON_COLOR_SETTINGS = (250, 24, 24)
SIDEBAR_COLOR = (24, 24, 24)
ACCENT_COLOR = (0, 120, 215)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (40, 40, 40)

# Signal colors
RAW_SIGNAL_COLOR = (220, 180, 0)
PROCESSED_SIGNAL_COLOR = (0, 180, 220)
ECG_SIGNAL_COLOR = (255, 0, 0)
EDA_SIGNAL_COLOR = (220, 120, 0)

# Status colors
OK_COLOR = (0, 220, 0)
WARNING_COLOR = (220, 220, 0)
ERROR_COLOR = (220, 0, 0)

# Font sizes
FONT_SIZE = 14
ICON_FONT_SIZE = 24
TAB_ICON_FONT_SIZE = 28

# Default settings
DEFAULT_SETTINGS = {
    'caps_enabled': True,
    'light_mode': False,
    'performance_mode': False,
    'connected_nodes': 0,
    'reset_plots': False,
    'reset_registry': False,
}
