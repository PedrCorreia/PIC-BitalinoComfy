"""
Constants and configuration values for the PlotUnit visualization system.

This module centralizes all constants, default settings, and configuration values
used throughout the plot visualization system to ensure consistency and easy updates.
"""

import pygame  # Required for defining the FONT constant

# Font settings
DEFAULT_FONT = "Arial"
DEFAULT_FONT_SIZE = 14

# Window dimensions and layout
WINDOW_WIDTH = 530  # Adjusted for better fit with controls
WINDOW_HEIGHT = 780  # Adjusted for better content display
SIDEBAR_WIDTH = 50   # Wider sidebar for better button spacing
STATUS_BAR_HEIGHT = 30

# UI Layout settings
PLOT_PADDING = 15    # Padding around plots (increased for better separation)
TAB_HEIGHT = 30      # Height of tab buttons
CONTROL_PANEL_HEIGHT = 40  # Height of control panel
TWIN_VIEW_SEPARATOR = 10   # Space between twin views (increased for better separation)

# Position for status bar - TOP instead of bottom
STATUS_BAR_TOP = True

# Additional UI spacing parameters
BUTTON_MARGIN = 8         # Margin between buttons
SECTION_MARGIN = 20       # Margin between major sections
TITLE_PADDING = 12        # Padding below titles
TEXT_MARGIN = 5           # Margin around text elements
CONTROL_MARGIN = 10       # Margin between control elements
ELEMENT_PADDING = 8       # General padding for UI elements

# Colors
# Core UI colors
BACKGROUND_COLOR = (14, 14, 14)    # Dark background
SIDEBAR_COLOR = (24, 24, 24)       # Slightly lighter sidebar
BUTTON_COLOR = (24, 24, 24)        # Button background
BUTTON_COLOR_SETTINGS = (250, 24, 24)  # Red color for settings action buttons
ACCENT_COLOR = (0, 120, 215)       # Blue accent
TEXT_COLOR = (220, 220, 220)       # Light text
GRID_COLOR = (40, 40, 40)          # Dark grid

# Signal colors
RAW_SIGNAL_COLOR = (220, 180, 0)   # Amber for raw signals
PROCESSED_SIGNAL_COLOR = (0, 180, 220)  # Cyan for processed signals
ECG_SIGNAL_COLOR = (255, 0, 0)   # Green for ECG signals
EDA_SIGNAL_COLOR = (220, 120, 0)   # Orange for EDA signals

# Status colors
OK_COLOR = (0, 220, 0)             # Green for good status
WARNING_COLOR = (220, 220, 0)      # Yellow for warnings
ERROR_COLOR = (220, 0, 0)          # Red for errors

# Tooltip settings
TOOLTIP_BG_COLOR = (40, 40, 40)    # Dark background for tooltips
TOOLTIP_TEXT_COLOR = (220, 220, 220)  # Light text for tooltips
TOOLTIP_BORDER_COLOR = (80, 80, 80)  # Border color for tooltips
TOOLTIP_FONT_SIZE = 14             # Font size for tooltip text

# Performance thresholds
LOW_LATENCY_THRESHOLD = 0.05       # Below this is considered good (50ms)
HIGH_LATENCY_THRESHOLD = 0.05      # Above this is considered poor (200ms)
TARGET_FPS = 30                    # Target frames per second

# Default settings
DEFAULT_SETTINGS = {
    'caps_enabled': True,          # Enable FPS cap
    'light_mode': False,           # Dark mode by default
    'performance_mode': False,     # Quality mode by default
    'connected_nodes': 0,          # Updated dynamically
    'reset_plots': False,          # For reset plots button
    'reset_registry': False,       # For reset registry button
}

# Signal processing
SIGNAL_HISTORY_LENGTH = 1000        # Default history length
SIGNAL_DOWNSAMPLE_THRESHOLD = 1000 # Threshold for smart downsampling

# Enum value names (for consistency across the system)
VIEW_MODE_RAW = "RAW"
VIEW_MODE_PROCESSED = "PROCESSED"
VIEW_MODE_TWIN = "TWIN"
VIEW_MODE_SETTINGS = "SETTINGS"

ICON_FONT_SIZE = 24
TAB_ICON_FONT_SIZE = 28  # Larger font size for tab icons

