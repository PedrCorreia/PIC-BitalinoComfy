"""
Controller components for the PlotUnit visualization system.

This module provides controller components that connect UI elements with functionality,
creating a clean separation of concerns between presentation and business logic.
"""

from .button_controller import ButtonController

# Define what gets imported with *
__all__ = [
    'ButtonController',
]
