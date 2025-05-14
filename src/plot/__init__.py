"""
PlotUnit Visualization System

This package provides a comprehensive visualization system for ComfyUI signals.
It enables real-time visualization of raw and processed signals from the signal registry.

Features:
- Real-time signal visualization with PyGame
- Multiple view modes (raw, processed, side-by-side)
- Performance monitoring (latency, FPS)
- Settings management
- Extensible architecture

Usage:
```python
from src.plot import PlotUnit

# Get the singleton instance
plot = PlotUnit.get_instance()

# Start visualization
plot.start()

# Queue data for visualization
plot.queue_data('signal_id', signal_data)
```
"""

# Import view modes first to avoid circular imports
from .view_mode import ViewMode

# Import main classes to expose at the package level
from .plot_unit import PlotUnit
from . import constants

# Import module namespaces for organization
from . import view
from . import ui
from . import performance
from . import utils
from . import controllers

# ViewMode is now imported from view_mode.py

# Define what gets imported with *
__all__ = [
    'PlotUnit',
    'ViewMode',
]
