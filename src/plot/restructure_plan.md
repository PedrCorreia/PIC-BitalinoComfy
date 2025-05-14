# Plot Folder Restructuring Plan

## Current Issues:
- `plot_unit.py` is too large (860+ lines)
- Scattered functionality across several files
- Unclear organization and responsibilities
- Mixed concerns (UI, data handling, threading, etc.)

## Proposed Structure:

### Core Module:
- `__init__.py`: Module initialization and imports
- `constants.py`: All constants, colors, settings defaults

### Visual Components:
- `view/`
  - `__init__.py`
  - `base_view.py`: Abstract base class for views
  - `raw_view.py`: Raw signal visualization
  - `processed_view.py`: Processed signal visualization
  - `twin_view.py`: Side-by-side visualization
  - `settings_view.py`: Settings panel view

### UI Components:
- `ui/`
  - `__init__.py`
  - `sidebar.py`: Sidebar navigation
  - `status_bar.py`: Status bar with metrics (FPS, latency)
  - `buttons.py`: Button classes and handlers
  - `tooltips.py`: Tooltip functionality

### Core Classes:
- `plot_unit.py`: Main class (significantly reduced, delegating to other modules)
- `signal_processor.py`: Signal processing functionality
- `event_handler.py`: Event processing and handling

### Performance Monitoring:
- `performance/`
  - `__init__.py`
  - `latency_monitor.py`: Latency tracking
  - `fps_counter.py`: FPS calculation and display
  - `metrics.py`: General performance metrics

### Utilities:
- `utils/`
  - `__init__.py`
  - `data_converter.py`: Data conversion utilities
  - `drawing.py`: PyGame drawing utilities
  - `colors.py`: Color schemes and utilities
  - `time_utils.py`: Timing utilities

## Integration:
- Create a clean public API in `__init__.py`
- Update imports in dependent files
- Ensure backward compatibility with existing code

## Documentation:
- Add comprehensive docstrings to all files
- Create module-level documentation with examples
- Document the new architecture
