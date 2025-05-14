# PlotUnit Visualization System

This folder contains the PlotUnit visualization system, a comprehensive real-time signal visualization tool for the PIC-2025 project.

## Architecture

The PlotUnit system has been restructured with a modular architecture to improve maintainability, extensibility, and code organization.

### Core Components

- **PlotUnit**: Main class coordinating visualization
- **ViewMode**: Enum defining the available visualization modes
- **Constants**: Centralized configuration values
- **EventHandler**: Processes user input and system events

### View Components

Located in the `view/` directory:

- **BaseView**: Abstract base class for all views
- **RawView**: Raw signal visualization
- **ProcessedView**: Processed signal visualization
- **TwinView**: Side-by-side visualization
- **SettingsView**: Settings panel view

### UI Components

Located in the `ui/` directory:

- **Sidebar**: Navigation sidebar component
- **StatusBar**: Performance metrics display with latency and FPS monitoring
- **Buttons**: Button class (pending implementation)
- **Tooltips**: Tooltip functionality (pending implementation)

### Performance Monitoring

Located in the `performance/` directory:

- **LatencyMonitor**: Signal latency tracking
- **FPSCounter**: Frame rate monitoring

### Utilities

Located in the `utils/` directory:

- **Drawing**: PyGame drawing utilities
- **DataConverter**: Data format conversion utilities
- **TimeUtils**: Time formatting and calculation utilities
- **SignalHistoryManager**: Manages signal history for visualization
- **SignalProcessingAdapter**: Adapter for core signal processing utilities

## Usage

The PlotUnit visualization system provides an API for visualizing signals:

```python
from src.plot import PlotUnit

# Get the singleton instance
plot = PlotUnit.get_instance()

# Start visualization
plot.start()

# Queue data for visualization
plot.queue_data('signal_id', signal_data)
```

## Migration

To help with the transition from the old structure to the new modular architecture, a migration script is provided. This script maintains backward compatibility while moving to the new organization.

To migrate:

```python
from src.plot.migrate import migrate

# Run migration
migrate()
```

## Extensibility

The modular architecture allows for easy extension:

1. To add a new view mode:
   - Create a new view class extending BaseView
   - Add a new enum value to ViewMode
   - Register the new view in PlotUnit._initialize_components()

2. To add new signal processing functionality:
   - Add new methods to the NumpySignalProcessor class in `src/utils/signal_processing.py`
   - Update the SignalProcessingAdapter if needed for the visualization system

3. To customize appearance:
   - Modify constants in constants.py

## Signal Processing

The signal processing functionality has been reorganized:

- Core signal processing algorithms are centralized in `src/utils/signal_processing.py`
- The `NumpySignalProcessor`, `TorchSignalProcessor`, and `CudaSignalProcessor` classes provide optimized implementations
- `SignalHistoryManager` manages signal history for visualization
- `SignalProcessingAdapter` provides a simplified interface to core algorithms
- See `docs/signal_processing.md` and `docs/signal_processing_migration.md` for more details

## Performance Features

The new architecture includes enhanced performance monitoring:

- Real-time latency calculation
- FPS monitoring
- Signal timestamps tracking
- Status bar with color-coded performance metrics

## Contributors

- Original implementation: PIC-2025 team
- Restructuring: [Your Name]

## License

[Include license information here]
