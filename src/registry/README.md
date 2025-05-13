# PIC-2025 Registry System

This directory contains the core registry components for the PIC-2025 signal system.

## Components

- **SignalRegistry**: Central repository for all signals in the system
- **PlotRegistry**: Manages visualization connections and signal display
- **PlotRegistryIntegration**: Bridge between the registry and visualization systems

## Architecture

The PIC-2025 system uses a two-registry architecture with a clear separation of concerns:

1. **SignalRegistry** acts as the central repository for all signal data
2. **PlotRegistry** handles visualization-specific concerns
3. **PlotRegistryIntegration** connects the registry to visualization components

## Usage

All components use the singleton pattern for global access:

```python
from src.registry.signal_registry import SignalRegistry
from src.registry.plot_registry import PlotRegistry

# Get registry instances
signal_registry = SignalRegistry.get_instance()
plot_registry = PlotRegistry.get_instance()

# Register a signal
signal_registry.register_signal("my_signal", signal_data, {"source": "generator"})
```

## Reorganization (May 2025)

These registry files were previously located in `src/plot` and have been moved to this dedicated directory to better organize the codebase and clarify the architecture.
