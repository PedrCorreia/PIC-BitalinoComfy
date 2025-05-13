# PIC-2025 Signal Architecture

This document describes the unified signal architecture for the PIC-2025 ComfyUI extension.

## Core Architecture

The architecture is built around two main registry classes:

1. **SignalRegistry** - For storing and managing signals
2. **PlotRegistry** - For managing visualization of signals

The flow is designed to be simple:

```
Signal Generators → SignalRegistry → SignalInputNode → PlotRegistry → Visualization
```

## Key Components

### SignalRegistry

Located at `src/plot/signal_registry.py`, this is the central repository for all signals in the system.

- Registers signals with unique IDs
- Stores signal metadata
- Provides access to signals via IDs
- Thread-safe singleton design

### PlotRegistry

Located at `src/plot/plot_registry.py`, this handles visualizing signals.

- Tracks nodes connected to signals 
- Manages which signals should be visualized
- Handles metadata for visualization (colors, etc)
- Thread-safe singleton design

### SignalInputNode

Located at `comfy/Registry/signal_input_node.py`, this node connects SignalRegistry to PlotRegistry.

- Takes a signal ID from SignalRegistry
- Transfers the signal to PlotRegistry
- Allows customization (color, alias)
- Controls visualization toggle

## Signal Flow

1. **Signal Generation**: Generators (like RegistrySyntheticGenerator) create signals and register them in SignalRegistry
   - Each signal is given a unique ID

2. **Signal Connection**: SignalInputNode receives signal IDs and connects them to PlotRegistry
   - Optional alias and color can be provided
   
3. **Visualization**: PlotUnit visualizes signals from PlotRegistry
   - Background thread checks for updates

## Usage Guidelines

### For Signal Generators:

1. Get the SignalRegistry instance
2. Register your signal with a unique ID
3. Return the ID for downstream nodes

### For Signal Consumers:

1. Accept signal ID as input
2. Use SignalInputNode to connect the signal for visualization 

## Example Workflow

```
RegistrySyntheticGenerator → [signal_id:EDA] → SignalInputNode → Visualization
```

With this architecture, signals can be shared across multiple visualizers, and multiple signals can be visualized together.
