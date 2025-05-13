# Signal Registry System Documentation

## Overview

The Signal Registry system provides a centralized way to manage and visualize signals in ComfyUI. The system consists of several components that work together to provide a flexible and powerful signal visualization framework.

## Components

### 1. PlotRegistry

The `PlotRegistry` class serves as a central repository for storing signals and their metadata. It keeps track of:
- Signal data
- Signal metadata (color, creation time, etc.)
- Signal-to-node connections

### 2. Signal Registry Connector

This node controls whether a signal is registered in the visualization system or not.

**Inputs:**
- `signal_id`: A unique identifier for the signal (string)
- `enabled`: Boolean toggle that controls whether the signal should be registered or removed
- `signal_data`: Optional signal data to register (if available)
- `display_color`: Color to use when visualizing the signal

**Behavior:**
- When `enabled` is TRUE: The signal ID is registered with the plot registry
- When `enabled` is FALSE: The signal ID is removed from the registry (if it exists)

### 3. Registry Plot Node

This node visualizes signals from the registry in a Pygame window.

**Inputs:**
- `reset`: Boolean to reset the plots
- `signal_id`: Optional specific signal to visualize
- `clear_registry`: Whether to clear the entire registry
- `auto_reset`: Whether to auto-reset on each run

### 4. Registry Signal Generator

This node generates synthetic signals and registers them in the registry.

**Inputs:**
- Various parameters for signal generation (type, frequency, amplitude, etc.)
- `signal_id`: ID to use for the generated signal

## How to Use

### Basic Signal Visualization Workflow

1. Generate a signal using `Registry Signal Generator`
2. Connect the signal ID to `Signal Registry Connector` with `enabled` set to TRUE
3. Add a `Registry Plot Node` to visualize the registered signals

### Toggling Signal Visibility

To toggle whether a signal is visible in the plot:

1. Connect the signal ID to a `Signal Registry Connector`
2. Toggle the `enabled` parameter:
   - TRUE: The signal will be visualized
   - FALSE: The signal will be removed from visualization

### Multiple Signals

You can visualize multiple signals by:

1. Creating multiple signal generators with different signal IDs
2. Connecting each signal ID to its own `Signal Registry Connector` with `enabled` set to TRUE
3. Using a single `Registry Plot Node` to visualize all registered signals

## Implementation Details

The system uses a combination of singleton patterns and thread-safe operations to ensure consistent signal handling across multiple nodes.

- `PlotRegistry`: Stores signal data and metadata
- `PlotRegistryIntegration`: Bridges between registry and visualization 
- `PlotUnit`: Handles the actual visualization in a Pygame window

## Example Workflow

```
[Registry Signal Generator] --> signal_id --> [Signal Registry Connector] --> signal_id --> [Another Node]
                                                    |
                                                    v
                                            [Registry Plot Node]
```

In this workflow:
1. `Registry Signal Generator` creates a signal and outputs its ID
2. `Signal Registry Connector` controls whether the signal is registered for visualization
3. `Registry Plot Node` visualizes all registered signals
