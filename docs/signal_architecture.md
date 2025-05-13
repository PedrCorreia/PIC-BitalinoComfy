# Unified Signal Architecture for PIC-2025

This document describes the unified signal architecture of the PIC-2025 system, highlighting the core components, data flow, and best practices for implementation.

## Architecture Overview

The PIC-2025 system uses a two-registry architecture with a clear separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│ Signal          │     │ Signal          │     │ Plot            │     │ Visualization   │
│ Generators      │ ──► │ Registry        │ ──► │ Registry        │ ──► │ System          │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Core Components

1. **Signal Registry** (`SignalRegistry`)
   - Central singleton class for storing all available signals
   - Located in `src/plot/signal_registry.py`
   - Acts as the primary repository for all signal data
   - Thread-safe with validation and error handling

2. **Plot Registry** (`PlotRegistry`)
   - Singleton class for managing signals for visualization
   - Located in `src/plot/plot_registry.py`
   - Manages connections between signals and visualization nodes
   - Handles signal metadata for proper visualization

3. **Signal Input Node** (`SignalInputNode`)
   - Bridges between SignalRegistry and PlotRegistry
   - Located in `comfy/Registry/signal_input_node.py`
   - Key connector in the unified architecture

4. **Plot Registry Integration** (`PlotRegistryIntegration`)
   - Middleware connecting Plot Registry to visualization system
   - Located in `src/plot/plot_registry_integration.py`
   - Handles the transfer of signals from registry to visualization

5. **Plot Unit Node** (`PlotUnitNode`)
   - Manages visualization using PlotRegistry
   - Located in `comfy/Registry/plot_unit_node.py`
   - Handles display and user interaction

## Data Flow

The data flows through the system in the following order:

1. **Signal Generation**
   - Signal generators (like `RegistrySyntheticGenerator` or `RegistrySignalGenerator`) create signal data
   - Generated signals are registered in the SignalRegistry with unique IDs
   - Type validation ensures all signal IDs are properly formatted as strings

2. **Signal Connection**
   - `SignalInputNode` receives signal IDs from generators
   - Retrieves signal data from SignalRegistry
   - Registers the signal in PlotRegistry with proper metadata
   - Establishes connection between the node and signal for visualization

3. **Visualization**
   - `PlotUnitNode` connects to PlotRegistry through `PlotRegistryIntegration`
   - `PlotRegistryIntegration._update_thread` continuously monitors for new signals
   - Signals are visualized in real-time in a Pygame window

## Signal Registry

The `SignalRegistry` is responsible for storing all signals in the system:

```python
class SignalRegistry:
    def register_signal(self, signal_id, signal_data, metadata=None):
        """Register a signal with the registry"""
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return None
            
        # Converts signal_data to numpy array if needed
        # Stores signal and its metadata with thread safety
        
    def get_signal(self, signal_id):
        """Retrieve a signal from the registry"""
        # Returns the signal data for the given ID
        
    def get_signal_metadata(self, signal_id):
        """Get metadata for a signal"""
        # Returns metadata for the given signal
```

## Plot Registry

The `PlotRegistry` manages signals that should be visualized:

```python
class PlotRegistry:
    def register_signal(self, signal_id, signal_data, metadata=None):
        """Register a signal for visualization"""
        # Similar to SignalRegistry but specifically for visualization
        
    def connect_node_to_signal(self, node_id, signal_id):
        """Connect a visualization node to a signal"""
        # Validates signal_id is a string
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return False
            
        # Tracks which nodes are visualizing which signals
        
    def disconnect_node(self, node_id):
        """Disconnect a node from all signals"""
        # Removes all connections for a node
```

## Plot Registry Integration

The `PlotRegistryIntegration` connects the Plot Registry to the visualization system:

```python
class PlotRegistryIntegration:
    def connect_node_to_signal(self, node_id, signal_id):
        """Connect a node to a signal for visualization"""
        # Validate the signal_id
        if not isinstance(signal_id, str):
            logger.warning(f"Attempted to connect node {node_id} to non-string signal ID: {signal_id}")
            return False
            
        # Add signal to node's connections and update registry
```

## Using the Architecture

### Creating Signals

Signal generators should register their signals with the `SignalRegistry`:

```python
# Example from a generator node
registry = SignalRegistry.get_instance()
registry.register_signal(
    signal_id="ECG",
    signal_data=data,
    metadata={
        'color': (255, 0, 0),  # RGB color
        'sampling_rate': 100,  # Hz
        'source': 'synthetic'
    }
)
```

### Connecting Signals to Visualization

To visualize signals, connect your generator node to a `SignalInputNode` in the ComfyUI workflow:

1. Your generator node outputs the signal ID(s) as strings
2. `SignalInputNode` takes the signal ID as input
3. `SignalInputNode` bridges between SignalRegistry and PlotRegistry
4. Connect a `PlotUnitNode` to visualize all signals in PlotRegistry

### Multiple Signals

The architecture supports handling multiple signals at once:

```python
# Generators can output comma-separated lists of signal IDs
return (",".join(active_signals),)

# SignalInputNode can process these to connect multiple signals
```

## Best Practices

1. **Use Proper Signal IDs**
   - Use descriptive, unique string IDs for signals
   - Standard physiological signals use "EDA", "ECG", and "RR"
   - Never use non-string values as signal IDs

2. **Include Useful Metadata**
   - Always include sampling rate, source information, and colors
   - Add x-values for time-series data when available

3. **Proper Cleanup**
   - Implement `__del__` methods to disconnect nodes when deleted
   - Use registry disconnect methods to keep connections clean

4. **Type Safety**
   - Always check that signal IDs are strings
   - Validate signal data can be converted to numpy arrays
   - Handle errors gracefully with proper logging

5. **Use New Architecture**
   - Use `SignalInputNode` instead of legacy connector nodes
   - Avoid direct connections to `SignalRegistry` from visualization nodes
   - Follow the proper signal flow from generation to visualization

## Example Workflow

1. **Generate Synthetic Signals**
   - Use `RegistrySyntheticGenerator` to create EDA, ECG, or RR signals
   - Signals are registered in SignalRegistry with proper IDs

2. **Connect Signals to Visualization**
   - Connect the output (signal_id) of the generator to `SignalInputNode`
   - Set the `visualize` toggle to TRUE
   - Set display color and optional alias for better identification

3. **Visualize Signals**
   - Add a `PlotUnitNode` to your workflow
   - It automatically connects to PlotRegistry
   - All connected signals are visualized in real-time

## Troubleshooting

- If you see "connecting to signal: False" error, check that your signal IDs are strings
- If signals aren't visible, ensure IDs match between generators and connectors
- Use the `SignalDebugNode` to inspect signals and diagnose issues
- Verify signals exist in SignalRegistry before trying to visualize them

## Diagnostic Tools

The system includes several diagnostic tools:

1. **Signal Debug Node**
   - Shows comprehensive information about registered signals
   - Verifies if signals are properly registered
   - Checks if signals are marked for visualization

2. **Registry Monitor**
   - External tool for monitoring registry state
   - Located in `tools/registry_monitor.py`
   - Real-time view of all signals and connections

## Architecture Benefits

- **Decoupling**: Signal generators are separated from visualization
- **Flexibility**: Signals can be generated once and visualized multiple ways
- **Type Safety**: Validation at all levels prevents the "connecting to signal: False" error
- **Extensibility**: New signal sources can be easily added
- **Thread-safety**: Registries use locks to prevent race conditions

## Legacy Support

The architecture maintains backward compatibility with older nodes:

- Legacy connector nodes are marked with deprecation notices
- The system can still handle old workflows with proper redirection
- Use `SignalInputNode` for all new development
