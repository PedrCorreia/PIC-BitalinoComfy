# ComfyUI Plot Signal Sharing Network Architecture

This document provides a comprehensive explanation of the signal sharing network architecture implemented in the PIC-2025 custom nodes for ComfyUI. It focuses on how signals are generated, registered, retrieved, and visualized across different components.

## Table of Contents
1. [Overall Network Architecture](#overall-network-architecture)
2. [Signal Generation and Registration](#signal-generation-and-registration)
3. [Signal Retrieval and Visualization](#signal-retrieval-and-visualization)
4. [Connection Mechanism Between Components](#connection-mechanism-between-components)
5. [Data Flow Example](#data-flow-example)
6. [Best Practices and Tips](#best-practices-and-tips)

## Overall Network Architecture

The signal sharing network architecture follows a registry pattern with three main components:

1. **Signal Generators**: Components that create synthetic or real-world signals (EDA, ECG, RR).
2. **Signal Registry**: A central hub that stores and manages signals by their unique IDs.
3. **Signal Consumers**: Components that retrieve signals from the registry for visualization or further processing.

The architecture enables decoupling between signal generators and consumers, allowing for flexible workflows where signals can be generated and visualized independently.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Signal          │     │ Signal          │     │ Signal          │
│ Generators      │ ──► │ Registry        │ ──► │ Consumers       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
 (SyntheticData,         (SignalRegistry)       (SignalInputNode,
  MockSignalGen,                                PlotUnit)
  BitalinoReciever)
```

## Signal Generation and Registration

### Signal Generators

The system includes several signal generators:

1. **`SyntheticDataGenerator`**: Generates synthetic physiological data like EDA, ECG, and RR signals.
2. **`MockSignalGenerator`**: Creates test signals (sine waves, square waves, etc.) for system testing.
3. **`BitalinoReceiver`**: Captures real physiological data from Bitalino devices.

### Registry Sender

The `RegistrySender` class bridges signal generators with the registry:

```python
class RegistrySender:
    def __init__(self):
        self.registry = SignalRegistry.get_instance()
        
    def send_signal(self, signal_id, signal_data, metadata=None):
        # Convert data to tensor if needed
        # Register the signal in the registry
```

Signal generators call the `send_signal` method to register their signals:

```python
# Example from SyntheticDataGenerator
if self.use_registry:
    for signal_type, enabled in self.enabled_signals.items():
        if enabled and self.signal_data[signal_type]:
            metadata = {"sampling_rate": self.sampling_rate}
            self.registry_sender.send_signal(signal_type, list(self.signal_data[signal_type]), metadata)
```

### Signal Registry

The `SignalRegistry` is a singleton that stores all signals by ID:

```python
class SignalRegistry:
    _instance = None
    
    @staticmethod
    def get_instance():
        if SignalRegistry._instance is None:
            SignalRegistry._instance = SignalRegistry()
        return SignalRegistry._instance
        
    def register_signal(self, signal_id, signal_tensor):
        self.signals[signal_id] = signal_tensor
        
    def get_signal(self, signal_id):
        if signal_id in self.signals:
            return self.signals[signal_id]
        return None
```

## Signal Retrieval and Visualization

### Signal Input Node

The `SignalInputNode` retrieves signals from the registry and sends them to visualization components:

```python
class SignalInputNode:
    def __init__(self):
        # Get singleton PlotUnit instance
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        # Register as a connected node
        self.plot_unit.increment_connected_nodes()
        
    def process_signal(self, signal_id, enabled, color_r=220, color_g=180, color_b=0):
        # Handle comma-separated signal IDs
        if ',' in signal_id:
            signal_id = signal_id.split(',')[0]
            
        # Get signal from registry
        registry = SignalRegistry.get_instance()
        signal = registry.get_signal(signal_id)
        
        if signal is not None:
            # Send to visualization
            self.plot_unit.add_signal_data(signal, name=signal_id, color=(color_r, color_g, color_b))
```

### PlotUnit

The `PlotUnit` is responsible for visualizing signals and managing the display window:

```python
class PlotUnit:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def add_signal_data(self, signal_data, name="signal", color=None):
        # Convert tensor to numpy if needed
        # Put update message in queue for visualization thread
        self.event_queue.put({
            'type': 'update_data',
            'data_type': name,
            'data': data,
            'color': color
        })
```

The PlotUnit uses a thread-safe message queue system to handle updates from multiple nodes:

```python
def _process_queue(self):
    """Process incoming messages from the main thread"""
    try:
        while True:
            message = self.event_queue.get_nowait()
            
            if message['type'] == 'update_data':
                with self.data_lock:
                    data_type = message['data_type']
                    self.data[data_type] = message['data']
                    
                    # Register new signal types with color information
                    if data_type not in ['raw', 'filtered'] and data_type not in self.data:
                        self.data[data_type] = message['data']
                        if 'color' in message:
                            if not hasattr(self, 'signal_colors'):
                                self.signal_colors = {}
                            self.signal_colors[data_type] = message['color']
            
            elif message['type'] == 'node_connected' or message['type'] == 'node_disconnected':
                # These messages update the node connection counter display
                pass
                
            self.event_queue.task_done()
    except queue.Empty:
        pass
```

## Connection Mechanism Between Components

The connection between components is primarily managed through the Signal Registry:

1. **Signal ID Generation**: Each signal is assigned a unique ID (e.g., "EDA", "ECG", "RR", or custom IDs).
2. **Signal Registration**: Generators register signals with the registry using their IDs.
3. **Signal Lookup**: Consumers look up signals in the registry using the same IDs.
4. **Node Connection Tracking**: Visualization nodes track connected components through increment/decrement methods.

The `SynthNode` outputs signal IDs that can be directly connected to `SignalInputNode` instances:

```python
# In SynthNode.generate()
active_signals = []
if show_eda:
    active_signals.append('EDA')
if show_ecg:
    active_signals.append('ECG')
if show_rr:
    active_signals.append('RR')

return x, y, plot_result, data, ','.join(active_signals)
```

### Node Connection Tracking

The `PlotUnit` implements a node connection tracking mechanism to monitor how many nodes are connected to the visualization system:

```python
def increment_connected_nodes(self):
    """Increment the count of connected nodes"""
    self.settings['connected_nodes'] += 1
    logger.info(f"Node connected. Total connected nodes: {self.settings['connected_nodes']}")
    self.event_queue.put({'type': 'node_connected'})
    return self.settings['connected_nodes']

def decrement_connected_nodes(self):
    """Decrement the count of connected nodes"""
    if self.settings['connected_nodes'] > 0:
        self.settings['connected_nodes'] -= 1
    logger.info(f"Node disconnected. Total connected nodes: {self.settings['connected_nodes']}")
    self.event_queue.put({'type': 'node_disconnected'})
    return self.settings['connected_nodes']
```

These methods are called when:
- A node connects to the visualization system (`increment_connected_nodes`) in the `__init__` method
- A node disconnects from the visualization system (`decrement_connected_nodes`) in the `__del__` method
- In `SignalInputNode` and `PlotUnitNode` initialization and cleanup

The connection counter is displayed in the settings panel of the visualization window.

```python
# Example from PlotUnitNode
def __init__(self):
    # Get singleton PlotUnit instance
    self.plot_unit = PlotUnit.get_instance()
    self.plot_unit.start()
    # Register as a connected node
    self.plot_unit.increment_connected_nodes()
    
def __del__(self):
    # Unregister as a connected node
    plot_unit = PlotUnit.get_instance()
    plot_unit.decrement_connected_nodes()
```

## Data Flow Example

Here's a step-by-step example of the data flow in the system:

1. **Signal Generation**:
   - `SynthNode` creates synthetic data in `SyntheticDataGenerator`
   - Generator calls `self.registry_sender.send_signal("EDA", signal_data, metadata)`

2. **Signal Registration**:
   - `RegistrySender` converts the data to a tensor if necessary
   - Calls `self.registry.register_signal("EDA", tensor)`
   - Signal is now stored in the `SignalRegistry` singleton

3. **Signal Retrieval**:
   - `SignalInputNode` receives the signal ID ("EDA") from a ComfyUI workflow connection
   - Calls `registry.get_signal("EDA")` to retrieve the tensor
   - Signal data is now available in the node

4. **Signal Visualization**:
   - `SignalInputNode` forwards the data to `PlotUnit`
   - `PlotUnit` converts data if needed and adds it to the rendering queue
   - The signal is displayed in the visualization window
   - The connected node counter is incremented to track active visualizations

## Best Practices and Tips

1. **Signal IDs**:
   - Use clear, descriptive signal IDs
   - Standard physiological signals use "EDA", "ECG", and "RR"
   - Custom signals should use unique identifiers

2. **Working with Multiple Signals**:
   - The architecture supports comma-separated signal IDs for handling multiple signals
   - When sending multiple signals from a generator, join their IDs with commas

3. **Custom Signal Generators**:
   - Implement your own signal generator by following the pattern in `synthetic_data.py`
   - Use the `RegistrySender` to send signals to the registry

4. **Extending the System**:
   - New visualization components can be added by retrieving signals from the registry
   - Process signals before visualization for analysis, filtering, etc.

5. **Cleanup**:
   - Use the `reset()` method on `SignalRegistry` when needed to clear all signals

By following this architecture, you can create complex signal processing and visualization workflows in ComfyUI that are modular, flexible, and maintainable.
