# Signal Processing in PIC-2025

## Overview

This document explains the organization of signal processing functionality in the PIC-2025 project.

## Architecture

The signal processing functionality is organized as follows:

1. **Core Signal Processing** - Located in `src/utils/signal_processing.py`
   - Contains the `NumpySignalProcessor`, `TorchSignalProcessor`, and `CudaSignalProcessor` classes
   - Provides comprehensive signal processing functionality including filters, peak detection, FFT, etc.
   - This is the source of truth for all signal processing algorithms

2. **Plot-specific Signal Processing** - Located in `src/plot/utils/`
   - `signal_processing_adapter.py` - Adapter for the core signal processing functionality
   - `signal_history.py` - Manages signal history specifically for visualization
   - Focuses on visualization-specific concerns rather than duplicating algorithms

## Usage Guidelines

### For Core Signal Processing

When implementing signal processing algorithms:

```python
from src.utils.signal_processing import NumpySignalProcessor

# Process a signal
filtered_signal = NumpySignalProcessor.bandpass_filter(signal, 0.5, 10, 1000)
```

### For Visualization Components

When working with the visualization system:

```python
from src.plot.utils import SignalProcessingAdapter, SignalHistoryManager

# Create a history manager
history_manager = SignalHistoryManager()
history_manager.update_history('ecg', new_data)

# Apply processing to the signal
processed = SignalProcessingAdapter.process_signal(
    history_manager.get_history('ecg'),
    processing_type='bandpass',
    lowcut=0.5,
    highcut=10,
    fs=1000
)
```

## Benefits of This Organization

1. **Separation of Concerns**
   - Core signal processing is independent of visualization
   - Visualization components can focus on display logic

2. **Single Source of Truth**
   - All signal processing algorithms are defined once in the core module
   - Prevents duplication and ensures consistency

3. **Flexibility**
   - Easy to switch between CPU (Numpy), GPU (Torch), or CUDA implementations
   - Specialized visualization requirements handled by adapters

4. **Maintainability**
   - Updates to signal processing algorithms only need to happen in one place
   - Clear boundaries between system components

## Implementation Notes

- The `SignalHistoryManager` maintains signal history in memory-efficient deque structures
- When algorithms need arrays, the deques are converted to numpy arrays on demand
- Signal processing adapters ensure consistent interfaces while delegating to the core implementations
