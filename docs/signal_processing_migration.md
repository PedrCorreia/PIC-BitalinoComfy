# Signal Processing Migration Guide

This document provides guidance for migrating from the old signal processing structure to the new structure.

## What Has Changed

1. **Removed Duplicate Signal Processor**
   - Removed `src/plot/signal_processor.py` which duplicated functionality
   - Signal processing now uses the core `NumpySignalProcessor` from `src/utils/signal_processing.py`

2. **Added Adapter Classes**
   - Created `SignalProcessingAdapter` to provide easy access to core functions
   - Created `SignalHistoryManager` to handle signal history specifically for visualization

3. **Updated Imports and References**
   - Updated imports to use the new structure
   - Fixed function calls to use the new adapter classes

## Migration Steps

### Step 1: Update Imports

Change from:
```python
from src.plot.signal_processor import SignalProcessor
```

Change to:
```python
from src.plot.utils import SignalHistoryManager, SignalProcessingAdapter
```

### Step 2: Update Class Instantiation

Change from:
```python
signal_processor = SignalProcessor()
```

Change to:
```python
history_manager = SignalHistoryManager()
```

### Step 3: Update History Management

Change from:
```python
signal_processor._update_history('ecg', data)
history = signal_processor.get_history('ecg')
```

Change to:
```python
history_manager.update_history('ecg', data)
history = history_manager.get_history('ecg')
```

### Step 4: Update Signal Processing Calls

Change from:
```python
processed = signal_processor.process_signal('ecg', data, 'bandpass')
```

Change to:
```python
data = history_manager.get_history('ecg')
processed = SignalProcessingAdapter.process_signal(
    data, 
    processing_type='bandpass', 
    lowcut=0.5, 
    highcut=10, 
    fs=1000
)
```

### Step 5: Update Analysis Calls

Change from:
```python
stats = signal_processor.analyze_signal('ecg', 'stats')
```

Change to:
```python
data = history_manager.get_history('ecg')
stats = SignalProcessingAdapter.analyze_signal(data, 'stats')
```

## Example Migration

### Before:

```python
from src.plot.signal_processor import SignalProcessor

# Initialize signal processor
processor = SignalProcessor()

# Update signal history
processor._update_history('ecg', new_data)

# Process signal
filtered = processor.process_signal('ecg', processor.get_history('ecg'), 'bandpass')

# Analyze signal
stats = processor.analyze_signal('ecg', 'stats')
```

### After:

```python
from src.plot.utils import SignalHistoryManager, SignalProcessingAdapter

# Initialize history manager
history_manager = SignalHistoryManager()

# Update signal history
history_manager.update_history('ecg', new_data)

# Get signal data
data = history_manager.get_history('ecg')

# Process signal
filtered = SignalProcessingAdapter.process_signal(data, 'bandpass', lowcut=0.5, highcut=10, fs=1000)

# Analyze signal
stats = SignalProcessingAdapter.analyze_signal(data, 'stats')
```

## Benefits of Migration

1. **Improved organization** - Clearer separation of concerns
2. **Reduced code duplication** - Single source of truth for signal processing
3. **Better maintainability** - Easier to update algorithms in one place
4. **Performance improvements** - Core implementations are optimized
5. **Advanced capabilities** - Access to more advanced signal processing features
