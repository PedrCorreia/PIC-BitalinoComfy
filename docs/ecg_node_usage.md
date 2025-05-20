# ECG Node with Registry Integration

## Overview

The ECG Node has been enhanced to fully integrate with the PIC-2025 Signal Registry system. This allows processed ECG signals to be visualized in real-time across the ComfyUI workflow, with configurable peak detection and heart rate calculation.

## New Features

- **Signal ID integration**: Accept input signals from the registry and output processed signals back to the registry
- **Peak visualization control**: Toggle peak markers on/off with a boolean input
- **Auto-registration**: Automatically register processed signals with the registry
- **Background processing**: Optimize CPU usage while maintaining real-time visualization
- **Heart rate calculation**: Real-time heart rate calculation from detected R peaks

## Parameters

### Required Inputs

- **signal_deque**: Array of (timestamp, value) pairs for ECG data
- **viz_buffer_size**: Buffer size for visualization (default: 1000)
- **feature_buffer_size**: Buffer size for feature extraction (default: 5000)
- **show_peaks**: Boolean to control peak visualization (default: True)
- **auto_register**: Boolean to control automatic registration with registry (default: True)
- **output_signal_id**: ID for the processed signal in registry (default: "ECG_PROCESSED")

### Optional Inputs

- **input_signal_id**: Signal ID to retrieve data from registry (if provided)

### Outputs

- **Visualization_Data**: Array of [timestamp, value, is_peak] rows
- **Heart_Rate**: Calculated heart rate in beats per minute (BPM)
- **Rpeak**: Boolean indicating if the current sample is a peak
- **Signal_ID**: The signal ID for registry integration (to connect to other nodes)

## Sample Usage

### Basic ECG Processing

1. Connect ECG data array to the `signal_deque` input
2. Configure buffer sizes based on your needs
3. Toggle `show_peaks` to show/hide peak markers
4. Use the `Heart_Rate` output for further processing

### Registry Integration

1. Connect ECG data array to the `signal_deque` input
2. Set `auto_register` to True
3. Provide a unique `output_signal_id` (e.g., "ECG_PROCESSED")
4. Connect the `Signal_ID` output to a Signal Connector Node
5. Visualize with the Plot Registry Node

### Processing Registry Signals

1. Provide an `input_signal_id` from another node
2. Set desired options for peak detection and visualization
3. Set `auto_register` to True with a unique `output_signal_id`
4. Connect the `Signal_ID` output to downstream nodes

## Example Workflow

```
[Signal Generator] --> signal_id --> [ECG Node] --> Signal_ID --> [Signal Connector]
                                          |                              |
                                          v                              v
                                    [Heart Rate Node]            [Plot Registry Node]
```

## Performance Considerations

- The ECG node runs processing in a background thread to maintain UI responsiveness
- For high-frequency signals, consider increasing the buffer sizes
- For resource-constrained systems, adjust the thread sleep time in the background processing

## Technical Implementation

The node uses the Signal Registry system to store and retrieve signal data, with special metadata to indicate peaks and heart rate information. This allows for advanced visualization options in the Plot Registry Node.
