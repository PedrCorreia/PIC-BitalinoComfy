# Registry Connection Troubleshooting Guide

This guide will help you resolve common issues with the Registry nodes in the PIC-2025 package.

## Common Issues and Their Fixes

### 1. ComfyUI Workflow Input Configuration Issues

**Error**: 
Registry nodes don't execute or produce visualizations, though Plot Unit nodes work correctly.

**Fix**:
ComfyUI requires that all inputs must have a widget definition, even if the input is connected to another node. When creating workflows, ensure:

- Every input has a `widget` property
- Every input has a `value` property, even if it's connected 
- For linked inputs, include both the `link` property and the correct default `value`

Example of a properly configured input (connected):
```json
{
  "name": "signal_id",
  "type": "STRING",
  "link": 3,
  "widget": {
    "name": "signal_id"
  },
  "value": "test_sine_wave",
  "slot_index": 0
}
```

**Validation Tool:**

You can use the `registry_diagnostic.py` tool to check your workflows:

```
# Validate a specific workflow
python tools/registry_diagnostic.py --validate-workflow workflows/my_workflow.json

# Validate all workflows in the directory
python tools/registry_diagnostic.py --validate-all
```

For more detailed guidance, see the [Workflow Widget Configuration Guide](workflow_widget_guide.md).

### 2. Type Mismatches in Connections

**Error**: 
```
Return type mismatch between linked nodes: signal_data, received_type(STRING) mismatch input_type(SIGNAL)
```

**Fix**:
- Make sure you're connecting compatible types between nodes
- The `signal_data` input in the SignalRegistryConnector expects a SIGNAL type
- Remove any incorrect connections in your workflow

### 3. Seed Value Errors

**Error**:
```
ValueError: Seed must be between 0 and 2**32 - 1
```

**Fix**:
This has been fixed by ensuring that all seed values are properly normalized to the valid range.

```python
# Set random seed if provided
if seed != -1:
    # Ensure seed is within numpy's valid range (0 to 2^32-1)
    valid_seed = abs(seed) % (2**32)
    random.seed(valid_seed)
    np.random.seed(valid_seed)
```

### 4. Registry Not Displaying Signals

If signals are registered but not showing in visualization:

- Check that the registry plot node is properly connected
- Verify that signals are being correctly registered (use SignalDebugNode)
- Make sure that signal IDs match between generator and connector nodes
- **Critical: Ensure the SignalRegistryConnector output is connected to the RegistryPlotNode's signal_id input**

## Registry Flow Example

A working registry flow should follow this pattern:

1. **Signal Generator**: Creates signal data and registers with the registry
2. **Signal Registry Connector**: Connects the signal to the visualization system
3. **Registry Plot Node**: Displays signals from the registry
4. **Signal Debug Node**: Provides diagnostic information about registered signals

## Debugging Recommendations

1. Use the `SignalDebugNode` to verify that signals are properly registered
2. Check ComfyUI console logs for registration messages
3. Verify that the `PlotRegistry` singleton is properly initialized
4. Ensure that nodes are in the correct category in the UI 
5. **Check all inputs have widget definitions in the workflow JSON**
6. **Validate all node connections, especially from the connector to the plot node**

## Working Sample Workflows

Two sample workflows have been provided:

1. `c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\workflows\debug_registry_flow.json` - A comprehensive workflow with debug nodes
2. `c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\workflows\simple_registry_test.json` - A minimal workflow for testing basic functionality

You can load these workflows to test if your registry system is working correctly.

## Diagnostic Tools

Use the `registry_diagnostic.py` tool to monitor the state of the registry:

```python
from tools.registry_diagnostic import monitor_registry

# Monitor all registry entries
monitor_registry()

# Monitor a specific signal
monitor_registry(watch_signal="test_sine_wave")

# Reset and monitor
monitor_registry(reset=True)
```
