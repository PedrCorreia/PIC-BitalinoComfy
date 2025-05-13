# Debugging the Signal Registry System

This document provides guidance for troubleshooting issues with the Signal Registry system in the PIC-2025 custom nodes for ComfyUI.

## Registry System Integration

As of May 2025, the Signal Registry system has been fully integrated into the main package. All registry nodes are now initialized from a single location in the main `__init__.py` file, which resolves previous initialization conflicts. The registry nodes are now organized into dedicated categories:

- **Pedro_PIC/🌊 Signal Registry**: Core registry nodes for signal management
- **Pedro_PIC/🔬 Diagnostics**: Debug and logging tools

If you encounter issues with node registration, check the ComfyUI console for initialization errors with the pattern `PIC nodes: failed to import`.

## Troubleshooting Registry Integration Issues

If you're experiencing issues with node registration or initialization, try these steps:

### Node Registration Issues

1. **Check for Import Errors**: 
   - Review the ComfyUI console for error messages during startup
   - Look for the pattern `PIC nodes: failed to import [NodeName]`
   - Verify that all required dependencies are installed

2. **Reset the Registry**:
   Use the registry_reset.py tool to clear any corrupted registry state:
   ```
   cd tools
   python registry_reset.py
   ```

3. **Verify Node Categories**:
   If nodes are missing from the UI or appear in incorrect categories:
   - Check if NODE_CLASS_MAPPINGS contains the node class
   - Verify NODE_CATEGORY_MAPPINGS has the correct category
   - Ensure NODE_DISPLAY_NAME_MAPPINGS has an entry for the node

4. **Test with a Minimal Workflow**:
   - Create a simple workflow with just the Signal Registry components
   - Use the debug_registry_flow.json template in the workflows folder
   - Add diagnostic nodes to trace signal flow

### Advanced Integration Troubleshooting

If problems persist after the above steps, you can:

1. **Inspect Module Resolution**:
   ```python
   import sys
   print(f"Python path: {sys.path}")
   ```

2. **Verify Package Structure**:
   ```python
   import os
   print(os.listdir("path/to/PIC-2025"))
   print(os.listdir("path/to/PIC-2025/comfy"))
   ```

3. **Force Registry Refresh**:
   - Stop ComfyUI
   - Delete all `__pycache__` folders in the PIC-2025 directory
   - Restart ComfyUI with the `--reinstall-extensions` flag

For further help, submit an issue with the output from these diagnostic steps.

## Diagnostic Tools

Several debugging tools are available to help diagnose issues with the Signal Registry system:

### 1. Signal Debug Node

The `Signal Debug Node` is a ComfyUI node that provides detailed information about a signal in the registry.

**Usage:**
- Connect it to a signal ID to inspect details about that signal
- Set the log level for different levels of detail
- Enable "Inspect Registry" to check overall registry state

### 2. Signal Logger Node

The `Signal Logger Node` logs messages and signal information during workflow execution.

**Usage:**
- Connect it to points in your workflow where you want to log information
- Pass signal IDs and/or data to log details about signals
- Choose the log level to control verbosity

### 3. Registry Monitor Script

A standalone script that monitors the PlotRegistry in real-time outside ComfyUI.

**Location:** `tools/registry_monitor.py`

**Usage:**
```powershell
python tools/registry_monitor.py [options]
```

**Options:**
- `--interval/-i`: Refresh interval in seconds (default: 1.0)
- `--max-rows/-m`: Maximum rows to display (default: 10)
- `--watch/-w`: Watch a specific signal ID
- `--log/-l`: Save monitoring logs to file

### 4. Registry Reset Script

A utility script to reset the PlotRegistry state outside of ComfyUI.

**Location:** `tools/registry_reset.py`

**Usage:**
```powershell
python tools/registry_reset.py [options]
```

**Options:**
- `--quiet/-q`: Don't show status information

## Common Issues and Solutions

### Signals Not Showing Up in Visualization

**Check:**
1. Is the signal registered in the registry? Use Signal Debug Node to verify.
2. Is the SignalRegistryConnector node's "enabled" toggle set to TRUE?
3. Is the node connected to the signal via `connect_node_to_signal`? Check with Registry Monitor.
4. Does the signal contain valid data? Use Signal Debug Node to inspect.

### Pygame Window Freezing

**Possible causes:**
1. Two systems trying to initialize Pygame at the same time
2. Signal data processing taking too long
3. Thread conflicts between registry and plot systems

**Solutions:**
1. Make sure only one visualization system is active at a time
2. Use the debug nodes to track signal flow and identify bottlenecks
3. Run the Registry Monitor to check for signal size or processing issues

### Multiple Plot Windows

**Check:**
1. Are you using both standalone plot methods and registry-based plotting in the same workflow?
2. Are there legacy nodes still active in the system?

**Solutions:**
1. Move conflicting nodes to the `legacy` folder
2. Update `__init__.py` to only import the nodes you're currently using

## Debug Workflow

A debug workflow is provided at `workflows/debug_registry_flow.json` to help test the registry system:

1. Load this workflow in ComfyUI
2. Run it to generate a test signal, connect it to the registry, and visualize it
3. Use the Signal Debug Node to inspect the signal in the registry
4. Run the Registry Monitor script in parallel to watch signal flow

## Issues with PlotRegistry Integration

If there are issues with the integration between PlotRegistry and visualization:

1. Check if `PlotRegistryIntegration` is working properly
2. Verify that `_update_thread` is running (check logs)
3. Make sure `_process_registry_updates` is being called
4. Check for orphaned signals in the registry (signals without connected nodes)

Use the Registry Reset tool if the system gets into an inconsistent state.

## Advanced Debugging

For deeper debugging, you may need to:

1. Add more debug logging in key files:
   - `plot_registry.py`
   - `plot_registry_integration.py`
   - `signal_registry_connector.py`

2. Use the `logging` module to increase verbosity:
   ```python
   import logging
   logging.getLogger('PlotRegistry').setLevel(logging.DEBUG)
   ```

3. Track thread activity to identify deadlocks:
   ```python
   import threading
   print(f"Active threads: {threading.enumerate()}")
   ```

4. Check for zombie processes using external tools like Process Explorer (Windows) or `ps` (Linux/Mac)
