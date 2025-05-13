# Signal Registry Debugging Tools

This directory contains tools for debugging and troubleshooting the Signal Registry system in the PIC-2025 custom nodes for ComfyUI.

## Available Tools

### 1. Registry Diagnostic (`registry_diagnostic.py`)

An enhanced diagnostic tool for monitoring registry status and validating workflow files, with special focus on proper widget configuration and node connections.

**Usage:**
```powershell
# Monitor registry status
python tools/registry_diagnostic.py --interval 1 --watch-signal test_signal --reset

# Validate workflow files
python tools/registry_diagnostic.py --validate-workflow workflows/my_workflow.json
python tools/registry_diagnostic.py --validate-all
```

**Options:**
- `--interval`: Update interval in seconds for monitoring (default: 1.0)
- `--watch-signal`: Monitor a specific signal ID
- `--reset`: Reset the registry before monitoring
- `--validate-workflow`: Path to workflow JSON file to validate
- `--validate-all`: Validate all workflows in the workflows directory

### 2. Registry Monitor (`registry_monitor.py`) 

A real-time monitor for the PlotRegistry that displays signal information and registry state in the terminal.

**Usage:**
```powershell
python tools/registry_monitor.py --interval 1 --max-rows 10 --watch signal_id --log
```

**Options:**
- `--interval/-i`: Refresh interval in seconds (default: 1.0)
- `--max-rows/-m`: Maximum rows to display (default: 10)
- `--watch/-w`: Watch a specific signal ID
- `--log/-l`: Save monitoring logs to file

### 2. Registry Reset (`registry_reset.py`)

A utility to reset the PlotRegistry state, useful when troubleshooting issues.

**Usage:**
```powershell
python tools/registry_reset.py --quiet
```

**Options:**
- `--quiet/-q`: Don't show status information

### 3. Registry Verification (`verify_registry.py`)

Verifies that the PlotRegistry components are properly set up and functioning.

**Usage:**
```powershell
python tools/verify_registry.py
```

### 4. Signal Flow Visualizer (`visualize_signal_flow.py`)

A graphical tool that visualizes signal values flowing through the registry.

**Usage:**
```powershell
python tools/visualize_signal_flow.py --signals signal_id1 signal_id2 --interval 100 --history 100
```

**Options:**
- `--signals/-s`: Specific signal IDs to watch (empty means all signals)
- `--interval/-i`: Update interval in milliseconds (default: 100)
- `--history/-l`: History length in data points (default: 100)

## Debugging Workflow

Here's a recommended workflow for debugging signal registry issues:

1. **Validate your workflow files**:
   ```powershell
   python tools/registry_diagnostic.py --validate-workflow workflows/my_workflow.json
   ```
   Fix any widget configuration or connection issues identified.

2. **Verify the registry setup**: 
   ```powershell
   python tools/verify_registry.py
   ```
   Fix any issues identified before proceeding.

3. **Reset the registry**:
   ```powershell
   python tools/registry_reset.py
   ```
   Start with a clean slate.

4. **Start the diagnostic monitor**:
   ```powershell
   python tools/registry_diagnostic.py --watch-signal my_signal
   ```
   Keep this running in a separate terminal to observe registry activity.

5. **Run your ComfyUI workflow**:
   Load and run the workflow you want to debug in ComfyUI.

6. **Visualize signal flow** (if needed):
   ```powershell
   python tools/visualize_signal_flow.py
   ```
   Use this tool if you need to see how signal values change over time.

## ComfyUI Debugging Nodes

In addition to these standalone tools, the PIC-2025 system provides ComfyUI nodes for in-workflow debugging:

- **Signal Debug Node**: Provides detailed signal information
- **Signal Logger Node**: Logs messages and signal data

Include these nodes in your workflow to get additional debugging information.

## Additional Resources

For more detailed debugging information, see:

- `docs/registry_troubleshooting.md`: Common registry issues and solutions
- `docs/workflow_widget_guide.md`: Guide for proper workflow widget configuration 
- `docs/debug_signal_registry.md`: Detailed debugging guide
- `docs/signal_registry_system.md`: Signal Registry system documentation
- `workflows/debug_registry_flow.json`: Comprehensive debugging workflow
- `workflows/simple_registry_test.json`: Minimal example workflow
