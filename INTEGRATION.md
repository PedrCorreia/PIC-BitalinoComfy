# PIC-2025 Custom Nodes Integration

## Signal Registry System

The PIC-2025 signal registry system has been consolidated to ensure proper initialization and registration
of all nodes in a single initialization file.

### Changes Made

1. Merged initialization code from:
   - Main package `__init__.py`
   - Comfy package `comfy/__init__.py`

2. Organized nodes into appropriate categories:
   - Pedro_PIC/🧰 Tools
   - Pedro_PIC/🔬 Processing
   - Pedro_PIC/🔬 Bio-Processing
   - Pedro_PIC/📡 Bitalino
   - Pedro_PIC/🌊 Signal Registry (new category for registry nodes)
   - Pedro_PIC/🔬 Diagnostics (new category for debug tools)

3. Implemented robust error handling to prevent initialization failures

4. Updated to use the newest version of SignalInputNode (from signal_input_node_new.py)

5. Added clear emoji-based icons to all node display names for better UI experience

### Node Registration

All nodes are now registered in a single location via the main `__init__.py` file, which:
- Prevents duplicate registrations
- Ensures consistent node categories
- Provides clear error reporting

### Debugging Tools

Several debugging tools have been added to help troubleshoot the signal registry system:
- `SignalDebugNode`: ComfyUI node for detailed registry inspection
- `LoggerNode`: ComfyUI node for logging signal data and messages
- `registry_monitor.py`: Command-line tool for real-time registry monitoring
- `registry_reset.py`: Tool to reset the registry state
- `verify_registry.py`: Tool to verify proper registry setup
- `visualize_signal_flow.py`: Graphical tool for visualizing signal data

For more information on debugging the signal registry system, see the documentation in `docs/debug_signal_registry.md`.
