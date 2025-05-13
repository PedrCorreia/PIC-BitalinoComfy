# PIC-2025 Signal Registry Nodes

## Signal Architecture Overview

The PIC-2025 system uses a two-registry architecture with a clear separation of concerns:

1. **SignalRegistry** - Central repository for all signals (`src/registry/signal_registry.py`)
2. **PlotRegistry** - Manages visualization of signals (`src/registry/plot_registry.py`)

> **Architecture Update (May 2025):** Registry components have been consolidated in the `src/registry` directory for better organization.

## Current Status (May 2025)

### Preferred Nodes (Current Architecture)

Use these nodes in all new workflows:

- **SignalInputNode** (signal_input_node.py): Bridges between SignalRegistry and PlotRegistry
- **PlotUnitNode** (plot_unit_node.py): Visualizes signals from PlotRegistry

### Deprecated Nodes (Legacy Architecture)

These nodes are kept only for backward compatibility and should not be used in new workflows:

- SignalConnectorNode (unified_signal_connector.py)
- SignalRegistryConnector (signal_registry_connector.py)
- RegistrySignalConnector (registry_signal_connector.py)
- SignalConnectorNode (signal_connector_node.py)

### Benefits of the Current Architecture

- Proper input validation for signal IDs
- Thread-safe operations
- Clear separation between signal generation and visualization
- Improved error handling
- Better debugging support

## Cleanup Roadmap

Future cleanup tasks for the PIC-2025 codebase:

1. Move all deprecated connector nodes to the legacy directory
2. Remove direct node registration from deprecated connectors
3. Create compatibility layer to redirect old workflows to new nodes
4. Update documentation to reflect the unified architecture

## See Also

For more information, see the unified signal architecture documentation in:
- [docs/signal_architecture.md](/docs/signal_architecture.md)
