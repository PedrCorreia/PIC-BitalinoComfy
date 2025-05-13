# ComfyUI Workflow Widget Configuration Guide

This guide explains how to properly configure input widgets in ComfyUI workflows, especially when working with registry nodes in the PIC-2025 package.

## Key Principles

1. **Every input needs a widget definition**, even if connected to another node
2. **Every input needs a default value**, even if connected to another node
3. **Signal IDs must match** across connected nodes
4. Inputs with links must include both `link` and `widget/value` properties

## Common Issues and Solutions

### Issue: Connected input missing widget definition

```json
// INCORRECT - Missing widget definition
{
  "name": "signal_id",
  "type": "STRING",
  "link": 3,
  "slot_index": 0
}
```

```json
// CORRECT - Includes both link and widget definition
{
  "name": "signal_id",
  "type": "STRING",
  "link": 3,
  "widget": {
    "name": "signal_id"
  },
  "value": "test_signal",
  "slot_index": 0
}
```

### Issue: Connected input missing default value

```json
// INCORRECT - Missing default value
{
  "name": "signal_id",
  "type": "STRING",
  "link": 3,
  "widget": {
    "name": "signal_id"
  },
  "slot_index": 0
}
```

```json
// CORRECT - Includes default value
{
  "name": "signal_id",
  "type": "STRING",
  "link": 3,
  "widget": {
    "name": "signal_id"
  },
  "value": "test_signal",
  "slot_index": 0
}
```

## Example of Properly Configured Node

```json
{
  "id": 3,
  "type": "RegistryPlotNode",
  "inputs": [
    {
      "name": "reset",
      "type": "BOOLEAN",
      "link": null,
      "widget": {
        "name": "reset"
      },
      "value": false
    },
    {
      "name": "signal_id",
      "type": "STRING",
      "link": 2,
      "widget": {
        "name": "signal_id"
      },
      "value": "test_signal",
      "slot_index": 0
    },
    {
      "name": "clear_registry",
      "type": "BOOLEAN",
      "link": null,
      "widget": {
        "name": "clear_registry"
      },
      "value": false
    }
  ]
}
```

## Registry Node Connections

For Registry nodes to work properly, specific connections must be established:

### Required Connection Pattern

```
SignalGenerator → SignalRegistryConnector → RegistryPlotNode
```

### Critical Connection

The most important connection is from the SignalRegistryConnector to the RegistryPlotNode:

```
SignalRegistryConnector.signal_id (output) → RegistryPlotNode.signal_id (input)
```

### Links Array Format

In the workflow JSON, links are defined as arrays with this format:
```
[link_id, source_node_id, source_output_index, target_node_id, target_input_index, "TYPE"]
```

Example link from SignalRegistryConnector (node 2) to RegistryPlotNode (node 3):
```json
[6, 2, 0, 3, 1, "STRING"]
```

## Using the Diagnostic Tool

You can validate your workflows using the registry_diagnostic.py tool:

```bash
python tools/registry_diagnostic.py --validate-workflow workflows/my_workflow.json
```

Or validate all workflows in the directory:

```bash
python tools/registry_diagnostic.py --validate-all
```

The diagnostic tool now checks for:
- Missing widget definitions
- Missing default values
- Missing registry connections

## Fixing Common Issues

1. **Missing widget definitions**:
   - Add a `widget` property to all inputs
   - The widget name should match the input name

2. **Missing default values**:
   - Add a `value` property with an appropriate default
   - Make sure the type matches the expected input type

3. **Signal ID mismatch**:
   - Ensure the signal_id is consistent across connected nodes
   - Set the same default value in all nodes that should receive the same signal

4. **Missing registry connections**:
   - Make sure SignalRegistryConnector's signal_id output connects to RegistryPlotNode's signal_id input
   - Verify the link exists in the workflow's links array

## Sample Workflows

Study these working examples:
1. `simple_registry_test.json` - Minimal workflow with registry nodes
2. `debug_registry_flow.json` - Comprehensive workflow with debugging tools

By following these guidelines, you can avoid many common issues with registry nodes in ComfyUI.
