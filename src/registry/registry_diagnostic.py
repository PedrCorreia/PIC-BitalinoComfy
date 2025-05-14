#!/usr/bin/env python
# filepath: c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\tools\registry_diagnostic.py
"""
Registry Diagnostic Tool

This script helps diagnose issues with the PlotRegistry system by monitoring 
active signals and connections in real time.

Usage:
    python registry_diagnostic.py

Options:
    --watch-signal=SIGNAL_ID   Monitor a specific signal ID
    --interval=SECONDS         Update interval (default: 1.0 seconds)
    --reset                    Reset the registry before monitoring
"""

import os
import sys
import time
import argparse
from colorama import init, Fore, Back, Style

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import registry - try multiple possible import paths
registry_available = False
try:
    # This is the most common location
    from src.registry.plot_registry import PlotRegistry
    registry_available = True
    print(f"{Fore.GREEN}Successfully imported PlotRegistry from src.registry{Style.RESET_ALL}")
except ImportError:
    try:
        # Try with full path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from src.registry.plot_registry import PlotRegistry
        registry_available = True
        print(f"{Fore.GREEN}Successfully imported PlotRegistry using absolute path{Style.RESET_ALL}")
    except ImportError:
        try:
            # Try from comfy module
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            comfy_path = os.path.join(parent_dir, 'comfy', 'Registry')
            if os.path.exists(os.path.join(comfy_path, 'plot_registry.py')):
                sys.path.append(comfy_path)
                from plot_registry import PlotRegistry
                registry_available = True
                print(f"{Fore.GREEN}Successfully imported PlotRegistry from comfy/Registry{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to import PlotRegistry. Check your Python path.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Looked in: {parent_dir}/src/plot and {comfy_path}{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.RED}Failed to import PlotRegistry. Check your Python path.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Could not find plot_registry.py in any expected locations{Style.RESET_ALL}")


def get_registry_status():
    """Get comprehensive registry status information"""
    if not registry_available:
        return "Registry not available - import failed"
        
    try:
        registry = PlotRegistry.get_instance()
        status = {}
        
        # Get basic info
        status['active_signals'] = list(registry.signals.keys()) if hasattr(registry, 'signals') else []
        status['signal_count'] = len(status['active_signals'])
        
        # Get node connections
        status['connected_nodes'] = []
        if hasattr(registry, 'node_connections'):
            for node_id, signals in registry.node_connections.items():
                status['connected_nodes'].append({
                    'node_id': node_id,
                    'connected_signals': list(signals)
                })
        
        # Get signal details
        status['signals'] = {}
        for signal_id in status['active_signals']:
            signal_data = registry.get_signal(signal_id)
            if signal_data is not None:
                if hasattr(signal_data, 'shape'):
                    shape = signal_data.shape
                elif isinstance(signal_data, dict) and 'tensor' in signal_data and hasattr(signal_data['tensor'], 'shape'):
                    shape = signal_data['tensor'].shape
                else:
                    shape = "unknown"
                    
                # Get metadata
                metadata = registry.get_signal_metadata(signal_id)
                
                status['signals'][signal_id] = {
                    'shape': shape,
                    'metadata': metadata
                }
        
        return status
    except Exception as e:
        return f"Error getting registry status: {str(e)}"
    

def display_registry_status(status, watch_signal=None):
    """Pretty print the registry status"""
    if isinstance(status, str):
        print(f"{Fore.RED}{status}{Style.RESET_ALL}")
        return
    
    # Print basic stats
    print(f"\n{Back.BLUE}{Fore.WHITE}===== Registry Status ====={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Active Signals: {status['signal_count']}{Style.RESET_ALL}")
    
    # Print signal IDs
    print(f"\n{Fore.YELLOW}Signal IDs:{Style.RESET_ALL}")
    for signal_id in status['active_signals']:
        if watch_signal and signal_id == watch_signal:
            print(f"{Fore.GREEN}* {signal_id} (WATCHING){Style.RESET_ALL}")
        else:
            print(f"* {signal_id}")
    
    # Print node connections
    print(f"\n{Fore.YELLOW}Connected Nodes:{Style.RESET_ALL}")
    for node in status['connected_nodes']:
        print(f"* {node['node_id']} -> {', '.join(node['connected_signals'])}")
    
    # Print detailed signal info
    if watch_signal and watch_signal in status['signals']:
        print(f"\n{Fore.GREEN}=== Details for {watch_signal} ==={Style.RESET_ALL}")
        signal_info = status['signals'][watch_signal]
        print(f"Shape: {signal_info['shape']}")
        print(f"Metadata: {signal_info['metadata']}")


def check_workflow_widgets(workflow_path):
    """
    Analyze a ComfyUI workflow JSON to check for missing widget definitions.
    Returns a list of issues found.
    
    Parameters:
    - workflow_path (str): Path to the workflow JSON file
    """
    import json
    
    if not os.path.exists(workflow_path):
        return [f"Workflow file not found: {workflow_path}"]
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
    except Exception as e:
        return [f"Failed to load workflow: {str(e)}"]
    
    issues = []
    
    # Check nodes
    for node in workflow.get('nodes', []):
        node_id = node.get('id', 'unknown')
        node_type = node.get('type', 'unknown')
        
        # Check inputs
        for input_idx, input_data in enumerate(node.get('inputs', [])):
            input_name = input_data.get('name', f'input_{input_idx}')
            link = input_data.get('link')
            
            # Check if input is connected but missing widget or value
            if link is not None:
                if 'widget' not in input_data:
                    issues.append(f"Node {node_id} ({node_type}): Connected input '{input_name}' is missing widget definition")
                if 'value' not in input_data:
                    issues.append(f"Node {node_id} ({node_type}): Connected input '{input_name}' is missing value definition")
    
    # Check links
    links = workflow.get('links', [])
    for link in links:
        if len(link) < 6:
            issues.append(f"Link {link[0]} has invalid format: {link}")
            continue
            
        link_id, from_node, from_output, to_node, to_input, link_type = link
        
        # Check if the nodes referenced in links exist
        node_ids = [node.get('id') for node in workflow.get('nodes', [])]
        if from_node not in node_ids:
            issues.append(f"Link {link_id}: Source node {from_node} not found in workflow")
        if to_node not in node_ids:
            issues.append(f"Link {link_id}: Target node {to_node} not found in workflow")
    
    return issues


def validate_registry_connections(workflow_path):
    """
    Specifically checks the connections between registry nodes in a workflow.
    Ensures that SignalRegistryConnector outputs are properly connected to RegistryPlotNode inputs.
    
    Parameters:
    - workflow_path (str): Path to the workflow JSON file
    
    Returns:
    - list: List of issues found with registry connections
    """
    import json
    
    if not os.path.exists(workflow_path):
        return [f"Workflow file not found: {workflow_path}"]
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
    except Exception as e:
        return [f"Failed to load workflow: {str(e)}"]
    
    issues = []
    node_map = {}  # Map node IDs to their types
    connector_outputs = {}  # Map connector node IDs to their output links
    plot_inputs = {}  # Map plot node IDs to their input links
    
    # First pass: Catalog node types and their IDs
    for node in workflow.get('nodes', []):
        node_id = node.get('id', 'unknown')
        node_type = node.get('type', 'unknown')
        node_map[node_id] = node_type
        
        # Track SignalRegistryConnector outputs
        if node_type == 'SignalRegistryConnector':
            connector_outputs[node_id] = []
            for output_idx, output in enumerate(node.get('outputs', [])):
                if output.get('name') == 'signal_id':
                    connector_outputs[node_id] = output.get('links', [])
        
        # Track RegistryPlotNode signal_id inputs
        if node_type == 'RegistryPlotNode':
            plot_inputs[node_id] = None
            for input_idx, input_data in enumerate(node.get('inputs', [])):
                if input_data.get('name') == 'signal_id':
                    plot_inputs[node_id] = input_data.get('link')
    
    # Second pass: Analyze links
    if len(plot_inputs) > 0 and len(connector_outputs) > 0:
        # Check if any connector output is connected to a plot input
        connected_plot_nodes = set()
        for conn_id, output_links in connector_outputs.items():
            for link_id in output_links:
                for plot_id, input_link in plot_inputs.items():
                    if input_link == link_id:
                        connected_plot_nodes.add(plot_id)
        
        # Check for plot nodes not connected to a connector
        for plot_id in plot_inputs:
            if plot_id not in connected_plot_nodes:
                issues.append(
                    f"RegistryPlotNode (ID: {plot_id}) is not connected to any SignalRegistryConnector output."
                )
    
    # No registry components in workflow
    if len(plot_inputs) == 0:
        issues.append("No RegistryPlotNode nodes found in workflow")
    if len(connector_outputs) == 0:
        issues.append("No SignalRegistryConnector nodes found in workflow")
    
    return issues


def monitor_registry(interval=1.0, watch_signal=None, reset=False):
    """Monitor the registry status in real time"""
    if not registry_available:
        print(f"{Fore.RED}Registry not available - cannot monitor{Style.RESET_ALL}")
        return
    
    if reset:
        try:
            registry = PlotRegistry.get_instance()
            registry.reset()
            print(f"{Fore.GREEN}Registry reset complete{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to reset registry: {str(e)}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Starting registry monitor (Press Ctrl+C to exit)...{Style.RESET_ALL}")
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Get and display status
            status = get_registry_status()
            display_registry_status(status, watch_signal)
            
            # Wait for next update
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"{Fore.CYAN}\nMonitor stopped{Style.RESET_ALL}")


def validate_workflow(workflow_path):
    """
    Validate a workflow file and print a report of issues.
    
    Parameters:
    - workflow_path (str): Path to the workflow JSON file
    """
    print(f"\n{Fore.YELLOW}Validating workflow: {workflow_path}{Style.RESET_ALL}")
    
    # Check for widget configuration issues
    widget_issues = check_workflow_widgets(workflow_path)
    
    # Check registry connections if it's a registry workflow
    registry_connection_issues = validate_registry_connections(workflow_path)
    
    # Combine all issues
    all_issues = widget_issues + registry_connection_issues
    
    if not all_issues:
        print(f"{Fore.GREEN}No issues found! All inputs have proper widget definitions and connections.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Found {len(all_issues)} potential issues:{Style.RESET_ALL}")
        
        if widget_issues:
            print(f"\n{Fore.YELLOW}Widget Configuration Issues:{Style.RESET_ALL}")
            for i, issue in enumerate(widget_issues):
                print(f"{i+1}. {issue}")
                
        if registry_connection_issues:
            print(f"\n{Fore.YELLOW}Registry Connection Issues:{Style.RESET_ALL}")
            for i, issue in enumerate(registry_connection_issues):
                print(f"{len(widget_issues)+i+1}. {issue}")


if __name__ == "__main__":
    # Initialize colorama for cross-platform color support
    init()
    
    parser = argparse.ArgumentParser(description='Monitor the PlotRegistry system and validate workflows')
    parser.add_argument('--watch-signal', type=str, help='Monitor a specific signal ID')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')
    parser.add_argument('--reset', action='store_true', help='Reset the registry before monitoring')
    parser.add_argument('--validate-workflow', type=str, help='Path to workflow JSON file to validate')
    parser.add_argument('--validate-all', action='store_true', help='Validate all workflows in the workflows directory')
    
    args = parser.parse_args()
    
    # Handle workflow validation
    if args.validate_workflow:
        validate_workflow(args.validate_workflow)
    elif args.validate_all:
        # Find the workflows directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        workflows_dir = os.path.join(parent_dir, 'workflows')
        
        if os.path.exists(workflows_dir) and os.path.isdir(workflows_dir):
            print(f"{Fore.CYAN}Validating all workflows in {workflows_dir}{Style.RESET_ALL}")
            for filename in os.listdir(workflows_dir):
                if filename.endswith('.json'):
                    workflow_path = os.path.join(workflows_dir, filename)
                    validate_workflow(workflow_path)
        else:
            print(f"{Fore.RED}Workflows directory not found at {workflows_dir}{Style.RESET_ALL}")
    else:
        # Default: monitor registry
        monitor_registry(interval=args.interval, watch_signal=args.watch_signal, reset=args.reset)
