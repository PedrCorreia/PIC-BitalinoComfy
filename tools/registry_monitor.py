#!/usr/bin/env python
"""
Registry Monitor - A tool to monitor the PlotRegistry in real-time
"""
import sys
import os
import time
import argparse
import threading
from tabulate import tabulate

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.registry.plot_registry import PlotRegistry
    import numpy as np
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to run this script from the PIC-2025 directory")
    sys.exit(1)

def pretty_format_value(value):
    """Format a value for display"""
    if isinstance(value, (int, float)):
        return value
    elif hasattr(value, 'shape'):
        return f"<shape: {value.shape}>"
    elif hasattr(value, '__len__'):
        return f"<len: {len(value)}>"
    else:
        return str(type(value)).__name__

def monitor_registry(interval=1.0, max_rows=10, watch_signal=None, save_logs=False):
    """Monitor the registry in real-time"""
    registry = PlotRegistry.get_instance()
    
    # Set up logging
    log_file = None
    if save_logs:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, f'registry_monitor_{time.strftime("%Y%m%d_%H%M%S")}.log'), 'w')
        print(f"Saving logs to {log_file.name}")

    try:
        print("\n=== Registry Monitor Started ===")
        print(f"Refresh interval: {interval} second{'s' if interval != 1 else ''}")
        print(f"Press Ctrl+C to exit\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            
            # Clear screen (os-agnostic)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n=== Registry Status (Refresh #{iteration}) ===")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            
            # Get all signals
            with registry.registry_lock:
                signals = registry.signals
                metadata = registry.metadata
                connections = registry.connections
                visualized = registry.visualized_signals
            
            # General stats
            print("\nGeneral Statistics:")
            stats = [
                ["Signals", len(signals)],
                ["Connected Nodes", registry.connected_nodes],
                ["Visualized Signals", len(visualized)],
                ["Registry Age", f"{time.time() - registry.creation_time:.1f}s"]
            ]
            print(tabulate(stats, tablefmt="simple"))
            
            # Signals table
            if signals:
                print("\nSignals:")
                table_data = []
                
                # Sort signal IDs
                signal_ids = list(signals.keys())
                if watch_signal and watch_signal in signal_ids:
                    # Put watched signal first
                    signal_ids.remove(watch_signal)
                    signal_ids.insert(0, watch_signal)
                
                # Limit to max_rows
                if len(signal_ids) > max_rows:
                    display_ids = signal_ids[:max_rows]
                    remaining = len(signal_ids) - max_rows
                else:
                    display_ids = signal_ids
                    remaining = 0
                
                for signal_id in display_ids:
                    signal_data = signals[signal_id]
                    meta = metadata.get(signal_id, {})
                    
                    # Get basic info
                    shape = signal_data.shape if hasattr(signal_data, 'shape') else len(signal_data) if hasattr(signal_data, '__len__') else '?'
                    data_type = type(signal_data).__name__
                    
                    # Get connected nodes
                    connected_nodes = sum(1 for node_id, sigs in connections.items() if signal_id in sigs)
                    
                    # Get min/max if applicable
                    if hasattr(signal_data, 'min') and hasattr(signal_data, 'max'):
                        try:
                            min_val = signal_data.min()
                            max_val = signal_data.max()
                            value_range = f"{min_val:.2f} to {max_val:.2f}"
                        except:
                            value_range = "N/A"
                    else:
                        value_range = "N/A"
                    
                    # Is visualized?
                    is_visualized = "✓" if signal_id in visualized else "-"
                    
                    # Color if present
                    color = str(meta.get('color', 'None'))
                    
                    # Highlight watched signal
                    if watch_signal and signal_id == watch_signal:
                        signal_id = f">> {signal_id} <<"
                    
                    row = [signal_id, shape, data_type, value_range, connected_nodes, is_visualized, color]
                    table_data.append(row)
                
                print(tabulate(
                    table_data, 
                    headers=["Signal ID", "Shape", "Type", "Range", "Connections", "Visualized", "Color"],
                    tablefmt="simple"
                ))
                
                if remaining > 0:
                    print(f"\n...and {remaining} more signals (use --max-rows to show more)")
            else:
                print("\nNo signals in registry.")
            
            # Connections
            if connections:
                print("\nNode Connections:")
                conn_data = []
                
                for node_id, signal_ids in list(connections.items())[:max_rows]:
                    signals_str = ", ".join(list(signal_ids)[:3])
                    if len(signal_ids) > 3:
                        signals_str += f", ... (+{len(signal_ids) - 3} more)"
                    conn_data.append([node_id, len(signal_ids), signals_str])
                
                print(tabulate(
                    conn_data,
                    headers=["Node ID", "Signal Count", "Connected Signals"],
                    tablefmt="simple"
                ))
                
                if len(connections) > max_rows:
                    print(f"\n...and {len(connections) - max_rows} more connections")
            else:
                print("\nNo node connections.")
            
            # Signal Details (if watching a specific signal)
            if watch_signal and watch_signal in signals:
                print(f"\n=== Details for Signal: {watch_signal} ===")
                signal_data = signals[watch_signal]
                meta = metadata.get(watch_signal, {})
                
                # Basic properties
                details_data = []
                
                # Type and shape
                details_data.append(["Type", type(signal_data).__name__])
                if hasattr(signal_data, 'shape'):
                    details_data.append(["Shape", signal_data.shape])
                elif hasattr(signal_data, '__len__'):
                    details_data.append(["Length", len(signal_data)])
                
                # Data statistics if numerical
                if hasattr(signal_data, 'min') and hasattr(signal_data, 'max'):
                    try:
                        details_data.append(["Min Value", f"{signal_data.min():.4f}"])
                        details_data.append(["Max Value", f"{signal_data.max():.4f}"])
                        if hasattr(signal_data, 'mean'):
                            details_data.append(["Mean", f"{signal_data.mean():.4f}"])
                        if hasattr(signal_data, 'std'):
                            details_data.append(["Std Dev", f"{signal_data.std():.4f}"])
                    except:
                        pass
                
                # Metadata
                for key, value in meta.items():
                    details_data.append([f"Meta: {key}", pretty_format_value(value)])
                
                # Connected nodes
                connected_nodes = [node_id for node_id, sigs in connections.items() if watch_signal in sigs]
                if connected_nodes:
                    details_data.append(["Connected Nodes", ", ".join(connected_nodes[:5]) + (", ..." if len(connected_nodes) > 5 else "")])
                
                # Visualization status
                details_data.append(["Visualized", "Yes" if watch_signal in visualized else "No"])
                
                print(tabulate(details_data, tablefmt="simple"))
                
                # Show first and last few values if numeric array
                try:
                    if isinstance(signal_data, (np.ndarray,)) and signal_data.size > 0:
                        print("\nSample Values:")
                        head_values = signal_data[:5].flatten()
                        tail_values = signal_data[-5:].flatten() if signal_data.size > 5 else []
                        
                        head_str = " ".join([f"{x:.2f}" for x in head_values])
                        if len(tail_values) > 0 and signal_data.size > 10:
                            print(f"First 5: [{head_str} ...]")
                            tail_str = " ".join([f"{x:.2f}" for x in tail_values])
                            print(f"Last 5:  [... {tail_str}]")
                        else:
                            print(f"Values: [{head_str}]")
                except:
                    pass
            
            # Log to file if enabled
            if log_file:
                log_file.write(f"\n=== Registry Status at {time.strftime('%H:%M:%S')} (Iteration {iteration}) ===\n")
                log_file.write(f"Total Signals: {len(signals)}, Connected Nodes: {registry.connected_nodes}\n")
                if signals:
                    log_file.write(f"Signal IDs: {', '.join(signals.keys())}\n")
                log_file.flush()
            
            # Sleep for the interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor the PlotRegistry in real-time")
    parser.add_argument("--interval", "-i", type=float, default=1.0, 
                        help="Refresh interval in seconds (default: 1.0)")
    parser.add_argument("--max-rows", "-m", type=int, default=10,
                        help="Maximum rows to display for each section (default: 10)")
    parser.add_argument("--watch", "-w", type=str, default=None,
                        help="Watch a specific signal ID")
    parser.add_argument("--log", "-l", action="store_true",
                        help="Save monitoring logs to file")
    
    args = parser.parse_args()
    
    # Run the monitor
    monitor_registry(
        interval=args.interval,
        max_rows=args.max_rows,
        watch_signal=args.watch,
        save_logs=args.log
    )
