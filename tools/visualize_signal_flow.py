#!/usr/bin/env python
"""
Signal Flow Visualizer - A tool to visualize signal flow through the registry system
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.registry.plot_registry import PlotRegistry
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to run this script from the PIC-2025 directory")
    sys.exit(1)

class SignalFlowVisualizer:
    """
    A class for visualizing signal flow through the registry system
    """
    
    def __init__(self, signal_ids=None, interval=100, history_length=100):
        """
        Initialize the visualizer
        
        Args:
            signal_ids: List of signal IDs to watch (None means all signals)
            interval: Update interval in milliseconds
            history_length: Number of data points to keep in history
        """
        self.signal_ids = signal_ids
        self.interval = interval
        self.history_length = history_length
        
        # Get the registry
        self.registry = PlotRegistry.get_instance()
        
        # Dictionary to store signal histories
        self.signal_histories = {}
        self.fig = None
        self.axes = None
        self.lines = {}
        
        # Last values for each signal
        self.last_values = {}
    
    def _extract_plot_value(self, data):
        """Extract a single value to plot from signal data"""
        try:
            if isinstance(data, (int, float)):
                return data
            
            # If it's an array or list
            if hasattr(data, '__len__'):
                if hasattr(data, 'mean'):
                    # Use mean for numerical arrays
                    return data.mean()
                elif len(data) > 0:
                    # Use first element for lists
                    return float(data[0])
            
            # Fallback
            return 0.0
        except:
            return 0.0
    
    def update_plot(self, frame):
        """Update function for animation"""
        # Get all signals if no specific ones are provided
        if self.signal_ids is None:
            with self.registry.registry_lock:
                self.signal_ids = list(self.registry.signals.keys())
        
        # Initialize histories for new signals
        for signal_id in self.signal_ids:
            if signal_id not in self.signal_histories:
                self.signal_histories[signal_id] = np.zeros(self.history_length)
                self.last_values[signal_id] = 0.0
                
                # Create a line if the plot is already set up
                if self.axes is not None:
                    line, = self.axes.plot([], [], label=signal_id)
                    self.lines[signal_id] = line
        
        # Update signal histories
        for signal_id in self.signal_ids:
            # Get signal data
            signal_data = self.registry.get_signal(signal_id)
            
            if signal_data is not None:
                # Extract a value to plot
                value = self._extract_plot_value(signal_data)
                
                # Store the value
                self.last_values[signal_id] = value
            else:
                # Use the last value if signal is not available
                value = self.last_values.get(signal_id, 0.0)
            
            # Update history
            self.signal_histories[signal_id] = np.roll(self.signal_histories[signal_id], -1)
            self.signal_histories[signal_id][-1] = value
            
            # Update the line
            if signal_id in self.lines:
                self.lines[signal_id].set_data(range(self.history_length), self.signal_histories[signal_id])
        
        # Remove signals that no longer exist
        existing_signals = set(self.registry.signals.keys())
        signals_to_remove = set(self.signal_histories.keys()) - existing_signals
        
        for signal_id in signals_to_remove:
            if signal_id in self.signal_histories:
                del self.signal_histories[signal_id]
            if signal_id in self.lines:
                self.lines[signal_id].remove()
                del self.lines[signal_id]
        
        # Update axis limits if we have any signals
        if self.signal_histories:
            all_values = np.concatenate(list(self.signal_histories.values()))
            if len(all_values) > 0:
                min_val = min(np.min(all_values), -0.1)
                max_val = max(np.max(all_values), 0.1)
                padding = (max_val - min_val) * 0.1
                self.axes.set_ylim(min_val - padding, max_val + padding)
        
        # Update legend if needed
        if len(self.lines) != len(self.axes.get_legend().get_texts()):
            self.axes.legend()
        
        # Update title with statistics
        title = f"Signal Flow Monitor - {len(self.signal_histories)} signals"
        title += f" | Last update: {time.strftime('%H:%M:%S')}"
        self.axes.set_title(title)
        
        return list(self.lines.values())
    
    def visualize(self):
        """Start the visualization"""
        # Create the figure and axes
        self.fig, self.axes = plt.subplots(figsize=(12, 6))
        self.axes.set_xlim(0, self.history_length)
        self.axes.set_ylim(-1, 1)
        self.axes.grid(True)
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Signal Value')
        self.axes.set_title('Signal Flow Monitor')
        
        # Initialize empty lines for each signal
        self.lines = {}
        
        # Create animation
        ani = FuncAnimation(
            self.fig, self.update_plot, interval=self.interval, 
            blit=True, save_count=self.history_length
        )
        
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize signal flow through the registry system')
    parser.add_argument('--signals', '-s', nargs='+', help='Signal IDs to watch (empty means all signals)')
    parser.add_argument('--interval', '-i', type=int, default=100, help='Update interval in milliseconds (default: 100)')
    parser.add_argument('--history', '-l', type=int, default=100, help='History length (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Create and run visualizer
        visualizer = SignalFlowVisualizer(
            signal_ids=args.signals,
            interval=args.interval,
            history_length=args.history
        )
        
        visualizer.visualize()
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
