"""
Example script demonstrating usage of the restructured plot components.

This script shows how to use the restructured plot components with the
new signal processing organization.
"""

import numpy as np
import time
import sys
import os

# Add the parent directory to sys.path to allow imports from the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plot import PlotUnit
from src.plot.utils import SignalHistoryManager, SignalProcessingAdapter
from src.utils.signal_processing import NumpySignalProcessor

def generate_example_signal(length=100, frequency=0.1):
    """
    Generate an example signal for demonstration.
    
    Args:
        length (int): Length of the signal
        frequency (float): Frequency of the sine wave
        
    Returns:
        numpy.ndarray: Generated signal
    """
    t = np.linspace(0, 10, length)
    signal = np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 0.2, length)
    return signal + noise

def main():
    """
    Main function demonstrating usage of restructured components.
    """
    # Get the plot unit instance
    plot = PlotUnit.get_instance()
    
    # Start visualization
    plot.start()
    
    # Create a signal history manager for local processing
    history_manager = SignalHistoryManager()
    
    # Main loop
    try:
        for i in range(100):
            # Generate a signal
            raw_signal = generate_example_signal()
            
            # Queue the raw signal for visualization
            plot.queue_data('example_signal', raw_signal)
            
            # Update history for local processing
            history_manager.update_history('example_signal', raw_signal)
            
            # Get signal history
            history = history_manager.get_history('example_signal')
            
            # Process the signal using the adapter
            filtered_signal = SignalProcessingAdapter.process_signal(
                history,
                processing_type='lowpass',
                cutoff=0.2,
                fs=10,
                order=2
            )
            
            # Queue the processed signal for visualization
            plot.queue_data('example_signal_processed', filtered_signal)
            
            # Alternatively, process directly using NumpySignalProcessor
            # This shows the direct use of the core signal processing
            bandpass_signal = NumpySignalProcessor.bandpass_filter(
                history,
                lowcut=0.05,
                highcut=0.15,
                fs=10,
                order=2
            )
            
            # Queue the bandpass signal for visualization
            plot.queue_data('example_signal_bandpass', bandpass_signal)
            
            # Sleep to simulate processing time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Exiting...")
    
    # Allow some time for visualization to finish
    time.sleep(1)
    
if __name__ == "__main__":
    main()
