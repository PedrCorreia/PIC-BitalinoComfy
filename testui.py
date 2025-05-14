# test_plot.py
import numpy as np
import time
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the plot unit
from src.plot import PlotUnit

def main():
    # Get plot unit instance
    plot = PlotUnit.get_instance()
    plot.start()
    
    # Generate and visualize some test data
    for i in range(100):
        # Create sine wave with increasing frequency
        t = np.linspace(0, 10, 200)
        freq = 0.1 + (i * 0.002)
        signal = np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, 200)
        signal = signal + noise
        
        # Queue for visualization
        plot.queue_data('test_signal', signal)
        
        # Create a filtered version
        processed = signal.copy()
        # Simple moving average filter
        window_size = 5
        processed = np.convolve(processed, np.ones(window_size)/window_size, mode='same')
        
        # Queue processed signal
        plot.queue_data('test_signal_processed', processed)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()