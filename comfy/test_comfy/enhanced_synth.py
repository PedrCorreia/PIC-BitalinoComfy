import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)

from src.utils.synthetic_data import SyntheticDataGenerator 
from src.plot.plot import PygamePlot

def test_enhanced_real_time_plot():
    """Test the enhanced real-time plotting with dynamic x-axis"""
    print("Starting enhanced real-time plot test")
    
    # Initialize the generator with test settings
    generator = SyntheticDataGenerator()
    generator.signal_type = "RR"  # Use respiratory rate for testing
    generator.sampling_rate = 100
    generator.duration = 20  # 20 seconds signal
    generator.buffer_size = 5  # 5 second plotting window
    generator.fps = 60
    
    # Start generating data
    generator._ensure_thread("RR", 20, 100, 5, auto_restart=True, keep_window=True)
    
    # Create a plot instance
    plot = PygamePlot()
    plot.sampling_rate = 100
    plot.FPS = 60
    
    # Register with generator for updates
    generator._plot_nodes.add(plot)
    
    # Start update thread
    if generator.plot_thread is None or not generator.plot_thread.is_alive():
        generator.plot_thread = threading.Thread(
            target=generator._continuous_plot_updater,
            daemon=True
        )
        generator.plot_thread.start()
    
    print("Press Ctrl+C to exit")
    try:
        # Run for the duration of the signal plus a buffer
        start_time = time.time()
        while time.time() - start_time < generator.duration + 2:
            # Just wait and let the threads do their work
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Test interrupted.")
    finally:
        # Clean up
        generator.running = False
        if generator.thread and generator.thread.is_alive():
            generator.thread.join(timeout=0.2)
        print("Test complete.")

if __name__ == "__main__":
    import threading
    test_enhanced_real_time_plot()
