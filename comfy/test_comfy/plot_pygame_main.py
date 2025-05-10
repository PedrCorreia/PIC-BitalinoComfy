import time
import numpy as np
from synthetic_data import SyntheticDataNode
from plot_node import PygamePlotNode  # Import your node

if __name__ == "__main__":
    node = SyntheticDataNode()
    plot_node = PygamePlotNode()  # Instantiate your node
    signal_type = "RR"
    duration = 5
    sampling_rate = 100
    as_points = False

    print("Press Ctrl+C or close the window to exit.")
    try:
        while True:
            x, y = node.generate(signal_type, duration, sampling_rate)
            
            # Use the node's plot method (mimic ComfyUI)
            #print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S"))
            plot_node.plot(x, y, as_points)
            time.sleep(0.001)  # Reasonable interval for UI update

    except KeyboardInterrupt:
        print("Exiting.")
