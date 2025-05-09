import time
import numpy as np
from ..src.plot import Plot

class PlotNode:
    """
    Node for plotting signal data with selectable signal processing backend (numpy, torch, cuda).
    Uses the Plot utility for visualization.
    """
    DEFAULT_HEIGHT = 400
    DEFAULT_WIDTH = 700
    DEFAULT_WINDOW_TITLE = "Real-Time Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True
    CATEGORY = "Custom/Visualization"

    def __init__(self):
        self.plotter = None

    @classmethod
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("DEQUE",),  # Input deque of (timestamp, value) pairs
                "height": ("FLOAT", {
                    "default": cls.DEFAULT_HEIGHT,
                    "min": 100,
                    "max": 1000,
                    "step": 50,
                    "display": "number"
                }),
                "width": ("FLOAT", {
                    "default": cls.DEFAULT_WIDTH,
                    "min": 100,
                    "max": 1000,
                    "step": 50,
                    "display": "number"
                }),
                "grid": ("BOOLEAN", {
                    "default": False,
                }),
                "auto_scale": ("BOOLEAN", {  # New input for scale mode
                    "default": True,
                }),
                "fixed_y_min": ("FLOAT", {  # Minimum Y-axis value for manual scale
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "fixed_y_max": ("FLOAT", {  # Maximum Y-axis value for manual scale
                    "default": 100,
                    "min": -1000,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "window_title": ("STRING", {
                    "default": cls.DEFAULT_WINDOW_TITLE,
                }),
                "show_peaks": ("BOOLEAN", {
                    "default": False,
                }),
                "peaks": ("LIST", {  # Input list of peak indices
                    "default": [],
                }),
                "process_signal": ("BOOLEAN", {  # Option to process the signal
                    "default": False,
                }),
                "peak_detection": ("BOOLEAN", {  # Option to detect peaks
                    "default": False,
                }),
                "peak_height": ("FLOAT", {  # Minimum height for peak detection
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "peak_distance": ("FLOAT", {  # Minimum distance between peaks
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "show_stft": ("BOOLEAN", {  # Option to show STFT
                    "default": False,
                }),
                "stft_window_size": ("INT", {  # Window size for STFT
                    "default": 256,
                    "min": 16,
                    "max": 1024,
                    "step": 16,
                }),
                "stft_overlap": ("INT", {  # Overlap for STFT
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 16,
                }),
                "signal_processor_type": ("STRING", {  # Choice of backend
                    "default": "numpy",
                    "choices": ["numpy", "torch", "cuda"]
                }),
            },
        }

    def plot(self, data, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, grid=False, auto_scale=True, fixed_y_min=0, fixed_y_max=100, window_title=DEFAULT_WINDOW_TITLE, show_peaks=False, peaks=[], process_signal=False, peak_detection=False, peak_height=0, peak_distance=1, show_stft=False, stft_window_size=256, stft_overlap=128, signal_processor_type="numpy"):
        """
        Plots the provided signal data using the Plot class from plot.py.

        Parameters:
        - data: List of up to 4 deques of (timestamp, value) or (timestamp, value, is_peak) pairs.
        """
        signals = []
        for signal in data:
            arr = np.array(signal)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                signals.append(arr)

        # Use the Plot class from plot.py
        if self.plotter is None:
            self.plotter = Plot(fs=1000, duration=max(arr[-1, 0] for arr in signals if len(arr) > 0), live=False)
        self.plotter.plot(signals)

        return ()
