import time
import numpy as np
from ..src.plot import Plot

class PlotNode:
    """
    Node for plotting signal data with selectable signal processing backend (numpy, torch, cuda).
    Uses the Plot utility for visualization.

    - The 'peaks' input is always received (list of peak indices or mask), but only plotted if 'show_peaks' is True.
    - If the signal data is (timestamp, value, is_peak), the third column is treated as a peak marker.
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
                "data": ("LIST",),  # Input deque of (timestamp, value) or (timestamp, value, is_peak) tuples
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

                "fixed_y_min": ("FLOAT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "fixed_y_max": ("FLOAT", {
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


            },
        }

    def plot(self, data, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, grid=False, auto_scale=True, fixed_y_min=0, fixed_y_max=100, window_title=DEFAULT_WINDOW_TITLE, show_peaks=False, peaks=[], process_signal=False, peak_detection=False, peak_height=0, peak_distance=1, signal_processor_type="numpy"):
        """
        Plots the provided signal data using the Plot class from plot.py.

        Parameters:
        - data: List of up to 4 deques of (timestamp, value) or (timestamp, value, is_peak) tuples.
        - peaks: List of peak indices or mask, always received, only plotted if show_peaks is True.
        """
        signals = []
        peaks_masks = []
        for signal in data:
            arr = np.array(signal)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                signals.append(arr[:, :2])
                # If third column exists, treat as is_peak mask
                if arr.shape[1] >= 3:
                    peaks_masks.append(arr[:, 2])
                else:
                    peaks_masks.append(None)
            else:
                signals.append(arr)
                peaks_masks.append(None)

        # Use the Plot class from plot.py
        if self.plotter is None:
            self.plotter = Plot(fs=1000, duration=max(arr[-1, 0] for arr in signals if len(arr) > 0), live=False,show_peaks=show_peaks)
        

        return ()
