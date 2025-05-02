import time
import numpy as np
from ..src.plot import PlotWindow
from ..src.signalprocessing import SignalProcessing  # Import SignalProcessing

class PlotNode:
    DEFAULT_HEIGHT = 400
    DEFAULT_WIDTH = 700
    DEFAULT_WINDOW_TITLE = "Real-Time Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True
    CATEGORY = "Custom/Visualization"

    def __init__(self):
        self.plot_window = None

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
            },
        }

    def plot(self, data, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, grid=False, auto_scale=True, fixed_y_min=0, fixed_y_max=100, window_title=DEFAULT_WINDOW_TITLE, show_peaks=False, peaks=[], process_signal=False, peak_detection=False, peak_height=0, peak_distance=1, show_stft=False, stft_window_size=256, stft_overlap=128):
        if self.plot_window is None:
            self.plot_window = PlotWindow(width=int(width), height=int(height), grid=grid)
            self.plot_window.show(window_name=window_title)

        # Process the signal if enabled
        if process_signal:
            signal_processor = SignalProcessing(signal=data)
            if peak_detection:
                peaks = signal_processor.detect_peaks(height=peak_height, distance=peak_distance)
                show_peaks = True

            # Compute STFT if enabled
            if show_stft:
                f, t, magnitudes = signal_processor.compute_stft(window_size=stft_window_size, overlap=stft_overlap)
                self.plot_window.set_stft_data((f, t, magnitudes))

        # Set scale mode and fixed Y-axis values
        self.plot_window.auto_scale = auto_scale
        self.plot_window.fixed_y_min = fixed_y_min
        self.plot_window.fixed_y_max = fixed_y_max

        # Update the plot with the entire buffer
        self.plot_window.data.clear()
        self.plot_window.data.extend(data)
        self.plot_window.update(window_name=window_title)
        self.plot_window.show_peaks = show_peaks
        self.plot_window.set_peaks(peaks)
        self.plot_window.show_stft = show_stft
        return ()
