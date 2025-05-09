import time
import numpy as np
from ..src.plot import Plot
import cv2  # Import OpenCV
from ..src.utils import weighted_hr, weighted_rr


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
        self.worker = None

    @classmethod
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ARRAY",),  # Input deque of (timestamp, value) or (timestamp, value, is_peak) tuples
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
                "mode": ("STRING", {
                    "default": "static",
                    "choices": ["static"],
                    "display": "dropdown"
                }),
                "buffer_size": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "display": "number"
                }),
            },
            "optional": {
                "data2": ("ARRAY",),
                "data3": ("ARRAY",),
                "data4": ("ARRAY",),
            }
        }

    def plot(
            self,
            data,
            data2=None,
            data3=None,
            data4=None,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            fixed_y_min=0,
            fixed_y_max=100,
            window_title=DEFAULT_WINDOW_TITLE,
            show_peaks=False,
            mode="static",  # Default to static mode
            buffer_size=300,
            ):
        """
        Plots the provided signal data using the Plot class from plot.py and embeds it into an OpenCV window.

        Parameters:
        - data: List of up to 4 deques of (timestamp, value) or (timestamp, value, is_peak) tuples.
        - mode: Only "static" mode is supported.
        """
        #print("plot Initialized")
        signals = []
        for signal in [data, data2, data3, data4]:
            if signal is not None:
                arr = np.array(signal)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    arr = arr[-buffer_size:]  # Keep only last buffer_size points
                    signals.append(arr[:, :3] if arr.shape[1] >= 3 else arr[:, :2])
                elif arr.size > 0:
                    arr = arr[-buffer_size:]
                    signals.append(arr)
        if not signals:
            print("No valid signals to plot.")
            return ()

        # Use the optimized static_opencv_plot (thinner lines, auto y-scale)
        if self.plotter is None:
            self.plotter = Plot()
        self.plotter.static_opencv_plot(
            signals,
            show_peaks=show_peaks,
            window_title=window_title,
            height=int(height),
            width=int(width),
            fixed_y_min=None,  # auto y-scale
            fixed_y_max=None   # auto y-scale
        )
        #print(f"Static plot rendered at: {time.time()}")
        return ()


    @staticmethod
    def static_opencv_plot_fast(
        signals,
        show_peaks=False,  # ignored for speed
        window_title="Static OpenCV Plot",
        height=None,
        width=None,
        fixed_y_min=None,
        fixed_y_max=None
    ):
        # FPS cap
        if not hasattr(Plot, "_last_plot_time"):
            Plot._last_plot_time = 0
        min_interval = 1.0 / 15  # 15 FPS
        now = time.time()
        elapsed = now - Plot._last_plot_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        Plot._last_plot_time = time.time()

        n_signals = len([arr for arr in signals if arr.shape[0] > 0])
        if n_signals == 0:
            return
        plot_height_per_signal = 200
        plot_width = width if width is not None else 800
        left_margin = 40
        right_margin = 0
        top_margin = 0
        bottom_margin = 30
        img_width = plot_width + left_margin + right_margin
        img_height = plot_height_per_signal * n_signals + top_margin + bottom_margin
        img = np.full((img_height, img_width, 3), 30, dtype=np.uint8)
        colors = np.array([
            [0, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [255, 0, 255]
        ], dtype=np.uint8)
        # Compute global x range for all signals
        valid_signals = [arr for arr in signals if arr.shape[0] > 0 and arr.shape[1] >= 2]
        min_ts = min(arr[:, 0].min() for arr in valid_signals)
        max_ts = max(arr[:, 0].max() for arr in valid_signals)
        x_min = min_ts
        x_max = max_ts if max_ts > min_ts else min_ts + 1.0

        for idx, arr in enumerate(signals):
            if arr.shape[1] < 2 or arr.shape[0] < 2:
                continue
            # Dynamic y: show at least 5 above and below window min/max
            y_min = arr[:, 1].min()
            y_max = arr[:, 1].max()
            y_min_plot = y_min - 5
            y_max_plot = y_max + 5
            if y_max_plot == y_min_plot:
                y_max_plot += 1.0
            y0 = top_margin + idx * plot_height_per_signal
            y1 = top_margin + (idx + 1) * plot_height_per_signal - 1
            x0 = left_margin
            x1 = left_margin + plot_width

            # Draw axes
            cv2.line(img, (x0, y1), (x1, y1), (200, 200, 200), 1)  # x-axis
            cv2.line(img, (x0, y1), (x0, y0), (200, 200, 200), 1)  # y-axis

            # X ticks/labels (time)
            num_xticks = 6
            xtick_fracs = np.linspace(0, 1, num_xticks + 1)
            xtick_vals = x_min + xtick_fracs * (x_max - x_min)
            xtick_px = (x0 + xtick_fracs * plot_width).astype(int)
            for px, x_val in zip(xtick_px, xtick_vals):
                cv2.line(img, (px, y1), (px, y1 + 6), (220, 220, 220), 1)
                if idx == n_signals - 1:
                    label = f"{x_val:.1f}"
                    cv2.putText(img, label, (px - 12, img_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

            # Map data to pixel coordinates (vectorized)
            timestamps = arr[:, 0]
            values = arr[:, 1]
            x_pix = np.interp(timestamps, (x_min, x_max), (x0, x1)).astype(np.int32)
            y_pix = np.interp(values, (y_min_plot, y_max_plot), (y1, y0)).astype(np.int32)
            color = tuple(int(c) for c in colors[idx % len(colors)])
            pts = np.stack([x_pix, y_pix], axis=1).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_4)

        cv2.imshow(window_title, img)
        cv2.waitKey(1)

    # Patch the static_opencv_plot method for maximum speed
    Plot.static_opencv_plot = staticmethod(static_opencv_plot_fast)


class PhysioNormalizeNode:
    """
    Node to compute weighted/normalized values for HR, RR, SCR, and arousal.
    Each output can be enabled/disabled via a boolean input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hr_signal": ("DEQUE",),
                "rr_signal": ("DEQUE",),
                "scr_signal": ("DEQUE",),
                "arousal_signal": ("DEQUE",),
                "calc_hr": ("BOOLEAN", {"default": True}),
                "calc_rr": ("BOOLEAN", {"default": True}),
                "calc_scr": ("BOOLEAN", {"default": True}),
                "calc_arousal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("hr_weighted", "rr_weighted", "scr_weighted", "arousal_weighted")
    FUNCTION = "normalize"
    CATEGORY = "Custom/Physio"

    def normalize(
        self,
        hr_signal,
        rr_signal,
        scr_signal,
        arousal_signal,
        calc_hr=True,
        calc_rr=True,
        calc_scr=True,
        calc_arousal=False,
    ):
        hr_w = weighted_hr(hr_signal) if calc_hr else float("nan")
        rr_w = weighted_rr(rr_signal) if calc_rr else float("nan")
        return (hr_w, rr_w, scr_w, arousal_w)
