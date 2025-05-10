import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPainter
from collections import deque
import cv2
import time
import torch
from PIL import Image as PILImage
import lunar_tools as lt

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_WINDOW_TITLE = "Render"
# You may need to import your renderer, e.g.:
# import lt

class Plot:
    """
    Plotting utility class for live and static signal visualization.
    """

    def __init__(self, fs=1000, duration=5, live=False, suffix="", window=None, show_peaks=False):
        self.fs = fs
        self.duration = duration
        self.live = live
        self.suffix = suffix  # Suffix for channel labels
        self.window = window if window is not None else duration  # Window size for live mode
        self.show_peaks = show_peaks

        # Create a GraphicsLayoutWidget for plotting
        self.widget = pg.GraphicsLayoutWidget(show=False)  # Set `show=False` to avoid displaying the widget

    def plot(self, signals, show_peaks=False, **kwargs):
        """
        Plot signals using the selected mode (live or static).
        Accepts 1, 2, 3, or 4 signals (as list or single array/callable).
        If show_peaks is True, plot points where is_peak==1 (third column).
        """
        # Ensure signals is a list of 1-4 items
        if not isinstance(signals, (list, tuple)):
            signals = [signals]
        if len(signals) > 4:
            signals = signals[:4]
        if self.live:
            self.live_pyqtgraph(signals, show_peaks=show_peaks)
        else:
            self.static_pyqtgraph(signals, show_peaks=show_peaks)

    def live_pyqtgraph(self, signals, show_peaks=False):
        """
        Live plot multiple signals using PyQtGraph, updating in real time.
        """
        app = QApplication.instance() or QApplication([])
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)
        plots = []
        curves = []
        peak_markers = []
        self._last_plotted_ts = [None] * n_channels

        for i in range(n_channels):
            p = win.addPlot(row=i, col=0)
            label = f'Ch {i+1}'
            if self.suffix:
                label += f' {self.suffix}'
            p.setLabel('left', label)
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (s)')
            curve = p.plot(pen=pg.mkPen(width=2))
            plots.append(p)
            curves.append(curve)
            peak_markers.append(pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0)))
            p.addItem(peak_markers[-1])

        def update():
            for i, sig in enumerate(signals):
                data = list(sig)
                if not data or len(data) == 0:
                    curves[i].setData([], [])
                    peak_markers[i].setData([], [])
                    self._last_plotted_ts[i] = None
                    continue
                arr = np.array(data)
                if arr.ndim < 2 or arr.shape[1] < 2:
                    curves[i].setData([], [])
                    peak_markers[i].setData([], [])
                    self._last_plotted_ts[i] = None
                    continue

                t_vals = arr[:, 0]
                y_vals = arr[:, 1]
                peaks = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(y_vals)

                window = self.window
                max_time = t_vals[-1]
                min_time = max_time - window
                in_window = t_vals >= min_time
                t_vals = t_vals[in_window]
                y_vals = y_vals[in_window]
                peaks = peaks[in_window]

                if self._last_plotted_ts[i] is not None:
                    idx = np.searchsorted(t_vals, self._last_plotted_ts[i], side='right')
                    t_vals = t_vals[idx:]
                    y_vals = y_vals[idx:]
                    peaks = peaks[idx:]

                curves[i].setData(t_vals, y_vals)
                if show_peaks:
                    peak_markers[i].setData(t_vals[peaks == 1], y_vals[peaks == 1])
                else:
                    peak_markers[i].setData([], [])
                self._last_plotted_ts[i] = t_vals[-1] if len(t_vals) > 0 else None

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)
        app.exec()

    def live_opencv_plot(self, signals, window_title="Live OpenCV Plot", buffer_size=300, window_size=300, height=300, width=600):
        """
        Live plot up to 3 channels using OpenCV, with moving x-axis and axes/labels.
        Each signal should be an iterable of (timestamp, value).
        buffer_size: max number of points kept in memory per channel.
        window_size: number of most recent points shown in the plot window.
        """
        n_channels = min(len(signals), 3)
        colors = [(0,255,0), (255,0,0), (0,255,255)]
        buffer_size = int(buffer_size)
        window_size = int(window_size)
        height = int(height)
        width = int(width)
        left_margin = 60
        right_margin = 20
        top_margin = 20
        bottom_margin = 40
        img_width = width + left_margin + right_margin
        img_height = height + top_margin + bottom_margin

        # Remove buffer logic for lower latency: always plot the latest window_size points from the signal
        while True:
            img = np.full((img_height, img_width, 3), 30, dtype=np.uint8)
            x0 = left_margin
            x1 = left_margin + width
            y0 = img_height - bottom_margin
            y1 = top_margin

            cv2.line(img, (x0, y0), (x1, y0), (200,200,200), 2)
            cv2.line(img, (x0, y0), (x0, y1), (200,200,200), 2)

            visible_bufs = []
            for sig in signals[:n_channels]:
                arr = np.asarray(sig)
                if arr.shape[0] > 0 and arr.shape[1] >= 2:
                    arr = arr[-window_size:]
                    visible_bufs.append(arr)
                else:
                    visible_bufs.append(np.zeros((0,2)))

            all_t = np.concatenate([arr[:,0] for arr in visible_bufs if arr.shape[0] > 0]) if any(arr.shape[0]>0 for arr in visible_bufs) else np.array([0,1])
            all_y = np.concatenate([arr[:,1] for arr in visible_bufs if arr.shape[0] > 0]) if any(arr.shape[0]>0 for arr in visible_bufs) else np.array([0,1])
            if all_t.size < 2:
                all_t = np.array([0,1])
            if all_y.size < 2:
                all_y = np.array([0,1])
            t_min, t_max = all_t.min(), all_t.max()
            y_min, y_max = all_y.min(), all_y.max()
            if t_max == t_min:
                t_max += 1
            if y_max == y_min:
                y_max += 1

            num_xticks = 5
            for i in range(num_xticks+1):
                frac = i/num_xticks
                px = int(x0 + frac*width)
                t_val = t_min + frac*(t_max-t_min)
                cv2.line(img, (px, y0), (px, y0+8), (220,220,220), 1)
                label = f"{t_val:.1f}"
                cv2.putText(img, label, (px-15, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
            num_yticks = 5
            for i in range(num_yticks+1):
                frac = i/num_yticks
                py = int(y0 - frac*(y0-y1))
                y_val = y_min + frac*(y_max-y_min)
                cv2.line(img, (x0-8, py), (x0, py), (220,220,220), 1)
                cv2.putText(img, f"{y_val:.1f}", (5, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

            for idx, arr in enumerate(visible_bufs):
                # Only plot if there are at least 2 points
                if arr is None or arr.shape[0] < 2 or arr.shape[1] < 2:
                    continue
                t_vals = arr[:,0]
                y_vals = arr[:,1]
                # Defensive: if all t_vals or y_vals are the same, skip to avoid OpenCV errors
                if np.all(t_vals == t_vals[0]) or np.all(y_vals == y_vals[0]):
                    continue
                x_pix = np.interp(t_vals, (t_min, t_max), (x0, x1)).astype(np.int32)
                y_pix = np.interp(y_vals, (y_min, y_max), (y0, y1)).astype(np.int32)
                if len(x_pix) < 2 or len(y_pix) < 2:
                    continue
                pts = np.stack([x_pix, y_pix], axis=1).reshape(-1,1,2)
                cv2.polylines(img, [pts], isClosed=False, color=colors[idx%len(colors)], thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(img, f"Ch {idx+1}", (x1-60, y1+30+idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[idx%len(colors)], 2, cv2.LINE_AA)

            cv2.putText(img, "Time", (img_width//2-30, img_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)
            cv2.putText(img, "Value", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)

            cv2.imshow(window_title, img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to quit
                break

        cv2.destroyWindow(window_title)

    @staticmethod
    def static_opencv_plot(
        signals,
        show_peaks=False,
        window_title="Static OpenCV Plot",
        height=None,
        width=None,
        fixed_y_min=None,
        fixed_y_max=None,
        peaks_info=None,
        buffer_sizes=None,  # list of buffer sizes per channel
        axis_info=None      # list of (t_min, t_max) per channel
    ):
        # Fastest possible: precompute all coordinates, use numpy vector ops, minimal Python loops
        n_signals = len([arr for arr in signals if getattr(arr, "size", 0) > 0])
        plot_height_per_signal = 250
        min_height = 200
        plot_height = max(min_height, plot_height_per_signal * n_signals)
        plot_width = width if width is not None else 800

        left_margin = 60
        right_margin = 20
        top_margin = 20
        bottom_margin = 50

        img_width = plot_width + left_margin + right_margin
        img_height = plot_height + top_margin + bottom_margin

        img = np.full((img_height, img_width, 3), 30, dtype=np.uint8)

        # Find valid signals
        valid_signals = [arr for arr in signals if getattr(arr, "size", 0) > 0 and arr.shape[1] >= 2]
        if not valid_signals:
            cv2.imshow(window_title, img)
            cv2.waitKey(1)
            return

        colors = np.array([
            [0, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [255, 0, 255]
        ], dtype=np.uint8)

        # Each channel gets its own time axis, buffer size, and axis info if provided
        for idx, arr in enumerate(signals):
            if arr.shape[1] < 2 or arr.shape[0] < 2:
                continue

            # Use buffer size per channel if provided
            if buffer_sizes is not None and idx < len(buffer_sizes):
                arr = arr[-int(buffer_sizes[idx]):]

            subplot_top = top_margin + idx * plot_height_per_signal
            subplot_bottom = subplot_top + plot_height_per_signal - 1
            x0 = left_margin
            x1 = left_margin + plot_width
            y0 = subplot_bottom - bottom_margin
            y1 = subplot_top + top_margin

            # Use axis info per channel if provided, else autoscale
            if axis_info is not None and idx < len(axis_info) and axis_info[idx] is not None:
                t_min, t_max = axis_info[idx]
            else:
                t_min = arr[:, 0].min()
                t_max = arr[:, 0].max()
                if t_max == t_min:
                    t_max += 1.0

            y_min = arr[:, 1].min()
            y_max = arr[:, 1].max()
            if y_max == y_min:
                y_max += 1.0

            # X ticks/labels
            num_xticks = 4
            xtick_fracs = np.linspace(0, 1, num_xticks + 1)
            xtick_vals = t_min + xtick_fracs * (t_max - t_min)
            xtick_px = (x0 + xtick_fracs * plot_width).astype(int)
            for px, t_val in zip(xtick_px, xtick_vals):
                cv2.line(img, (px, y0), (px, y0 + 4), (220, 220, 220), 1)
                cv2.putText(img, f"{t_val:.1f}", (px-15, y0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1, cv2.LINE_4)

            # Y ticks/labels
            num_yticks = 4
            ytick_fracs = np.linspace(0, 1, num_yticks + 1)
            ytick_vals = y_min + ytick_fracs * (y_max - y_min)
            ytick_py = (y1 + ytick_fracs * (y0 - y1)).astype(int)
            for py, y_val in zip(ytick_py, ytick_vals):
                cv2.line(img, (x0 - 4, py), (x0, py), (220, 220, 220), 1)
                cv2.putText(img, f"{y_val:.1f}", (5, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1, cv2.LINE_4)

            # Map data to pixel coordinates
            timestamps = arr[:, 0]
            values = arr[:, 1]
            x_pix = np.interp(timestamps, (t_min, t_max), (x0, x1)).astype(np.int32)
            y_pix = np.interp(values, (y_min, y_max), (y0, y1)).astype(np.int32)
            color = tuple(int(c) for c in colors[idx % len(colors)])

            if len(x_pix) > 1:
                pts = np.stack([x_pix, y_pix], axis=1).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_4)

            # Highlight peaks
            if show_peaks and arr.shape[1] >= 3:
                peaks = arr[:, 2].astype(bool)
                if np.any(peaks):
                    for px, py in zip(x_pix[peaks], y_pix[peaks]):
                        cv2.circle(img, (px, py), 2, (0, 0, 255), -1, lineType=cv2.LINE_4)

            cv2.putText(img, f"Ch {idx+1}", (x1 - 40, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_4)

        # Axis labels (bottom plot only, small)
        cv2.putText(img, "Time", (img_width // 2 - 20, img_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_4)
        cv2.putText(img, "Value", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_4)

        cv2.imshow(window_title, img)
        cv2.waitKey(1)

    def static_pyqtgraph(self, signals, show_peaks=False):
        """
        Static plot of multiple signals using PyQtGraph.
        """
        # Explicitly remove all items from the layout
        while len(self.widget.ci.items):  # `ci` is the central item (GraphicsLayout)
            self.widget.ci.removeItem(self.widget.ci.items[-1])

        n_channels = len(signals)
        for i, sig in enumerate(signals):
            p = self.widget.addPlot(row=i, col=0)
            label = f'Ch {i+1}'
            if self.suffix:
                label += f' {self.suffix}'
            p.setLabel('left', label)
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (s)')

            arr = np.array(sig)
            if arr.ndim == 1:
                t_vals = np.linspace(0, self.duration, len(arr))
                arr = np.column_stack((t_vals, arr))

            t_vals = arr[:, 0]
            y_vals = arr[:, 1]
            peaks = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(y_vals)
            p.plot(t_vals, y_vals, pen=pg.mkPen(width=2))
            peak_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0))
            if show_peaks:
                peak_scatter.setData(t_vals[peaks == 1], y_vals[peaks == 1])
            else:
                peak_scatter.setData([], [])
            p.addItem(peak_scatter)

    def render_to_image(self):
        """
        Render the PyQtGraph widget to a QImage.
        """
        size = self.widget.size()
        qimage = QImage(size.width(), size.height(), QImage.Format.Format_RGBA8888)
        painter = QPainter(qimage)
        self.widget.render(painter)
        painter.end()
        return qimage

    def render_to_numpy(self):
        """
        Render the PyQtGraph widget to a NumPy array for OpenCV.
        """
        qimage = self.render_to_image()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())  # <-- fix: use sizeInBytes() for PyQt6
        arr = np.array(ptr).reshape((height, width, 4))  # Convert to numpy array
        return arr

    def get_plot_data(self, signals, show_peaks=False):
        """
        Extract the data of up to three signals for plotting without rendering as an image.
        Returns a list of dictionaries containing time values, signal values, and peaks.
        """
        plot_data = []
        n_channels = min(len(signals), 3)  # Limit to three channels
        for i, sig in enumerate(signals[:3]):  # Process only the first three signals
            arr = np.array(sig)
            if arr.ndim == 1:
                t_vals = np.linspace(0, self.duration, len(arr))
                arr = np.column_stack((t_vals, arr))

            t_vals = arr[:, 0]
            y_vals = arr[:, 1]
            peaks = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(y_vals)

            # Store the data in a dictionary
            channel_data = {
                "channel": i + 1,
                "time": t_vals,
                "values": y_vals,
                "peaks": t_vals[peaks == 1] if show_peaks else [],
            }
            plot_data.append(channel_data)

        return plot_data

    def render(self, image=None, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, window_title=DEFAULT_WINDOW_TITLE, cap_fps=False):
        """
        Render an image using the internal renderer, handling PIL, numpy, or torch images.
        Optionally cap the FPS.
        """
        current_time = time.time()
        elapsed_time = current_time - getattr(self, 'last_exec_time', 0)
        if cap_fps:
            sleep_time = max(0, (1.0 / getattr(self, 'MAX_FPS', 30)) - elapsed_time)
            time.sleep(sleep_time)

        if image is None:
            return ()

        if isinstance(image, PILImage.Image):
            image = np.asarray(image)
            image = torch.from_numpy(image.copy())
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            if image.max() <= 1.0:
                image = image * 255
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            image = torch.from_numpy(image.copy())
        elif torch.is_tensor(image):
            if image.ndim == 2:
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            if torch.max(image) <= 1.0:
                image = image * 255
            image = image.to(torch.uint8)
            image = image.squeeze(0)

        if not hasattr(self, 'renderer') or self.renderer is None or height != getattr(self, 'render_size', (None, None))[0] or width != getattr(self, 'render_size', (None, None))[1]:
            self.render_size = (height, width)
            # Replace lt.Renderer with your actual renderer class if needed
            self.renderer = lt.Renderer(width=int(width), height=int(height), window_title=window_title)

        self.renderer.render(image)
        self.last_exec_time = time.time()

def main():
    """
    Create a static plot using the Plot class and display it in an OpenCV window.
    """
    app = QApplication([])

    # Create the plot
    plotter = Plot()

    # Generate dummy data
    t = np.linspace(0, 10, 500)
    signal1 = np.column_stack((t, 200 + 100 * np.sin(0.5 * t)))
    signal2 = np.column_stack((t, 300 + 50 * np.cos(0.5 * t)))
    signals = [signal1, signal2]

    # Plot the data using PyQtGraph (existing example)
    plotter.static_pyqtgraph(signals)

    # Access the PyQtGraph widget directly
    if hasattr(plotter, 'widget'):  # Assuming `Plot` exposes the widget as `widget`
        widget = plotter.widget
    else:
        raise AttributeError("The Plot class does not expose a PyQt widget for rendering.")

    # Render the widget into a QImage
    size = widget.size()
    qimage = QImage(size.width(), size.height(), QImage.Format.Format_RGBA8888)
    painter = QPainter(qimage)
    widget.render(painter)
    painter.end()

    # Convert QImage to NumPy array
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.sizeInBytes())  # Use sizeInBytes() instead of byteCount()
    arr = np.array(ptr).reshape((height, width, 4))  # Convert to numpy array

    # Convert RGBA to BGR for OpenCV
    bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    # Display the image in an OpenCV window
    cv2.imshow("Static Plot", bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Example: Plot the data using OpenCV static plot directly
    Plot.static_opencv_plot(
        signals,
        show_peaks=False,
        window_title="Static OpenCV Plot Example"
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()