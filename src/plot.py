import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from collections import deque
import random

class Plot:
    """
    Plotting utility class for live and static signal visualization.
    """

    def __init__(self, fs=1000, duration=5, live=False, suffix="", window=None):
        self.fs = fs
        self.duration = duration
        self.live = live
        self.suffix = suffix  # Suffix for channel labels
        self.window = window if window is not None else duration  # Window size for live mode

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
        signals: list of deques of (timestamp, value) or (timestamp, value, is_peak) pairs.
        If show_peaks is True, plot points where is_peak==1.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)
        plots = []
        curves = []
        peak_markers = []  # To store peak markers for each signal
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
                data = list(sig)  # Convert deque to a list
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

                # Handle optional is_peak flag
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

    def static_pyqtgraph(self, signals, show_peaks=False):
        """
        Static plot of multiple signals using PyQtGraph.
        signals: list of arrays with (timestamp, value) or (timestamp, value, is_peak).
        If show_peaks is True, plot points where is_peak==1.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)
        for i, sig in enumerate(signals):
            p = win.addPlot(row=i, col=0)
            label = f'Ch {i+1}'
            if self.suffix:
                label += f' {self.suffix}'
            p.setLabel('left', label)
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (s)')
            
            # Ensure signal is 2D with timestamps
            arr = np.array(sig)
            print("Warning: Signal is 1D, assuming evenly spaced timestamps.")
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
        app.exec()

def main():
    import threading
    import time

    mode = input("Type 'live' or 'static': ").strip().lower()
    live = mode == "live"

    fs = 1000
    duration = 100
    window = 10  # Example: set window size to 2 seconds for live mode
    t = np.linspace(0, duration, int(fs * duration))
    # Generate example signals
    # For both static and live, generate the same signals
    # sig1: simulate an ECG-like waveform
    heart_rate = 60  # bpm
    beats_per_sec = heart_rate / 60
    # Basic synthetic ECG: sum of narrow positive and negative Gaussians for QRS, P, T waves
    sig1 = np.zeros_like(t)
    for beat in np.arange(0, duration, 1/beats_per_sec):
        # QRS complex (sharp spike)
        sig1 += 1.2 * np.exp(-0.5 * ((t - beat) / 0.03) ** 2)
        sig1 -= 0.5 * np.exp(-0.5 * ((t - (beat - 0.04)) / 0.01) ** 2)  # Q dip
        sig1 += 0.3 * np.exp(-0.5 * ((t - (beat + 0.04)) / 0.02) ** 2)  # S
        # P wave (small bump before QRS)
        sig1 += 0.2 * np.exp(-0.5 * ((t - (beat - 0.2)) / 0.04) ** 2)
        # T wave (broad bump after QRS)
        sig1 += 0.35 * np.exp(-0.5 * ((t - (beat + 0.2)) / 0.07) ** 2)
    sig1 += 0.05 * np.random.randn(len(t))  # Add some noise
    # sig2: simulate electrodermal activity (EDA) with larger, slower, and less frequent Gaussian "bakes" (sweat bursts)
    period = 8  # seconds, less frequent events
    center_times = np.arange(0, duration, period)
    width = 0.8  # seconds, wider Gaussian for slower rise/fall
    sig2 = np.zeros_like(t)
    for ct in center_times:
        sig2 += 3.0 * np.exp(-0.5 * ((t - ct) / width) ** 2)  # larger amplitude, slower decay
    drift_rate = 0.1  # slower baseline drift
    sig2 += drift_rate * t
    sig2 += 0.05 * np.random.randn(len(t))  # small noise to mimic skin conductance variability
    # sig3: trigonometric wave with phase shift and occasional quick noise bursts
    # Make phase a random walk over time
    np.random.seed(42)
    phase = np.cumsum(np.random.randn(len(t)) * 0.02)
    sig3 = 2*np.sin(2 * np.pi * 1 * t + phase) 
    # Add quick noise bursts at random times
    burst_indices = np.random.choice(len(t), size=10, replace=False)
    for idx in burst_indices:
        if idx + 5 < len(sig3):
            sig3[idx:idx+5] += np.random.normal(0, 2, size=5)

    # For live mode, use the same signal generation in the acquisition threads

    if live:
        # Start with empty deques for live mode
        dq1 = deque(maxlen=int(fs * duration))
        dq2 = deque(maxlen=int(fs * duration))
        dq3 = deque(maxlen=int(fs * duration))
        t0 = time.time()
        running = True

        def acq_thread(dq, idx):
            while running:
                now = time.time()
                rel_time = now - t0
                sample_idx = int(rel_time * fs)
                if sample_idx < len(t):
                    if idx == 0:
                        val = sig1[sample_idx]
                    elif idx == 1:
                        val = sig2[sample_idx]
                    else:
                        val = sig3[sample_idx]
                    # Append (timestamp, value) pair to the deque
                    dq.append((rel_time, val))
                    print(f"Thread {idx}: Appended value {val} at time {rel_time:.2f}s")
                    print(dq)
                

        threads = [
            threading.Thread(target=acq_thread, args=(dq1, 0), daemon=True),
            threading.Thread(target=acq_thread, args=(dq2, 1), daemon=True),
            threading.Thread(target=acq_thread, args=(dq3, 2), daemon=True),
        ]
        for th in threads:
            th.start()

        plotter = Plot(fs=fs, duration=duration, live=live, window=window)
        try:
            plotter.plot([dq1, dq2, dq3])
        finally:
            running = False
            # Optionally join threads if you want to wait for them to finish
            # for th in threads:
            #     th.join()
    else:
        # Ensure synthetic signals are 2D arrays with timestamps for static mode
        sig1 = np.column_stack((t, sig1))  # Add timestamps to sig1
        sig2 = np.column_stack((t, sig2))  # Add timestamps to sig2
        sig3 = np.column_stack((t, sig3))  # Add timestamps to sig3

        # Pass the properly formatted signals to the Plot class
        plotter = Plot(fs=fs, duration=duration, live=live)
        plotter.plot([sig1, sig2, sig3])

if __name__ == "__main__":
    main()