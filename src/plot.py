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

    def plot(self, signals):
        """
        Plot signals using the selected mode (live or static).
        Accepts 1, 2, or 3 signals (as list or single array/callable).
        """
        # Ensure signals is a list of 1-3 items
        if not isinstance(signals, (list, tuple)):
            signals = [signals]
        if len(signals) > 3:
            signals = signals[:3]
        if self.live:
            self.live_pyqtgraph(signals)
        else:
            self.static_pyqtgraph(signals)

    def live_pyqtgraph(self, signals):
        """
        Live plot multiple signals using PyQtGraph, updating in real time.
        signals: list of deques of (timestamp, value) pairs.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)
        plots = []
        curves = []
        # Store last plotted timestamp for each channel
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

        def update():
            for i, sig in enumerate(signals):
                data = sig() if callable(sig) else sig
                if not data or len(data) == 0:
                    curves[i].setData([], [])
                    self._last_plotted_ts[i] = None
                    continue
                arr = np.array(data)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    curves[i].setData([], [])
                    self._last_plotted_ts[i] = None
                    continue

                # Only plot new points since last update, optimized with numpy
                window = self.window  # seconds
                # Keep only points within the last 'window' seconds
                max_time = arr[-1, 0]
                min_time = max_time - window
                in_window = arr[:, 0] >= min_time
                arr = arr[in_window]
                if self._last_plotted_ts[i] is not None:
                    idx = np.searchsorted(arr[:, 0], self._last_plotted_ts[i], side='right')
                    t_vals = arr[idx:, 0]
                    y_vals = arr[idx:, 1]
                    old_t, old_y = curves[i].getData()
                    if old_t is not None and len(old_t) > 0:
                        # Keep only old points within the window
                        keep_idx = old_t >= min_time
                        old_t = old_t[keep_idx]
                        old_y = old_y[keep_idx]
                        t_vals = np.concatenate((old_t, t_vals))
                        y_vals = np.concatenate((old_y, y_vals))
                else:
                    t_vals = arr[:, 0]
                    y_vals = arr[:, 1]
                curves[i].setData(t_vals, y_vals)
                self._last_plotted_ts[i] = arr[-1, 0]
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)
        app.exec()

    def static_pyqtgraph(self, signals):
        """
        Static plot of multiple signals using PyQtGraph.
        Call this method again with new data to update the plot.
        signals: list of arrays.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)
        t = np.linspace(0, self.duration, int(self.fs * self.duration))
        for i, sig in enumerate(signals):
            p = win.addPlot(row=i, col=0)
            label = f'Ch {i+1}'
            if self.suffix:
                label += f' {self.suffix}'
            p.setLabel('left', label)
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (s)')
            y = np.asarray(sig)
            if len(y) > len(t):
                y = y[-len(t):]
            elif len(y) < len(t):
                y = np.pad(y, (len(t)-len(y), 0), mode='constant')
            p.plot(t, y, pen=pg.mkPen(width=2))
        app.exec()

def main():
    import threading
    import time

    mode = input("Type 'live' or 'static': ").strip().lower()
    live = mode == "live"

    fs = 500
    duration = 100
    window = 5  # Example: set window size to 2 seconds for live mode
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
                    dq.append((rel_time, val))
                time.sleep(1/fs)

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
        plotter = Plot(fs=fs, duration=duration, live=live)
        plotter.plot([sig1, sig2, sig3])

if __name__ == "__main__":
    main()