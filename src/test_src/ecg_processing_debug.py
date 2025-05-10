import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore  # Import QtCore for DashLine
import time
import os
from scipy.signal import hilbert

from signal_processing import NumpySignalProcessor
from ecg_signal_processing import ECG

def plot_methods_grid(methods_steps, fs, methods_names, peaks_list):
    """
    Plot: first two columns are always 'Raw' and 'Bandpass' (with envelope overlay on Bandpass),
    then each method's main output (with envelope overlay), and the last column is peaks/validated peaks.
    Only show the first 5 seconds of the signal.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    n_methods = len(methods_steps)
    n_cols = 4
    win = pg.GraphicsLayoutWidget(show=True, title="ECG Signal Processing Comparison")

    # Dynamically determine plot size based on screen size and grid
    screen = app.primaryScreen().availableGeometry()
    margin_w = 100
    margin_h = 150
    plot_width = max(120, int((screen.width() - margin_w) / n_cols))
    plot_height = max(80, int((screen.height() - margin_h) / n_methods))

    smoothing_window = 7  # Reduced for less overlap and better fit
    max_samples = int(fs * 5)  # Only show first 5 seconds

    for row in range(n_methods):
        steps = methods_steps[row]
        peaks, validated_peaks = peaks_list[row]
        # Find indices for required steps
        idx_raw = 0
        idx_bandpass = 1
        idx_env = None
        idx_main = None
        # Find envelope and main step indices
        for i, (label, _) in enumerate(steps):
            if "envelope" in label.lower() and idx_env is None:
                idx_env = i
            if "normalized" in label.lower() or "baseline" in label.lower():
                idx_main = i

        # Column 0: Raw
        label, data = steps[idx_raw]
        p = win.addPlot(row=row, col=0)
        p.setFixedWidth(plot_width)
        p.setFixedHeight(plot_height)
        p.setTitle("Raw")
        p.showGrid(x=True, y=True, alpha=0.7)
        p.setLabel('bottom', 'Time', units='s')
        p.setLabel('left', 'Amplitude')
        x = np.arange(len(data)) / fs
        x = x[:max_samples]
        y = data[:max_samples]
        p.plot(x, y, pen=pg.mkPen('b', width=1))

        # Column 1: Bandpass + Envelope
        label, data = steps[idx_bandpass]
        p = win.addPlot(row=row, col=1)
        p.setFixedWidth(plot_width)
        p.setFixedHeight(plot_height)
        p.setTitle("Bandpass + Envelope")
        p.showGrid(x=True, y=True, alpha=0.7)
        p.setLabel('bottom', 'Time', units='s')
        p.setLabel('left', 'Amplitude')
        x = np.arange(len(data)) / fs
        x = x[:max_samples]
        y = data[:max_samples]
        p.plot(x, y, pen=pg.mkPen('b', width=1))
        # Overlay envelope if available
        if idx_env is not None:
            env_label, env_data = steps[idx_env]
            env = env_data[:max_samples]
            smoothed_env = np.convolve(env, np.ones(smoothing_window) / smoothing_window, mode='same')[:max_samples]
            p.plot(x, smoothed_env, pen=pg.mkPen('r', width=2))

        # Column 2: Main method output + envelope
        label, data = steps[idx_main]
        p = win.addPlot(row=row, col=2)
        p.setFixedWidth(plot_width)
        p.setFixedHeight(plot_height)
        p.setTitle(label)
        p.showGrid(x=True, y=True, alpha=0.7)
        p.setLabel('bottom', 'Time', units='s')
        p.setLabel('left', 'Amplitude')
        if isinstance(data, tuple):
            y1, y2 = data
            x = np.arange(len(y1)) / fs
            x = x[:max_samples]
            y1 = y1[:max_samples]
            y2 = y2[:max_samples]
            p.plot(x, y1, pen=pg.mkPen('b', width=1))
            smoothed_envelope = np.convolve(y2, np.ones(smoothing_window) / smoothing_window, mode='same')
            smoothed_envelope = smoothed_envelope[:max_samples]
            p.plot(x, smoothed_envelope, pen=pg.mkPen('r', width=2))
        else:
            x = np.arange(len(data)) / fs
            x = x[:max_samples]
            y = data[:max_samples]
            p.plot(x, y, pen=pg.mkPen('b', width=1))

        # Column 3: Peaks/Validated Peaks (from main output)
        # Use the same data as main output for peaks
        p = win.addPlot(row=row, col=3)
        p.setFixedWidth(plot_width)
        p.setFixedHeight(plot_height)
        p.setTitle("Peaks/Validated Peaks")
        p.showGrid(x=True, y=True, alpha=0.7)
        p.setLabel('bottom', 'Time', units='s')
        p.setLabel('left', 'Amplitude')
        if isinstance(data, tuple):
            y1, y2 = data
            x = np.arange(len(y1)) / fs
            x = x[:max_samples]
            y1 = y1[:max_samples]
            y2 = y2[:max_samples]
            p.plot(x, y1, pen=pg.mkPen('b', width=1))
            smoothed_envelope = np.convolve(y2, np.ones(smoothing_window) / smoothing_window, mode='same')
            smoothed_envelope = smoothed_envelope[:max_samples]
            p.plot(x, smoothed_envelope, pen=pg.mkPen('r', width=2))
            y_plot = y1
        else:
            x = np.arange(len(data)) / fs
            x = x[:max_samples]
            y_plot = data[:max_samples]
            p.plot(x, y_plot, pen=pg.mkPen('b', width=1))
        # Plot peaks and validated peaks
        if len(peaks) > 0:
            peaks_in_window = [pk for pk in peaks if pk < max_samples]
            x_peaks = np.array(peaks_in_window) / fs
            y_peaks = y_plot[peaks_in_window]
            p.plot(x_peaks, y_peaks, pen=None, symbol='o', symbolBrush='g', symbolSize=8, name="Detected Peaks")
            if validated_peaks is not None and len(validated_peaks) > 0:
                val_in_window = [pk for pk in validated_peaks if pk < max_samples]
                x_val = np.array(val_in_window) / fs
                y_val = y_plot[val_in_window]
                p.plot(x_val, y_val, pen=None, symbol='t', symbolBrush='m', symbolSize=10, name="Validated Peaks")
        # Left label for method
        for c in range(n_cols):
            if c == 0:
                win.getItem(row=row, col=c).setLabel('left', methods_names[row])
    win.show()
    app.exec()

def run_methods(raw, fs, mode="all"):
    """
    Run different ECG processing methods and calculate the envelope within each method.
    """
    benchmarks = []

    # If mode is 'qrs', always apply bandpass 8-15Hz at the start of each method
    t0 = time.perf_counter()
    bandpassed = NumpySignalProcessor.bandpass_filter(raw, 8, 15, fs, order=4) if mode == "qrs" else raw
    t_bandpass = time.perf_counter() - t0

    # Method 1: bandpass, moving avg, normalize, envelope, peaks
    steps1, peaks1, validated_peaks1 = process_method_1(raw, bandpassed, fs, mode)
    benchmarks.append(("Bandpass+Peaks", time.perf_counter() - t0))

    # Method 2: ALS baseline only, envelope corrected, peaks
    steps2, peaks2, validated_peaks2 = process_method_2(raw, bandpassed, fs, mode)
    benchmarks.append(("ALS Baseline+Peaks", time.perf_counter() - t0))

    # Method 3: moving average only, normalize, envelope, peaks
    steps3, peaks3, validated_peaks3 = process_method_3(raw, bandpassed, fs, mode)
    benchmarks.append(("Moving Avg Only+Peaks", time.perf_counter() - t0))

    methods_steps = [steps1, steps2, steps3]
    methods_names = [
        "Bandpass+Peaks",
        "ALS Baseline+Peaks",
        "Moving Avg Only+Peaks"
    ]
    peaks_list = [
        (peaks1, validated_peaks1),
        (peaks2, validated_peaks2),
        (peaks3, validated_peaks3)
    ]

    print("\n--- Benchmarks (seconds) ---")
    for name, t in benchmarks:
        print(f"{name:25s}: {t:.6f}")

    return methods_steps, methods_names, peaks_list

def process_method_1(raw, bandpassed, fs, mode):
    """
    Method 1: Bandpass, moving average, normalize, envelope, peaks.
    """
    steps = []
    steps.append(("Raw", raw))
    steps.append(("Bandpass", bandpassed))
    envelope = np.abs(hilbert(bandpassed))  # Calculate envelope before normalization
    steps.append(("Envelope", envelope))
    normed = NumpySignalProcessor.normalize_signal(bandpassed)
    normed_envelope = NumpySignalProcessor.normalize_signal(envelope)
    normed_envelope = NumpySignalProcessor.moving_average(normed_envelope, window_size=5)
    steps.append(("Normalized", (normed, normed_envelope)))  # Plot both normed and normed_envelope
    peaks = ECG.detect_r_peaks(normed, fs, mode=mode)
    validated_peaks = ECG.validate_r_peaks(normed_envelope, peaks, fs, envelope_threshold=0.5)
    return steps, peaks, validated_peaks

def process_method_2(raw, bandpassed, fs, mode):
    """
    Method 2: Apply ALS baseline correction to the envelope derived from the bandpassed signal.
    Show both the envelope of the bandpassed signal and the envelope after baseline correction.
    """
    steps = []
    steps.append(("Raw", raw))
    steps.append(("Bandpass", bandpassed))
    envelope = np.abs(hilbert(bandpassed))
    steps.append(("Envelope", envelope))  # Plot envelope over bandpass
    baseline_corrected = NumpySignalProcessor.correct_baseline(bandpassed, method="als", lam=1e5, p=0.01, niter=10)
    steps.append(("Baseline Corrected Envelope", baseline_corrected))
    normed_baseline = NumpySignalProcessor.normalize_signal(baseline_corrected)
    normed_envelope = NumpySignalProcessor.normalize_signal(envelope)
    normed_envelope = NumpySignalProcessor.moving_average(normed_envelope, window_size=5)
    steps.append(("Normalized", (normed_baseline, normed_envelope)))  # Use same label for comparability
    peaks = ECG.detect_r_peaks(normed_baseline, fs, mode=mode)
    validated_peaks = ECG.validate_r_peaks(normed_envelope, peaks, fs, envelope_threshold=0.5)
    return steps, peaks, validated_peaks

def process_method_3(raw, bandpassed, fs, mode):
    """
    Method 3: Moving average, normalize, and apply envelope logic.
    """
    steps = []
    steps.append(("Raw", raw))
    steps.append(("Bandpass", bandpassed))
    smoothed = NumpySignalProcessor.moving_average(bandpassed, window_size=15)
    steps.append(("Moving Avg", smoothed))
    envelope = np.abs(hilbert(smoothed))
    steps.append(("Envelope", envelope))
    normed = NumpySignalProcessor.normalize_signal(smoothed)
    normed_envelope = NumpySignalProcessor.normalize_signal(envelope)
    normed_envelope = NumpySignalProcessor.moving_average(normed_envelope, window_size=5)
    steps.append(("Normalized", (normed, normed_envelope)))
    peaks = ECG.detect_r_peaks(normed, fs, mode=mode)
    validated_peaks = ECG.validate_r_peaks(normed_envelope, peaks, fs, envelope_threshold=0.5)
    return steps, peaks, validated_peaks

if __name__ == "__main__":
    # Load sample ECG signals (use your own paths as needed)
    heart_file_path = os.path.join(os.path.dirname(__file__), "ECG", "heart_signal_data.json")
    collarbone_file_path = os.path.join(os.path.dirname(__file__), "ECG", "collarbone_signal_data.json")

    raw_heart_signal = NumpySignalProcessor.load_signal(heart_file_path)
    raw_collarbone_signal = NumpySignalProcessor.load_signal(collarbone_file_path)
    fs = 1000

    print("Processing heart signal...")
    print("Plotting all methods for mode='all'...")
    methods_steps, methods_names, peaks_list = run_methods(raw_heart_signal, fs, mode="all")
    plot_methods_grid(methods_steps, fs, methods_names, peaks_list)

    print("Plotting all methods for mode='qrs'...")
    methods_steps_qrs, methods_names_qrs, peaks_list_qrs = run_methods(raw_heart_signal, fs, mode="qrs")
    plot_methods_grid(methods_steps_qrs, fs, methods_names_qrs, peaks_list_qrs)

    print("Processing collarbone signal...")
    print("Plotting all methods for mode='all'...")
    methods_steps_cb, methods_names_cb, peaks_list_cb = run_methods(raw_collarbone_signal, fs, mode="all")
    plot_methods_grid(methods_steps_cb, fs, methods_names_cb, peaks_list_cb)

    print("Plotting all methods for mode='qrs'...")
    methods_steps_cb_qrs, methods_names_cb_qrs, peaks_list_cb_qrs = run_methods(raw_collarbone_signal, fs, mode="qrs")
    plot_methods_grid(methods_steps_cb_qrs, fs, methods_names_cb_qrs, peaks_list_cb_qrs)
