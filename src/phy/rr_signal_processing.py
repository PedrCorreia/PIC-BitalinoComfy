import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from ..utils.signal_processing import NumpySignalProcessor
import os

class RR:

    @staticmethod
    def extract_respiration_rate(filtered_signal, fs):
        """
        Extracts the respiration rate and average breath duration.
        """
        peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs)
        if len(peaks) < 2:
            return 0, 0
        rr = len(peaks) * 60 / (len(filtered_signal) / fs)
        durations = np.diff(peaks) / fs
        avg_breath_duration = float(np.mean(durations)) if len(durations) > 0 else 0
        return rr, avg_breath_duration

    @staticmethod
    def preprocess_signal(signal, fs):
        """
        Preprocesses the signal by filtering and normalizing.
        """
        # Filtering
        filtered_signal = NumpySignalProcessor.bandpass_filter(signal, 0.1, 0.5, fs)

        # Normalization
        normalized_signal = NumpySignalProcessor.normalize_signal(filtered_signal)
        
        return normalized_signal

    @staticmethod
    def detect_deep_breaths(filtered_signal, peaks, threshold_factor=1.5):
        """
        Detects deep breaths based on peak amplitude.
        
        Parameters:
        - filtered_signal: The filtered signal.
        - peaks: Indices of detected peaks.
        - threshold_factor: Multiplier for the standard deviation to set the deep breath threshold.
        
        Returns:
        - List of indices corresponding to deep breaths.
        """
        peak_amplitudes = filtered_signal[peaks]
        threshold = np.mean(peak_amplitudes) + threshold_factor * np.std(peak_amplitudes)
        deep_breaths = [i for i, amp in zip(peaks, peak_amplitudes) if amp > threshold]
        return deep_breaths

    @staticmethod
    def plot_signals(raw, preprocessed, filtered, peaks, valleys, rr, avg_breath_duration, fs, deep_breaths):
        """
        Plots the raw, preprocessed, and filtered signals along with detected peaks, valleys, and deep breaths.
        """
        time_np = np.arange(len(raw)) / fs
        freqs_raw, psd_raw = NumpySignalProcessor.compute_psd_numpy(raw, fs)
        freqs_preprocessed, psd_preprocessed = NumpySignalProcessor.compute_psd_numpy(preprocessed, fs)
        freqs_filtered, psd_filtered = NumpySignalProcessor.compute_psd_numpy(filtered, fs)
        duration = len(raw) / fs

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="Respiratory Signal Analysis (NumPy)")
        win.resize(1800, 1200)
        win.setWindowTitle("Respiratory Signal Analysis (NumPy)")

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # Raw signal plot
        p1 = win.addPlot(row=0, col=0, title="<b>Raw Signal</b>")
        p1.plot(time_np, raw, pen=pg.mkPen(color=(100, 200, 255), width=1.2))
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p1.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Preprocessed signal plot
        p2 = win.addPlot(row=1, col=0, title="<b>Preprocessed Signal</b>")
        p2.plot(time_np, preprocessed, pen=pg.mkPen(color=(255, 255, 0), width=1.2))
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p2.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Filtered signal with peaks, valleys, and deep brhzeaths plot
        p3 = win.addPlot(row=2, col=0, title="<b>Filtered Signal with Peaks, Valleys, and Deep Breaths</b>")
        p3.plot(time_np, filtered, pen=pg.mkPen(color=(255, 170, 0), width=2))
        if len(peaks) > 0:
            p3.plot(time_np[peaks], filtered[peaks], pen=None, symbol='x', symbolBrush=(255, 80, 80), symbolPen='r', symbolSize=14)
        if len(valleys) > 0:
            p3.plot(time_np[valleys], filtered[valleys], pen=None, symbol='o', symbolBrush=(80, 255, 80), symbolPen='g', symbolSize=10)
        if len(deep_breaths) > 0:
            p3.plot(time_np[deep_breaths], filtered[deep_breaths], pen=None, symbol='s', symbolBrush=(0, 255, 255), symbolPen='c', symbolSize=16)
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p3.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Text summary and PSD legend
        text = f"<span style='font-size:12pt'><b>Respiration Rate:</b> <span style='color:#ffae00'>{rr:.2f}</span> breaths/min<br>"
        text += f"<b>Average Breath Duration:</b> <span style='color:#ffae00'>{avg_breath_duration:.2f}</span> s<br>"
        text += f"<b>Sampling Rate:</b> <span style='color:#ffae00'>{fs} Hz</span><br>"
        text += f"<b>Signal Duration:</b> <span style='color:#ffae00'>{duration:.2f} s</span><br><br>"
        text += "<b>PSD Legend:</b><br>"
        text += "<span style='color:#64c8ff'>Raw Signal (Blue)</span><br>"
        text += "<span style='color:#ffff00'>Preprocessed Signal (Yellow)</span><br>"
        text += "<span style='color:#ffaa00'>Filtered Signal (Orange)</span><br>"
        text += "<span style='color:#00ffff'>Deep Breaths (Cyan Squares)</span></span>"
        label = pg.LabelItem(text, justify='left')
        win.addItem(label, row=0, col=1, rowspan=1)

        # PSD spectrum plot
        p4 = win.addPlot(row=1, col=1, rowspan=2, title="<b>Power Spectral Density (PSD)</b>")
        p4.plot(freqs_raw, psd_raw, pen=pg.mkPen(color=(100, 200, 255), width=1.2))
        p4.plot(freqs_preprocessed, psd_preprocessed, pen=pg.mkPen(color=(255, 255, 0), width=1.2))
        p4.plot(freqs_filtered, psd_filtered, pen=pg.mkPen(color=(255, 170, 0), width=2))
        p4.setLabel('left', "<span style='color:white'>PSD [V**2/Hz]</span>")
        p4.setLabel('bottom', "<span style='color:white'>Frequency [Hz]</span>")
        p4.showGrid(x=True, y=True, alpha=0.3)

        app.exec()

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "RR", "signal_data.json")
    raw_signal = NumpySignalProcessor.load_signal(file_path)  # Use NumpySignalProcessor to load the signal
    
    fs = 1000
    # Preprocess the signal (filtering only)
    preprocessed_signal = RR.preprocess_signal(raw_signal, fs)
    
    rr, avg_breath_duration = RR.extract_respiration_rate(preprocessed_signal, fs)
    # print(f"Respiration Rate: {rr:.2f} breaths per minute")
    # print(f"Average Breath Duration: {avg_breath_duration:.2f} s")
    
    peaks = NumpySignalProcessor.find_peaks(preprocessed_signal, fs)
    valleys = NumpySignalProcessor.find_peaks(-preprocessed_signal, fs)  # Find valleys as negative peaks
    
    # Detect deep breaths
    deep_breaths = RR.detect_deep_breaths(preprocessed_signal, peaks)
    # print(f"Deep Breaths Detected: {len(deep_breaths)} at indices {deep_breaths}")
    
    # Plot signals with deep breaths
    RR.plot_signals(raw_signal, preprocessed_signal, preprocessed_signal, peaks, valleys, rr, avg_breath_duration, fs, deep_breaths)
