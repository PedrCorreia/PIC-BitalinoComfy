import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.signal import hilbert
from ..utils.signal_processing import NumpySignalProcessor

import os
import time

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
        filtered_signal = NumpySignalProcessor.bandpass_filter(signal, 0.1, 1, fs, order=1)

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

    @staticmethod
    def is_peak(filtered_rr, feature_timestamps, fs, last_peak_time=None, epsilon=None, start_time=None, rr=None, used_peaks=None):
        """
        Returns True if the latest detected peak is within epsilon seconds of the current time,
        and the peak has not been used before (not in used_peaks).
        used_peaks: set or list of previously used peak times (in seconds).
        """
        # Detect peaks and validate them with lag-based edge avoidance
        detected_peaks = NumpySignalProcessor.find_peaks(filtered_rr, fs=fs)
        validated_peaks =detected_peaks    #RR.validate_rr_peaks(filtered_rr, detected_peaks, lag=50, match_window=20)
        
        if isinstance(validated_peaks, np.ndarray) and len(validated_peaks) > 0 and len(feature_timestamps) > 0:
            peak_times = feature_timestamps[validated_peaks]
            latest_peak_time = peak_times[-1]
            # Estimate RR from breath intervals (if possible)
            if len(peak_times) > 1:
                breath_intervals = np.diff(peak_times)
                avg_breath = np.mean(breath_intervals)
                rr_est = 60.0 / avg_breath if avg_breath > 0 else 12.0
            else:
                rr_est = 12.0  # fallback default (12 breaths/min)
            rr_used = rr if rr is not None else rr_est
            if epsilon is None:
                epsilon = 0.5
            if start_time is not None:
                now = time.time() - start_time
            else:
                now = feature_timestamps[-1]
            is_in_time_neighborhood = abs(now - latest_peak_time) <= epsilon
            # Check if this peak has been used before
            is_new_peak = True
            if used_peaks is not None:
                is_new_peak = not any(abs(latest_peak_time - t) < 1e-4 for t in used_peaks)
            # print(f"[RR.is_peak] start_time: {start_time}, now: {now}, latest_peak_time: {latest_peak_time}, RR: {rr_used:.2f}, epsilon: {epsilon:.3f}, is_in_time_neighborhood: {is_in_time_neighborhood}, is_new_peak: {is_new_peak}")
            return (is_in_time_neighborhood and is_new_peak), latest_peak_time
        return False, None
    
    @staticmethod
    def validate_rr_peaks(filtered_signal, detected_peaks, lag=50, match_window=20):
        """
        Validate RR peaks by ensuring they are not within lag samples of the end and match signal envelope.
        - lag: number of samples at the end to ignore for validation (to avoid edge artifacts)
        - match_window: max distance (samples) to consider a filtered peak as matching an envelope peak
        Returns validated peak indices (subset of detected_peaks).
        """
        # Calculate signal envelope for RR using hilbert transform
        envelope = np.abs(hilbert(filtered_signal))
        smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=3)
        
        # Use adaptive threshold for envelope peaks
        env_threshold = 0.4 * np.mean(smoothed_envelope) + 0.3 * np.max(smoothed_envelope)
        env_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs=1, 
                                                   threshold=env_threshold,
                                                   window=match_window, 
                                                   prominence=None)
        
        # Only keep detected peaks that are close to envelope peaks and not in lag region
        valid_peaks = []
        for idx in detected_peaks:
            # Skip peaks in lag region (edge avoidance)
            if idx >= len(filtered_signal) - lag:
                continue
                
            # Match peak to envelope peak
            if np.any(np.abs(env_peaks - idx) <= match_window):
                valid_peaks.append(idx)
                
        return np.array(valid_peaks, dtype=int)

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "RR", "signal_data.json")
    raw_signal = NumpySignalProcessor.load_signal(file_path)  # Use NumpySignalProcessor to load the signal
    
    fs = 1000
    # Preprocess the signal (filtering only)
    preprocessed_signal = RR.preprocess_signal(raw_signal, fs)
    
    rr, avg_breath_duration = RR.extract_respiration_rate(preprocessed_signal, fs)
    print(f"Respiration Rate: {rr:.2f} breaths per minute")
    print(f"Average Breath Duration: {avg_breath_duration:.2f} s")
    
    peaks = NumpySignalProcessor.find_peaks(preprocessed_signal, fs)
    valleys = NumpySignalProcessor.find_peaks(-preprocessed_signal, fs)  # Find valleys as negative peaks
    
    # Detect deep breaths
    deep_breaths = RR.detect_deep_breaths(preprocessed_signal, peaks)
    print(f"Deep Breaths Detected: {len(deep_breaths)} at indices {deep_breaths}")
    
    # Plot signals with deep breaths
    RR.plot_signals(raw_signal, preprocessed_signal, preprocessed_signal, peaks, valleys, rr, avg_breath_duration, fs, deep_breaths)
