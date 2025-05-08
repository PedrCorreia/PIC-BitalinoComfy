# filepath: /media/lugo/data/ComfyUI/custom_nodes/PIC_BitalinoComfy/src/ecg_signal_processing.py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.signal import welch

from signal_processing import NumpySignalProcessor
import os


class ECG:

    @staticmethod
    def detect_r_peaks(filtered_signal, fs, mode="qrs"):
        """
        Detects R-peaks in the filtered ECG signal.
        
        Parameters:
        - filtered_signal: The filtered ECG signal.
        - fs: Sampling frequency in Hz.
        - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
        
        Returns:
        - r_peaks: Indices of detected R-peaks.
        """
        if mode == "qrs":
            threshold = 0.8 * np.max(filtered_signal)  # No threshold for QRS mode
        elif mode == "all":
            # Calculate a threshold near the maximum value of the signal
            threshold = 0.8 * np.max(filtered_signal)
        else:
            raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
        
        r_peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs, threshold=threshold)
        return r_peaks

    @staticmethod
    def extract_heart_rate(filtered_signal, fs, mode="qrs"):
        """
        Extracts the heart rate from the filtered ECG signal.
        
        Parameters:
        - filtered_signal: The filtered ECG signal.
        - fs: Sampling frequency in Hz.
        - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
        
        Returns:
        - heart_rate: The calculated heart rate in beats per minute.
        """
        r_peaks = ECG.detect_r_peaks(filtered_signal, fs, mode=mode)
        
        if len(r_peaks) < 2:
            return 0
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        return heart_rate

    @staticmethod
    def calculate_lf_hf(rr_intervals, fs):
        """
        Calculates the Low-Frequency (LF) and High-Frequency (HF) components of HRV.
        
        Parameters:
        - rr_intervals: Array of RR intervals in seconds.
        - fs: Sampling frequency in Hz.
        
        Returns:
        - lf_power: Power in the LF band (0.04–0.15 Hz).
        - hf_power: Power in the HF band (0.15–0.4 Hz).
        """
        f, psd = welch(rr_intervals, fs=1.0 / np.mean(rr_intervals), nperseg=len(rr_intervals))
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        lf_power = np.trapz(psd[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
        hf_power = np.trapz(psd[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])
        return lf_power, hf_power

    @staticmethod
    def calculate_lf_hf_ratio(lf_power, hf_power):
        """
        Calculates the LF/HF ratio.
        
        Parameters:
        - lf_power: Power in the LF band.
        - hf_power: Power in the HF band.
        
        Returns:
        - lf_hf_ratio: The ratio of LF power to HF power.
        """
        return lf_power / hf_power if hf_power > 0 else 0

    @staticmethod
    def calculate_hrv(filtered_signal, fs, mode="qrs"):
        """
        Calculates Heart Rate Variability (HRV) metrics from the filtered ECG signal.
        
        Parameters:
        - filtered_signal: The filtered ECG signal.
        - fs: Sampling frequency in Hz.
        - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
        
        Returns:
        - hrv_metrics: A dictionary containing HRV metrics (e.g., SDNN, RMSSD, LF, HF, LF/HF).
        """
        r_peaks = ECG.detect_r_peaks(filtered_signal, fs, mode=mode)
        
        if len(r_peaks) < 2:
            return {"SDNN": 0, "RMSSD": 0, "LF": 0, "HF": 0, "LF/HF": 0}
        
        rr_intervals = np.diff(r_peaks) / fs  # RR intervals in seconds
        
        # HRV metrics
        sdnn = np.std(rr_intervals)  # Standard deviation of RR intervals
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
        
        # LF and HF power
        lf_power, hf_power = ECG.calculate_lf_hf(rr_intervals, fs)
        lf_hf_ratio = ECG.calculate_lf_hf_ratio(lf_power, hf_power)
        
        return {
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "LF": lf_power,
            "HF": hf_power,
            "LF/HF": lf_hf_ratio
        }

    @staticmethod
    def artifact_removal(signal, threshold=3):
        """
        Removes artifacts from the ECG signal by detecting and replacing outliers.
        
        Parameters:
        - signal: The input ECG signal (array).
        - threshold: The z-score threshold for detecting outliers (default: 3).
        
        Returns:
        - cleaned_signal: The ECG signal with artifacts removed.
        """
        # Calculate the z-score of the signal
        z_scores = (signal - np.mean(signal)) / np.std(signal)
        
        # Identify outliers (absolute z-score > threshold)
        outliers = np.abs(z_scores) > threshold
        
        # Replace outliers with interpolated values
        cleaned_signal = signal.copy()
        if np.any(outliers):
            indices = np.arange(len(signal))
            cleaned_signal[outliers] = np.interp(indices[outliers], indices[~outliers], signal[~outliers])
        
        return cleaned_signal

    @staticmethod
    def pll_artifact_reduction(signal, fs, pll_band=(0.5, 2.0)):
        """
        Applies a Phase-Locked Loop (PLL)-like approach to reduce artifacts in the ECG signal.
        
        Parameters:
        - signal: The input ECG signal (array).
        - fs: Sampling frequency in Hz.
        - pll_band: Frequency band for PLL operation (default: 0.5–2.0 Hz).
        
        Returns:
        - corrected_signal: The artifact-reduced ECG signal.
        """
        from scipy.signal import hilbert

        # Step 1: Bandpass filter the signal to isolate the desired frequency range
        filtered_signal = NumpySignalProcessor.bandpass_filter(signal, pll_band[0], pll_band[1], fs, order=4)

        # Step 2: Apply the Hilbert transform to extract the analytic signal
        analytic_signal = hilbert(filtered_signal)

        # Step 3: Extract the instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Step 4: Generate a synthetic signal based on the phase
        synthetic_signal = np.sin(instantaneous_phase)

        # Step 5: Subtract the synthetic signal from the original to reduce artifacts
        corrected_signal = signal - synthetic_signal

        return corrected_signal

    @staticmethod
    def remove_artifacts(signal, fs, zscore_threshold=5, flatline_threshold=0.05, min_artifact_duration=0.1):
        """
        Removes artifacts from the ECG signal by detecting abnormal segments.
        
        Parameters:
        - signal: The input ECG signal (array).
        - fs: Sampling frequency in Hz.
        - zscore_threshold: Z-score threshold for detecting spikes (default: 5).
        - flatline_threshold: Minimum variance to detect flatline segments (default: 0.05).
        - min_artifact_duration: Minimum duration of an artifact in seconds (default: 0.1).
        
        Returns:
        - cleaned_signal: The ECG signal with artifacts removed.
        """
        # Step 1: Detect spikes using z-score
        z_scores = (signal - np.mean(signal)) / np.std(signal)
        spike_mask = np.abs(z_scores) > zscore_threshold

        # Step 2: Detect flatline segments using variance
        window_size = int(min_artifact_duration * fs)
        flatline_mask = np.zeros_like(signal, dtype=bool)
        for start in range(0, len(signal), window_size):
            end = start + window_size
            segment = signal[start:end]
            if np.var(segment) < flatline_threshold:
                flatline_mask[start:end] = True

        # Combine masks for artifacts
        artifact_mask = spike_mask | flatline_mask

        # Debugging: Print the percentage of points marked as artifacts
        artifact_percentage = np.sum(artifact_mask) / len(signal) * 100
        print(f"Artifact Percentage: {artifact_percentage:.2f}%")

        # Step 3: Replace artifacts with interpolated values
        cleaned_signal = signal.copy()
        if np.any(artifact_mask):
            indices = np.arange(len(signal))
            valid_indices = indices[~artifact_mask]
            if valid_indices.size == 0:
                # If no valid points are left, return the original signal with a warning
                print("Warning: All points in the signal are marked as artifacts. Returning the original signal.")
                return signal
            cleaned_signal[artifact_mask] = np.interp(indices[artifact_mask], valid_indices, signal[~artifact_mask])

        return cleaned_signal

    @staticmethod
    def preprocess_signal(signal, fs, mode="all"):
        """
        Preprocesses the ECG signal by removing artifacts, filtering, and normalizing.
        
        Parameters:
        - signal: The input ECG signal (array).
        - fs: Sampling frequency in Hz.
        - mode: Preprocessing mode. "all" for the entire heart complex, "qrs" for QRS complex only.
        
        Returns:
        - preprocessed_signal: The preprocessed ECG signal.
        """


        # Step 1: Remove artifacts from the signal
        artifact_free_signal = signal

        if mode == "all":
            # Step 2: Filtering with a bandpass filter to isolate heart-related frequencies (0.5–40 Hz)
            filtered_signal = NumpySignalProcessor.bandpass_filter(artifact_free_signal, 0.5, 40, fs, order=4)
        elif mode == "qrs":
            # Step 2: Filtering with a bandpass filter to isolate QRS complex frequencies (8–20 Hz)
            filtered_signal = NumpySignalProcessor.bandpass_filter(artifact_free_signal, 8, 15, fs, order=4)
        else:
            raise ValueError("Invalid mode. Use 'all' for the entire heart complex or 'qrs' for QRS complex only.")

        # Step 3: Smoothing the filtered signal
        smoothed_signal = NumpySignalProcessor.moving_average(filtered_signal, window_size=5)

        # Step 4: Normalization
        normalized_signal = NumpySignalProcessor.normalize_signal(smoothed_signal)
        
        return normalized_signal

    @staticmethod
    def convert_adc_to_voltage(adc_values, channel_index=0, vcc=3.3, gain=1100):
        """
        Converts ADC values to ECG voltage in millivolts, considering channel resolution.
        
        Parameters:
        - adc_values: Array of ADC values.
        - channel_index: Index of the channel (default: 0).
        - vcc: Operating voltage of the system (default: 3.3V).
        - gain: Gain of the ECG sensor (default: 1100).
        
        Returns:
        - ecg_mv: ECG signal in millivolts.
        """
        # Determine resolution based on channel index
        n_bits = 10 if channel_index < 4 else 6
        
        # Convert ADC values to voltage
        ecg_voltage = (adc_values / (2**n_bits - 1) - 0.5) * vcc / gain
        
        # Convert voltage to millivolts
        ecg_mv = ecg_voltage * 1000
        return ecg_mv

    @staticmethod
    def plot_signals(raw, artifact_removed, filtered, r_peaks, heart_rate, fs, hrv_metrics):
        """
        Plots the raw, artifact-removed, and filtered ECG signals along with detected R-peaks, PSD, and Poincaré plot.
        Displays only the first 10 seconds of the signal.
        """
        # Limit the signal to the first 10 seconds
        max_samples = int(10 * fs)  # Number of samples for 10 seconds
        raw = raw[:max_samples]
        artifact_removed = artifact_removed[:max_samples]
        filtered = filtered[:max_samples]
        time_np = np.arange(len(raw)) / fs  # Time array for the limited window

        # Filter R-peaks to the 10-second window
        r_peaks = r_peaks[r_peaks < max_samples]

        # Calculate the duration of the signal
        duration = len(raw) / fs

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="ECG Signal Analysis")
        win.resize(1800, 1000)  # Larger window size for a scientific layout
        win.setWindowTitle("ECG Signal Analysis")

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # Calculate PSD for all signals
        f_raw, psd_raw = welch(raw, fs, nperseg=1024)
        f_artifact_removed, psd_artifact_removed = welch(artifact_removed, fs, nperseg=1024)
        f_filtered, psd_filtered = welch(filtered, fs, nperseg=1024)
        # Apply baseline correction using NumpySignalProcessor
        baseline_corrected = NumpySignalProcessor.correct_baseline(filtered, method="als", lam=1e6, p=0.01, niter=10)

        f_baseline_corrected, psd_baseline_corrected = welch(baseline_corrected, fs, nperseg=1024)

        # Normalize PSDs for comparison
        psd_raw /= np.max(psd_raw)
        psd_artifact_removed /= np.max(psd_artifact_removed)
        psd_filtered /= np.max(psd_filtered)
        psd_baseline_corrected /= np.max(psd_baseline_corrected)

        # Row 0: Raw and Artifact-Removed Signals
        p1 = win.addPlot(row=0, col=0, title="<b>Raw ECG Signal</b>")
        p1.plot(time_np, raw, pen=pg.mkPen(color=(100, 200, 255), width=1.2), name="Raw Signal")
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p1.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        p2 = win.addPlot(row=0, col=1, title="<b>Artifact-Removed ECG Signal</b>")
        p2.plot(time_np, artifact_removed, pen=pg.mkPen(color=(255, 255, 0), width=1.2), name="Artifact-Removed Signal")
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p2.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Row 1: Filtered Signal and PSD
        p3 = win.addPlot(row=1, col=0, title="<b>Filtered ECG Signal with R-Peaks</b>")
        p3.plot(time_np, filtered, pen=pg.mkPen(color=(255, 170, 0), width=2), name="Filtered Signal")
        if len(r_peaks) > 0:
            p3.plot(time_np[r_peaks], filtered[r_peaks], pen=None, symbol='x', symbolBrush=(255, 80, 80), symbolPen='r', symbolSize=14, name="R-Peaks")
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p3.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        p4 = win.addPlot(row=1, col=1, title="<b>Power Spectral Density (PSD)</b>")
        p4.plot(f_raw, psd_raw, pen=pg.mkPen(color=(100, 200, 255), width=1.2), name="Raw Signal PSD")
        p4.plot(f_artifact_removed, psd_artifact_removed, pen=pg.mkPen(color=(255, 255, 0), width=1.2), name="Artifact-Removed PSD")
        p4.plot(f_filtered, psd_filtered, pen=pg.mkPen(color=(255, 170, 0), width=2), name="Filtered PSD")
        p4.plot(f_baseline_corrected, psd_baseline_corrected, pen=pg.mkPen(color=(128, 0, 128), width=1.5), name="Baseline-Corrected PSD")
        p4.showGrid(x=True, y=True, alpha=0.3)
        p4.setLabel('left', "<span style='color:white'>Normalized Power</span>")
        p4.setLabel('bottom', "<span style='color:white'>Frequency (Hz)</span>")

        # Row 2: Baseline-Corrected Signal and Poincaré Plot
        p5 = win.addPlot(row=2, col=0, title="<b>Baseline-Corrected Filtered Signal</b>")
        p5.plot(time_np, baseline_corrected, pen=pg.mkPen(color=(128, 0, 128), width=2), name="Baseline-Corrected Signal")
        p5.showGrid(x=True, y=True, alpha=0.3)
        p5.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p5.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        rr_intervals = np.diff(r_peaks) / fs  # RR intervals in seconds
        if len(rr_intervals) > 1:
            p6 = win.addPlot(row=2, col=1, title="<b>Poincaré Plot (HRV)</b>")
            p6.plot(rr_intervals[:-1], rr_intervals[1:], pen=None, symbol='o', symbolBrush=(255, 255, 0), symbolSize=6, name="Poincaré Points")
            p6.showGrid(x=True, y=True, alpha=0.3)
            p6.setLabel('left', "<span style='color:white'>RR(n+1) (s)</span>")
            p6.setLabel('bottom', "<span style='color:white'>RR(n) (s)</span>")

        # Side Column: Relevant Information and Legend
        info_text = f"<span style='font-size:10pt'><b>Heart Rate:</b> <span style='color:#ffae00'>{heart_rate:.2f}</span> bpm<br>"
        info_text += f"<b>SDNN:</b> <span style='color:#ffae00'>{hrv_metrics['SDNN']:.2f}</span> s<br>"
        info_text += f"<b>RMSSD:</b> <span style='color:#ffae00'>{hrv_metrics['RMSSD']:.2f}</span> s<br>"
        info_text += f"<b>LF:</b> <span style='color:#ffae00'>{hrv_metrics['LF']:.2f}</span><br>"
        info_text += f"<b>HF:</b> <span style='color:#ffae00'>{hrv_metrics['HF']:.2f}</span><br>"
        info_text += f"<b>LF/HF:</b> <span style='color:#ffae00'>{hrv_metrics['LF/HF']:.2f}</span><br>"
        info_text += f"<b>Sampling Rate:</b> <span style='color:#ffae00'>{fs} Hz</span><br>"
        info_text += f"<b>Signal Duration:</b> <span style='color:#ffae00'>{duration:.2f} s</span></span>"
        info_label = pg.LabelItem(info_text, justify='left')
        win.addItem(info_label, row=0, col=2, rowspan=3)

        legend_text = (
            "<b>Legend:</b><br>"
            "<span style='color:#64c8ff'>Raw Signal (Blue)</span><br>"
            "<span style='color:#ffff00'>Artifact-Free Signal (Yellow)</span><br>"
            "<span style='color:#ffaa00'>Filtered Signal (Orange)</span><br>"
            "<span style='color:#ff5050'>R-Peaks (Red X)</span><br>"
            "<span style='color:#800080'>Baseline-Corrected Signal (Purple)</span><br>"
            "<span style='color:#ffff00'>Poincaré Points (Yellow Circles)</span>"
        )
        legend_label = pg.LabelItem(legend_text, justify='left', size='10pt')
        win.addItem(legend_label, row=3, col=2, rowspan=2)  # Place legend in the side column

        app.exec()

def demo():
    # Allow the user to select electrode placement
    placement = input("Enter electrode placement ('heart' or 'collarbone'): ").strip().lower()
    if placement not in ["heart", "collarbone"]:
        print("Invalid placement. Please enter 'heart' or 'collarbone'.")
        return

    # Dynamically select the JSON file based on placement
    file_name = "heart_signal_data.json" if placement == "heart" else "collarbone_signal_data.json"
    file_path = os.path.join(os.path.dirname(__file__), "ECG", file_name)
    raw_signal = NumpySignalProcessor.load_signal(file_path)  # Use NumpySignalProcessor to load the signal

    fs = 1000
    # Convert ADC values to millivolts
    ecg_mv = ECG.convert_adc_to_voltage(raw_signal)

    # Allow the user to select the mode ("all" or "qrs")
    mode = input("Enter preprocessing mode ('all' for entire heart complex, 'qrs' for QRS complex only): ").strip().lower()
    if mode not in ["all", "qrs"]:
        print("Invalid mode. Please enter 'all' or 'qrs'.")
        return

    # Preprocess the signal based on the selected mode
    preprocessed_signal = ECG.preprocess_signal(ecg_mv, fs, mode=mode)

    # Extract heart rate
    heart_rate = ECG.extract_heart_rate(preprocessed_signal, fs, mode=mode)
    print(f"Heart Rate: {heart_rate:.2f} bpm")

    # Calculate HRV metrics
    hrv_metrics = ECG.calculate_hrv(preprocessed_signal, fs, mode=mode)
    print(f"HRV Metrics: SDNN = {hrv_metrics['SDNN']:.2f} s, RMSSD = {hrv_metrics['RMSSD']:.2f} s, LF = {hrv_metrics['LF']:.2f}, HF = {hrv_metrics['HF']:.2f}, LF/HF = {hrv_metrics['LF/HF']:.2f}")

    # Detect R-peaks
    r_peaks = ECG.detect_r_peaks(preprocessed_signal, fs, mode=mode)

    # Plot signals with HRV metrics
    ECG.plot_signals(raw_signal, preprocessed_signal, preprocessed_signal, r_peaks, heart_rate, fs, hrv_metrics)

if __name__ == "__main__":
    demo()