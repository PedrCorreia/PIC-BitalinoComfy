from pyqtgraph.Qt import QtWidgets
from scipy.signal import welch, hilbert
import numpy as np
import pyqtgraph as pg
import os

from ..utils.signal_processing import NumpySignalProcessor


class ECG:
    @staticmethod
    def detect_r_peaks(filtered_signal, fs, mode="qrs", prominence=None):
        """
        Returns detected R-peaks using adaptive thresholding.
        
        Dynamic threshold adapts to:
        1. Recent signal history using multiple window sizes
        2. Quickly responds to amplitude changes
        3. Handles both surges and drops in signal strength
        """
        # Get signal length for window calculations
        signal_length = len(filtered_signal)
        
        # Use multiple window sizes for more robust detection
        short_window = min(int(fs * 0.5), signal_length)  # 0.5 second (recent activity)
        med_window = min(int(fs * 2), signal_length)      # 2 seconds (medium history)
        
        # Compute thresholds at different time scales
        if signal_length > short_window:
            # Recent window threshold (responds quickly to surges)
            recent_max = np.max(filtered_signal[-short_window:])
            recent_mean = np.mean(np.abs(filtered_signal[-short_window:]))
            short_threshold = 0.5 * recent_max + 0.25 * recent_mean
            
            # Medium window threshold (more stable)
            med_max = np.max(filtered_signal[-med_window:])
            med_mean = np.mean(np.abs(filtered_signal[-med_window:]))
            med_threshold = 0.4 * med_max + 0.3 * med_mean
            
            # Combine thresholds with bias toward recent activity for faster response
            adaptive_threshold = 0.7 * short_threshold + 0.3 * med_threshold
        else:
            # Fallback for very short signals
            adaptive_threshold = 0.5 * np.max(filtered_signal)
        
        # Use a minimum threshold to avoid detecting noise when signal is weak
        min_threshold = 0.2 * np.max(filtered_signal)
        final_threshold = max(adaptive_threshold, min_threshold)
        
        # Apply threshold to find peaks
        r_peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs, threshold=final_threshold, prominence=prominence)
        r_peaks = ECG.validate_r_peaks(filtered_signal, r_peaks)
        
        return np.array(r_peaks, dtype=int)

    @staticmethod
    def validate_r_peaks(filtered_signal, detected_peaks, lag=100, match_window=30):
        """
        Validate filtered-signal peaks only if they are near envelope peaks and not within lag samples of the end.
        - lag: number of samples at the end to ignore for validation (to avoid edge artifacts)
        - match_window: max distance (samples) to consider a filtered peak as matching an envelope peak
        Returns validated peak indices (subset of detected_peaks).
        """
        # Calculate signal envelope with optimized parameters for surge detection
        envelope = np.abs(hilbert(filtered_signal))
        smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
        
        # Use adaptive threshold for envelope peaks as well
        env_threshold = 0.5 * np.mean(smoothed_envelope) + 0.2 * np.max(smoothed_envelope)
        env_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs=1, 
                                                   threshold=env_threshold,
                                                   window=match_window, 
                                                   prominence=None)
        
        # Only keep detected peaks that are close to envelope peaks and not in lag region
        valid_peaks = []
        for idx in detected_peaks:
            # Skip peaks in lag region
            if idx >= len(filtered_signal) - lag:
                continue
                
            # Match peak to envelope peak with more lenient window during surges
            # During signal surges (high amplitude), use wider window for matching
            local_amplitude = np.max(filtered_signal[max(0, idx-30):min(len(filtered_signal), idx+30)])
            global_amplitude = np.max(filtered_signal)
            is_surge = local_amplitude > 0.8 * global_amplitude
            
            # Adjust matching window during surges to be more permissive
            actual_match_window = match_window * 1.5 if is_surge else match_window
                
            if np.any(np.abs(env_peaks - idx) <= actual_match_window):
                valid_peaks.append(idx)
                
        return np.array(valid_peaks, dtype=int)

    @staticmethod
    def preprocess_signal(
        ecg_raw, fs, mode="qrs", 
        bandpass_low=8, bandpass_high=15, 
        envelope_smooth=5, 
        dynamic_factor=1.5, 
        fixed_threshold=0.8, 
        validate_peaks=True, 
        visualization=False,
        lag=100,  # samples
        match_window=30  # samples
    ):
        """
        Modular pipeline: bandpass, envelope, normalization, smoothing, peak detection, validation.
        Returns:
            normed: normalized and zero-centered bandpassed signal
            smoothed_envelope: smoothed envelope (if visualization is True)
            detected_peaks: indices of detected peaks
            validated_peaks: indices of validated peaks (if validate_peaks is True)
        """
        bandpassed = NumpySignalProcessor.bandpass_filter(ecg_raw, bandpass_low, bandpass_high, fs, order=2)
        normed = NumpySignalProcessor.normalize_signal(bandpassed)
        detected_peaks, smoothed_envelope = ECG.detect_r_peaks(normed, fs, mode=mode)
       

        return normed, detected_peaks

    @staticmethod
    def extract_heart_rate(signal, fs, mode="qrs", r_peaks=None):
        """
        Extracts the heart rate from the filtered ECG signal or from provided r_peaks.
        """
        if r_peaks is None:
            r_peaks = ECG.detect_r_peaks(signal, fs, mode=mode)
        if len(r_peaks) < 2:
            return 0
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        return heart_rate

    @staticmethod
    def calculate_lf_hf(rr_intervals, fs):
        """
        Calculates the Low-Frequency (LF) and High-Frequency (HF) power of HRV from RR intervals.

        Parameters:
        - rr_intervals: Array of RR intervals in seconds (can be derived from HR).
        - fs: Sampling frequency in Hz (used for interpolation).

        Steps:
        - If input is HR (heart rate in bpm), convert to RR intervals (RR = 60 / HR).
        - Interpolate RR intervals to a uniform time grid (e.g., 4 Hz).
        - Compute PSD and extract LF/HF bands.

        Returns:
        - lf_power: Power in the LF band (0.04–0.15 Hz).
        - hf_power: Power in the HF band (0.15–0.4 Hz).
        - f: Frequency array for PSD.
        - psd: PSD array.
        """
        # Interpolate RR intervals to get evenly sampled signal
        if len(rr_intervals) < 4:
            return 0, 0, np.array([]), np.array([])
        rr_times = np.cumsum(np.insert(rr_intervals, 0, 0))
        interp_fs = 4.0  # 4 Hz is standard for HRV
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_fs)
        rr_interp = np.interp(t_uniform, rr_times[:-1], rr_intervals)
        n = len(rr_interp)
        nperseg = min(256, n)
        noverlap = min(128, nperseg - 1) if nperseg > 1 else 0
        f, psd = NumpySignalProcessor.compute_psd_numpy(rr_interp, fs=interp_fs, nperseg=nperseg, noverlap=noverlap)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        lf_mask = (f >= lf_band[0]) & (f < lf_band[1])
        hf_mask = (f >= hf_band[0]) & (f < hf_band[1])
        lf_power = np.trapz(psd[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0
        return lf_power, hf_power, f, psd

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
    def calculate_hrv(signal, fs, mode="qrs", r_peaks=None):
        """
        Calculates HRV metrics from the filtered ECG signal or from provided r_peaks.
        
        Parameters:
        - signal: The filtered ECG signal.
        - fs: Sampling frequency in Hz.
        - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
        - r_peaks: Indices of detected R-peaks (optional).
        
        Returns:
        - hrv_metrics: A dictionary containing HRV metrics (e.g., SDNN, RMSSD, LF, HF, LF/HF).
        """
        if r_peaks is None:
            r_peaks = ECG.detect_r_peaks(signal, fs, mode=mode)
        
        if len(r_peaks) < 2:
            return {"SDNN": 0, "RMSSD": 0, "LF": 0, "HF": 0, "LF/HF": 0, "PSD_F": np.array([]), "PSD": np.array([])}
        
        rr_intervals = np.diff(r_peaks) / fs  # RR intervals in seconds
        
        # HRV metrics
        sdnn = np.std(rr_intervals)  # Standard deviation of RR intervals
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
        
        # LF and HF power
        lf_power, hf_power, f_psd, psd = ECG.calculate_lf_hf(rr_intervals, fs)
        lf_hf_ratio = ECG.calculate_lf_hf_ratio(lf_power, hf_power)
        
        return {
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "LF": lf_power,
            "HF": hf_power,
            "LF/HF": lf_hf_ratio,
            "PSD_F": f_psd,
            "PSD": psd
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
    def convert_adc_to_voltage(adc_values, channel_index=0, vcc=3.3, gain=1100):
        """
        Converts ADC values to ECG voltage in millivolts, considering channel resolution.

        Formula:
        ECG(V) = ((ADC / (2^n - 1)) - 0.5) * VCC / GECG
        ECG(mV) = ECG(V) * 1000

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

        # ECG(V) = ((ADC / (2^n - 1)) - 0.5) * VCC / GECG
        ecg_voltage = (adc_values / (2**n_bits - 1) - 0.5) * vcc / gain

        # ECG(mV) = ECG(V) * 1000
        ecg_mv = ecg_voltage * 1000
        return ecg_mv

    @staticmethod
    def plot_signals(raw, filtered, r_peaks, heart_rate, fs, hrv_metrics, envelope=None, detected_peaks=None, validated_peaks=None, lag=100):
        """
        Plots the raw and filtered ECG signals along with detected R-peaks, validated peaks, envelope, and shades lag region.
        """
        total_samples = len(raw)
        max_samples = int(10 * fs)
        plot_slice = slice(0, max_samples)
        raw_plot = raw[plot_slice]
        filtered_plot = filtered[plot_slice]
        time_np = np.arange(len(raw_plot)) / fs
        raw_mv_plot = ECG.convert_adc_to_voltage(raw_plot)
        r_peaks_plot = r_peaks[r_peaks < max_samples] if r_peaks is not None else np.array([])
        detected_peaks_plot = detected_peaks[detected_peaks < max_samples] if detected_peaks is not None else np.array([])
        validated_peaks_plot = validated_peaks[validated_peaks < max_samples] if validated_peaks is not None else np.array([])
        envelope_plot = envelope[plot_slice] if envelope is not None else None
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="ECG Signal Analysis")
        win.resize(1800, 1200)
        win.setWindowTitle("ECG Signal Analysis")
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        # Raw signal in ADC units
        p1 = win.addPlot(row=0, col=0, title="<b>Raw ECG Signal (ADC Units)</b>")
        p1.plot(time_np, raw_plot, pen=pg.mkPen(color=(100, 200, 255), width=1.2), name="Raw Signal (ADC)")
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel('left', "<span style='color:white'>ADC Value</span>")
        p1.setLabel('bottom', "<span style='color:white'>Time (s)</span>")
        # Raw signal in millivolts
        p1_mv = win.addPlot(row=0, col=1, title="<b>Raw ECG Signal (mV)</b>")
        p1_mv.plot(time_np, raw_mv_plot, pen=pg.mkPen(color=(0, 255, 255), width=1.2), name="Raw Signal (mV)")
        p1_mv.showGrid(x=True, y=True, alpha=0.3)
        p1_mv.setLabel('left', "<span style='color:white'>Amplitude (mV)</span>")
        p1_mv.setLabel('bottom', "<span style='color:white'>Time (s)</span>")
        # Filtered/processed signal with R-peaks, detected peaks, validated peaks, and envelope
        p2 = win.addPlot(row=1, col=0, title="<b>Processed ECG Signal with Peaks and Envelope</b>")
        p2.plot(time_np, filtered_plot, pen=pg.mkPen(color=(255, 170, 0), width=2), name="Filtered Signal")
        # Envelope
        if envelope_plot is not None:
            p2.plot(time_np, envelope_plot, pen=pg.mkPen(color=(0, 255, 0), width=1), name="Envelope")
        # Detected peaks (green circles)
        if detected_peaks is not None and len(detected_peaks_plot) > 0 and envelope_plot is not None:
            p2.plot(time_np[detected_peaks_plot], envelope_plot[detected_peaks_plot], pen=None, symbol='o', symbolBrush=(0,255,0), symbolSize=10, name="Detected Peaks")
        # Validated peaks (magenta triangles)
        if validated_peaks is not None and len(validated_peaks_plot) > 0 and envelope_plot is not None:
            p2.plot(time_np[validated_peaks_plot], envelope_plot[validated_peaks_plot], pen=None, symbol='t', symbolBrush=(255,0,255), symbolSize=12, name="Validated Peaks")
        # Shade lag region at the end
        if lag > 0:
            lag_start = (len(filtered_plot) - lag) / fs
            lag_end = len(filtered_plot) / fs
            p2.addItem(pg.LinearRegionItem([lag_start, lag_end], brush=(100,100,100,80), movable=False))
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p2.setLabel('bottom', "<span style='color:white'>Time (s)</span>")
        # PSD for all processed plots (not raw)
        f_mv, psd_mv = NumpySignalProcessor.compute_psd_numpy(ECG.convert_adc_to_voltage(raw), fs)
        f_filtered, psd_filtered = NumpySignalProcessor.compute_psd_numpy(filtered, fs)
        p3 = win.addPlot(row=1, col=1, title="<b>Power Spectral Density (PSD) - Processed Signals</b>")
        if f_mv is not None and psd_mv is not None and len(f_mv) > 0 and len(psd_mv) > 0 and len(f_mv) == len(psd_mv):
            p3.plot(f_mv, psd_mv, pen=pg.mkPen(color=(0,255,255), width=1.2), name="Raw PSD")
        if f_filtered is not None and psd_filtered is not None and len(f_filtered) > 0 and len(psd_filtered) > 0 and len(f_filtered) == len(psd_filtered):
            p3.plot(f_filtered, psd_filtered, pen=pg.mkPen(color=(255,170,0), width=1.2), name="Filtered PSD")
        p3.setLabel('left', "<span style='color:white'>PSD [V**2/Hz]</span>")
        p3.setLabel('bottom', "<span style='color:white'>Frequency [Hz]</span>")
        p3.showGrid(x=True, y=True, alpha=0.3)
        # Show the window
        win.show()

def demo():
    # Allow the user to select electrode placement
    placement = input("Enter electrode placement ('heart' or 'collarbone'): ").strip().lower()
    if placement not in ["heart", "collarbone"]:
        print("Invalid placement. Please enter 'heart' or 'collarbone'.")
        return

    # Dynamically select the JSON file based on placement
    file_name = "heart_signal_data.json" if placement == "heart" else "collarbone_signal_data.json"
    file_path = os.path.join(os.path.dirname(__file__), "ECG", file_name)
    raw_signal = NumpySignalProcessor.load_signal(file_path)

    fs = 1000

    # Always use Bandpass+Peaks method for demo
    print("Using Bandpass+Peaks method for demo...")

    # Modular pipeline
    normed, smoothed_envelope, detected_peaks, validated_peaks = ECG.preprocess_signal(
        raw_signal, fs, mode="qrs",
        bandpass_low=8, bandpass_high=15,
        envelope_smooth=5,
        dynamic_factor=1.5,
        fixed_threshold=0.8,
        validate_peaks=True
    )

    # Use validated peaks for HR/HRV if available, else detected
    r_peaks_for_metrics = validated_peaks if validated_peaks is not None and len(validated_peaks) > 1 else detected_peaks

    heart_rate = ECG.extract_heart_rate(normed, fs, mode="qrs", r_peaks=r_peaks_for_metrics)
    print(f"Heart Rate: {heart_rate:.2f} bpm")

    hrv_metrics = ECG.calculate_hrv(normed, fs, mode="qrs", r_peaks=r_peaks_for_metrics)
    print(f"HRV Metrics: SDNN = {hrv_metrics['SDNN']:.2f} s, RMSSD = {hrv_metrics['RMSSD']:.2f} s, LF = {hrv_metrics['LF']:.2f}, HF = {hrv_metrics['HF']:.2f}, LF/HF = {hrv_metrics['LF/HF']:.2f}")

    ECG.plot_signals(
        raw_signal,
        normed,
        r_peaks_for_metrics,
        heart_rate,
        fs,
        hrv_metrics,
        envelope=smoothed_envelope,
        detected_peaks=detected_peaks,
        validated_peaks=validated_peaks
    )

if __name__ == "__main__":
    demo()