import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.signal import welch, hilbert

from ..utils.signal_processing import NumpySignalProcessor
import os


class ECG:

    @staticmethod
    def detect_r_peaks(filtered_signal, fs, mode="qrs", prominence=None):
        """
        
        Parameters:
        - filtered_signal: The filtered ECG signal.
        - fs: Sampling frequency in Hz.
        - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
        - prominence: Minimum prominence of peaks to detect (default: None).
        
        Returns:
        - r_peaks: Indices of detected R-peaks.
        """
        
        envelope = np.abs(hilbert(filtered_signal))
        smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)

        if mode == "qrs":
            threshold = 0.30 * np.max(envelope)
        elif mode == "all":
            # Calculate a threshold near the maximum value of the signal
            threshold = 0.95 * np.max(envelope)
        else:
            raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
        
        # Ensure the window parameter passed to find_peaks is an integer
        r_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs, threshold=threshold, prominence=prominence)
        
        return np.array(r_peaks, dtype=int)

    @staticmethod
    def validate_r_peaks(envelope, r_peaks, mode="qrs", dynamic_factor=1.5, fixed_threshold=0.8):
        """
        Validates R-peaks dynamically using local envelope statistics for "qrs" mode,
        and a fixed threshold for "all" mode.

        Parameters:
        - envelope: The envelope or smoothed envelope of the ECG signal.
        - r_peaks: Indices of detected R-peaks.
        - mode: Validation mode. "qrs" for dynamic thresholding, "all" for fixed thresholding.
        - dynamic_factor: Multiplier for the dynamic threshold based on local envelope statistics (for "qrs").
        - fixed_threshold: Fraction of the maximum envelope value to use as a fixed threshold (for "all").

        Returns:
        - valid_r_peaks: Indices of validated R-peaks.
        """
        valid_r_peaks = []

        if mode == "qrs":
            # future :Dynamic thresholding based on local envelope statistics
            for idx in r_peaks:
                if idx < 0 or idx >= len(envelope):
                    continue
                # Calculate local mean and standard deviation in a window around the peak
                window_size = 50  # Adjust window size as needed
                start = max(0, idx - window_size)
                end = min(len(envelope), idx + window_size)
                local_mean = np.mean(envelope[start:end])
                local_std = np.std(envelope[start:end])
                dynamic_threshold = local_mean + dynamic_factor * local_std

                # Validate the peak if it exceeds the dynamic threshold
                if envelope[idx] >= dynamic_threshold:
                    valid_r_peaks.append(idx)

        elif mode == "all":
            # Fixed thresholding based on a fraction of the maximum envelope value
            threshold = fixed_threshold * np.max(envelope)
            for idx in r_peaks:
                if idx < 0 or idx >= len(envelope):
                    continue
                # Validate the peak if it exceeds the fixed threshold
                if envelope[idx] >= threshold:
                    valid_r_peaks.append(idx)

        else:
            raise ValueError("Invalid mode. Use 'qrs' for dynamic thresholding or 'all' for fixed thresholding.")

        return np.array(valid_r_peaks)

    @staticmethod
    def preprocess_signal(
        ecg_raw, fs, mode="qrs", 
        bandpass_low=8, bandpass_high=15, 
        envelope_smooth=5, 
        dynamic_factor=1.5, 
        fixed_threshold=0.8, 
        validate_peaks=True, 
        visualization=False
    ):
        """
        Modular pipeline: bandpass, envelope, normalization, smoothing, peak detection, validation.

        Returns:
            normed: normalized and zero-centered bandpassed signal
            smoothed_envelope: smoothed envelope (if visualization is True)
            detected_peaks: indices of detected peaks
            validated_peaks: indices of validated peaks (if validate_peaks is True)
        """
        bandpassed = NumpySignalProcessor.bandpass_filter(ecg_raw, bandpass_low, bandpass_high, fs, order=4)
        normed = NumpySignalProcessor.normalize_signal(bandpassed)
        normed = normed - np.mean(normed)

        smoothed_envelope = None
        if visualization:
            envelope = np.abs(hilbert(normed))
            smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=envelope_smooth)

        detected_peaks = ECG.detect_r_peaks(bandpassed, fs, mode=mode)
        validated_peaks = None
        if validate_peaks:
            envelope = np.abs(hilbert(normed))
            smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=envelope_smooth)
            validated_peaks = ECG.validate_r_peaks(
                smoothed_envelope, detected_peaks, mode=mode, dynamic_factor=dynamic_factor, fixed_threshold=fixed_threshold
            )

        return normed, smoothed_envelope, detected_peaks, validated_peaks

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
    def plot_signals(raw, filtered, r_peaks, heart_rate, fs, hrv_metrics, envelope=None, detected_peaks=None, validated_peaks=None):
        """
        Plots the raw and filtered ECG signals along with detected R-peaks, validated peaks, envelope, PSDs, LF/HF bands, Poincaré plot, and tachogram.
        Displays only the first 10 seconds of the signal.
        All parameters must be precomputed and passed in.
        """
        total_samples = len(raw)
        total_duration = total_samples / fs

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
            p2.plot(time_np, envelope_plot, pen=pg.mkPen(color=(0, 255, 0), width=1.5), name="Envelope")
        # Detected peaks (green circles) on envelope
        if detected_peaks is not None and len(detected_peaks_plot) > 0 and envelope_plot is not None:
            p2.plot(time_np[detected_peaks_plot], envelope_plot[detected_peaks_plot], pen=None, symbol='o', symbolBrush=(0, 255, 0), symbolPen='g', symbolSize=10, name="Detected Peaks")
        # Validated peaks (magenta triangles) on envelope
        if validated_peaks is not None and len(validated_peaks_plot) > 0 and envelope_plot is not None:
            p2.plot(time_np[validated_peaks_plot], envelope_plot[validated_peaks_plot], pen=None, symbol='t', symbolBrush=(255, 0, 255), symbolPen='m', symbolSize=12, name="Validated Peaks")
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p2.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # PSD for all processed plots (not raw)
        f_mv, psd_mv = NumpySignalProcessor.compute_psd_numpy(ECG.convert_adc_to_voltage(raw), fs)
        f_filtered, psd_filtered = NumpySignalProcessor.compute_psd_numpy(filtered, fs)

        p3 = win.addPlot(row=1, col=1, title="<b>Power Spectral Density (PSD) - Processed Signals</b>")
        if len(f_mv) > 0 and len(psd_mv) > 0 and len(f_mv) == len(psd_mv):
            p3.plot(f_mv, psd_mv, pen=pg.mkPen(color=(0, 255, 255), width=1.2), name="Raw mV PSD")
        if len(f_filtered) > 0 and len(psd_filtered) > 0 and len(f_filtered) == len(psd_filtered):
            p3.plot(f_filtered, psd_filtered, pen=pg.mkPen(color=(255, 170, 0), width=2), name="Filtered PSD")
        p3.setLabel('left', "<span style='color:white'>PSD [V**2/Hz]</span>")
        p3.setLabel('bottom', "<span style='color:white'>Frequency [Hz]</span>")
        p3.showGrid(x=True, y=True, alpha=0.3)

        # PSD for filtered signal with LF/HF bands highlighted and area under curve colored
        p4 = win.addPlot(row=2, col=0, title="<b>PSD with LF/HF Bands (Filtered Signal, <0.5Hz)</b>")
        f_psd = hrv_metrics.get("PSD_F", np.array([]))
        psd = hrv_metrics.get("PSD", np.array([]))
        if f_psd is not None and psd is not None and len(f_psd) > 0 and len(psd) > 0:
            mask = f_psd < 0.5
            p4.plot(f_psd[mask], psd[mask], pen=pg.mkPen(color=(255, 170, 0), width=2))
            # Fill LF band
            lf_band = (0.04, 0.15)
            lf_mask = (f_psd >= lf_band[0]) & (f_psd < lf_band[1]) & mask
            if np.any(lf_mask):
                p4.plot(f_psd[lf_mask], psd[lf_mask], pen=None, fillLevel=0, brush=(50, 255, 50, 120))
            # Fill HF band
            hf_band = (0.15, 0.4)
            hf_mask = (f_psd >= hf_band[0]) & (f_psd < hf_band[1]) & mask
            if np.any(hf_mask):
                p4.plot(f_psd[hf_mask], psd[hf_mask], pen=None, fillLevel=0, brush=(50, 50, 255, 120))
        p4.setLabel('left', "<span style='color:white'>PSD [V**2/Hz]</span>")
        p4.setLabel('bottom', "<span style='color:white'>Frequency [Hz]</span>")
        p4.showGrid(x=True, y=True, alpha=0.3)

        # Poincaré plot (HRV)
        rr_intervals = np.diff(r_peaks) / fs  # Use all R-peaks for HRV
        if len(rr_intervals) > 1:
            p5 = win.addPlot(row=2, col=1, title="<b>Poincaré Plot (HRV)</b>")
            p5.plot(rr_intervals[:-1], rr_intervals[1:], pen=None, symbol='o', symbolBrush=(255, 255, 0), symbolSize=6, name="Poincaré Points")
            p5.showGrid(x=True, y=True, alpha=0.3)
            p5.setLabel('left', "<span style='color:white'>RR(n+1) (s)</span>")
            p5.setLabel('bottom', "<span style='color:white'>RR(n) (s)</span>")

        # Tachogram (RR intervals over time) - left Y: RR, right Y: HRV (SDNN)
        if len(rr_intervals) > 0:
            rr_times = np.cumsum(np.insert(rr_intervals, 0, 0))
            p6 = win.addPlot(row=3, col=0, title="<b>Tachogram (RR Intervals & HRV Over Time)</b>")
            # Left Y: RR intervals
            p6.plot(rr_times[1:], rr_intervals, pen=pg.mkPen(color=(255, 255, 0), width=2), symbol='o', symbolBrush=(255, 255, 0), symbolSize=6, name="RR Interval")
            p6.setLabel('left', "<span style='color:white'>RR Interval (s)</span>")
            p6.setLabel('bottom', "<span style='color:white'>Time (s)</span>")
            # Right Y: HRV (SDNN up to each point)
            p6r = pg.ViewBox()
            p6.showAxis('right')
            p6.scene().addItem(p6r)
            p6.getAxis('right').linkToView(p6r)
            p6r.setXLink(p6)
            # Compute rolling SDNN (window=5 by default)
            window = 5
            if len(rr_intervals) >= window:
                rolling_sdnn = np.array([np.std(rr_intervals[max(0, i-window+1):i+1]) for i in range(len(rr_intervals))])
                p6r.addItem(pg.PlotCurveItem(rr_times[1:], rolling_sdnn, pen=pg.mkPen(color=(255, 0, 0), width=2), name="Rolling SDNN"))
                p6.getAxis('right').setLabel("<span style='color:red'>HRV (SDNN, s)</span>")
            else:
                p6.getAxis('right').setLabel("<span style='color:red'>HRV (SDNN, s)</span>")
            p6r.setYRange(0, np.max(rolling_sdnn) if len(rr_intervals) >= window else 1)
            p6r.setGeometry(p6.vb.sceneBoundingRect())
            p6.vb.sigResized.connect(lambda: p6r.setGeometry(p6.vb.sceneBoundingRect()))

        info_text = f"<span style='font-size:10pt'><b>Heart Rate:</b> <span style='color:#ffae00'>{heart_rate:.2f}</span> bpm<br>"
        info_text += f"<b>SDNN:</b> <span style='color:#ffae00'>{hrv_metrics['SDNN']:.2f}</span> s<br>"
        info_text += f"<b>RMSSD:</b> <span style='color:#ffae00'>{hrv_metrics['RMSSD']:.2f}</span> s<br>"
        info_text += f"<b>LF:</b> <span style='color:#ffae00'>{hrv_metrics['LF']:.2f}</span><br>"
        info_text += f"<b>HF:</b> <span style='color:#ffae00'>{hrv_metrics['HF']:.2f}</span><br>"
        info_text += f"<b>LF/HF:</b> <span style='color:#ffae00'>{hrv_metrics['LF/HF']:.2f}</span><br>"
        info_text += f"<b>Sampling Rate:</b> <span style='color:#ffae00'>{fs} Hz</span><br>"
        info_text += f"<b>Signal Duration:</b> <span style='color:#ffae00'>{total_duration:.2f} s</span></span>"
        info_label = pg.LabelItem(info_text, justify='left')
        win.addItem(info_label, row=0, col=2, rowspan=3)

        legend_text = (
            "<b>Legend:</b><br>"
            "<span style='color:#64c8ff'>Raw Signal (ADC, Blue)</span><br>"
            "<span style='color:#00ffff'>Raw Signal (mV, Cyan)</span><br>"
            "<span style='color:#ffaa00'>Processed/Filtered Signal (Orange)</span><br>"
            "<span style='color:#ff5050'>R-Peaks (Red X)</span><br>"
            "<span style='color:#32ff32'>Detected Peaks (Green Circles)</span><br>"
            "<span style='color:#ff00ff'>Validated Peaks (Magenta Triangles)</span><br>"
            "<span style='color:#32ff32'>LF Band (Green Region)</span><br>"
            "<span style='color:#3232ff'>HF Band (Blue Region)</span><br>"
            "<span style='color:#ffff00'>Poincaré Points (Yellow Circles)</span><br>"
            "<span style='color:#ffff00'>Tachogram (Yellow Line)</span>"
        )
        legend_label = pg.LabelItem(legend_text, justify='left', size='10pt')
        win.addItem(legend_label, row=3, col=2, rowspan=2)

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