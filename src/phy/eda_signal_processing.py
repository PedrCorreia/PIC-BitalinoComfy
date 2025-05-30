import numpy as np
from ..utils.signal_processing import NumpySignalProcessor


class EDA:

    @staticmethod
    def convert_adc_to_eda(adc_values, channel_index=0, vcc=3.3):
        """
        Converts ADC values to EDA in micro-Siemens (μS).
        
        Parameters:
        - adc_values: Array of ADC values.
        - channel_index: Index of the channel (default: 0).
        - vcc: Operating voltage of the system (default: 3.3V).
        
        Returns:
        - eda_us: EDA signal in micro-Siemens (μS).
        """
        n_bits = 10 if channel_index < 4 else 6  # Determine resolution based on channel index
        eda_us = (adc_values / (2**n_bits) * vcc) / 0.132  # Transfer function for EDA in μS
        return eda_us


    @staticmethod
    def extract_tonic_phasic(eda_signal, fs):
        """
        Extracts tonic and phasic components from the EDA signal.
        
        Parameters:
        - eda_signal: The EDA signal in micro-Siemens (μS).
        - fs: Sampling frequency in Hz.
        
        Returns:
        - tonic: Tonic component of the EDA signal.
        - phasic: Phasic component of the EDA signal.
        """
        # Baseline correction for tonic component
        tonic = NumpySignalProcessor.lowpass_filter(eda_signal, cutoff=0.1, fs=fs)
        
        # Phasic component is the difference between the original signal and tonic
        phasic = eda_signal - tonic
        
        return tonic, phasic

    @staticmethod
    def detect_events(phasic_signal, fs, threshold=0.01):
        """
        Detects events in the phasic component of the EDA signal using robust peak finding.
        
        Parameters:
        - phasic_signal: The phasic component of the EDA signal.
        - fs: Sampling frequency in Hz.
        - threshold: Minimum peak amplitude (default: 0.01 μS).
        
        Returns:
        - events: Indices of detected events (peaks).
        """
        # Use NumpySignalProcessor.find_peaks for robust detection
        events = NumpySignalProcessor.find_peaks(phasic_signal, fs=fs, threshold=threshold)
        return events

    @staticmethod
    def validate_events(phasic_signal, events, envelope_smooth=5, envelope_threshold=0.5, amplitude_proximity=0.1):
        """
        Validates EDA events using a smoothed envelope and amplitude thresholding (similar to ECG peak validation).

        Parameters:
        - phasic_signal: The phasic component of the EDA signal.
        - events: Indices of detected events.
        - envelope_smooth: Window size for smoothing the envelope.
        - envelope_threshold: Fraction of the maximum envelope value to use as a validation threshold.
        - amplitude_proximity: Maximum allowed difference (fraction of envelope max) between event amplitude and local envelope maximum.

        Returns:
        - valid_events: Indices of validated events.
        - smoothed_envelope: The smoothed envelope used for validation.
        """
        envelope = np.abs(phasic_signal)
        smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=envelope_smooth)
        threshold = envelope_threshold * np.max(smoothed_envelope)
        valid_events = []
        for idx in events:
            if idx < 0 or idx >= len(smoothed_envelope):
                continue
            local_env = smoothed_envelope[idx]
            if phasic_signal[idx] >= threshold and abs(phasic_signal[idx] - local_env) <= amplitude_proximity * local_env:
                valid_events.append(idx)
        return np.array(valid_events), smoothed_envelope

    @staticmethod
    def preprocess_signal(signal, fs):
        """
        Preprocesses the EDA signal by filtering and normalizing.
        
        Parameters:
        - signal: Raw EDA signal.
        - fs: Sampling frequency in Hz.
        
        Returns:
        - preprocessed_signal: Preprocessed EDA signal.
        """
        # Filtering
        filtered_signal = NumpySignalProcessor.bandpass_filter(signal, 0.01, 2, fs)
        
        # Normalization
        normalized_signal = NumpySignalProcessor.normalize_signal(filtered_signal)
        
        return normalized_signal

    @staticmethod
    def calculate_metrics(eda_signal, tonic, phasic, events, fs):
        """
        Calculates EDA metrics such as event count and mean tonic/phasic levels.
        
        Parameters:
        - eda_signal: The raw EDA signal.
        - tonic: Tonic component of the EDA signal.
        - phasic: Phasic component of the EDA signal.
        - events: Detected events in the phasic signal.
        - fs: Sampling frequency in Hz.
        
        Returns:
        - metrics: A dictionary containing EDA metrics.
        """
        event_count = len(events)
        mean_tonic = np.mean(tonic)
        mean_phasic = np.mean(phasic)
        
        return {
            "Event Count": event_count,
            "Mean Tonic Level (μS)": mean_tonic,
            "Mean Phasic Level (μS)": mean_phasic
        }

    @staticmethod
    def demo():
        """
        Demonstrates EDA signal processing by generating synthetic data,
        processing it, and plotting the results using PyQtGraph.
        """
        # Generate synthetic EDA data: periodic Gaussian pulses on a drift
        fs = 1000  # Sampling frequency
        duration = 10  # Duration in seconds
        t = np.linspace(0, duration, fs * duration)
        drift = 0.5 * t / duration  # Slow drift
        pulses = np.zeros_like(t)
        for i in range(1, duration):
            pulses += np.exp(-0.5 * ((t - i) / 0.05) ** 2)  # Gaussian pulse every second
        noise = 0.01 * np.random.randn(len(t))
        synthetic_adc = ((drift + pulses + noise) * 0.132 * (2 ** 10) / 3.3).clip(0, 2 ** 10 - 1)  # Simulate ADC values

        # Convert ADC values to EDA in micro-Siemens
        eda_us = EDA.convert_adc_to_eda(synthetic_adc)

        # Preprocess the signal
        preprocessed_signal = EDA.preprocess_signal(eda_us, fs)

        # Extract tonic and phasic components
        tonic, phasic = EDA.extract_tonic_phasic(preprocessed_signal, fs)

        # Detect peaks in the phasic component
        peaks = NumpySignalProcessor.find_peaks(phasic, fs)

        # Create synthetic data with peaks
        synthetic_data = np.column_stack((t, phasic, np.zeros_like(phasic)))
        synthetic_data[peaks, 2] = 1  # Mark peaks with 1 in the third column

        # Validate events
        validated_events, smoothed_envelope = EDA.validate_events(
            phasic, peaks, envelope_smooth=15, envelope_threshold=0.5, amplitude_proximity=0.1
        )

        # Calculate EDA metrics
        metrics = EDA.calculate_metrics(preprocessed_signal, tonic, phasic, validated_events, fs)
        print(f"EDA Metrics: {metrics}")
        print(f"Total detected events: {len(peaks)}")
        print(f"Validated events: {len(validated_events)}")

        # PyQtGraph visualization
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(show=True, title="EDA Signal Analysis")
        win.resize(1800, 1200)
        win.setWindowTitle("EDA Signal Analysis")

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # Plot 1: Raw EDA signal
        p1 = win.addPlot(row=0, col=0, title="<b>Raw EDA Signal (μS)</b>")
        p1.plot(t, eda_us, pen=pg.mkPen(color=(100, 200, 255), width=1.2))
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel('left', "<span style='color:white'>Amplitude (μS)</span>")
        p1.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Plot 2: Preprocessed EDA signal
        p2 = win.addPlot(row=1, col=0, title="<b>Preprocessed EDA Signal</b>")
        p2.plot(t, preprocessed_signal, pen=pg.mkPen(color=(255, 255, 0), width=1.2))
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p2.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Plot 3: Tonic and Phasic components
        p3 = win.addPlot(row=2, col=0, title="<b>Tonic and Phasic Components</b>")
        p3.plot(t, tonic, pen=pg.mkPen(color=(0, 255, 0), width=1.2), name="Tonic")
        p3.plot(t, phasic, pen=pg.mkPen(color=(255, 170, 0), width=1.2), name="Phasic")
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p3.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        # Plot 4: Phasic component with events
        p4 = win.addPlot(row=3, col=0, title="<b>Phasic Component with Events</b>")
        p4.plot(t, phasic, pen=pg.mkPen(color=(255, 170, 0), width=1.2), name="Phasic")
        p4.plot(t, smoothed_envelope, pen=pg.mkPen(color=(0, 255, 255), width=1.2), name="Smoothed Envelope")
        p4.plot(
            t[peaks], phasic[peaks], pen=None, symbol='x', symbolBrush=(255, 0, 0),
            symbolPen=pg.mkPen(color=(255, 0, 0), width=1.5), symbolSize=12, name="Detected Peaks"
        )
        p4.plot(
            t[validated_events], phasic[validated_events], pen=None, symbol='t',
            symbolBrush=(0, 255, 0), symbolPen=pg.mkPen(color=(0, 255, 0), width=1.5),
            symbolSize=12, name="Validated Events"
        )
        p4.showGrid(x=True, y=True, alpha=0.3)
        p4.setLabel('left', "<span style='color:white'>Amplitude</span>")
        p4.setLabel('bottom', "<span style='color:white'>Time (s)</span>")

        app.exec()


if __name__ == "__main__":
    EDA.demo()
