import numpy as np
from signal_processing import NumpySignalProcessor


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
        tonic = NumpySignalProcessor.lowpass_filter(eda_signal, cutoff=0.05, fs=fs)
        
        # Phasic component is the difference between the original signal and tonic
        phasic = eda_signal - tonic
        
        return tonic, phasic

    @staticmethod
    def detect_events(phasic_signal, threshold=0.01):
        """
        Detects events in the phasic component of the EDA signal.
        
        Parameters:
        - phasic_signal: The phasic component of the EDA signal.
        - threshold: Threshold for event detection (default: 0.01 μS).
        
        Returns:
        - events: Indices of detected events.
        """
        events = np.where(phasic_signal > threshold)[0]
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic EDA data: periodic Gaussian pulses on a drift
    fs = 1000
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)
    drift = 0.5 * t / duration  # slow drift
    pulses = np.zeros_like(t)
    for i in range(1, duration):
        pulses += np.exp(-0.5 * ((t - i) / 0.05) ** 2)  # Gaussian pulse every second
    noise = 0.01 * np.random.randn(len(t))
    synthetic_adc = ((drift + pulses + noise) * 0.132 * (2 ** 10) / 3.3).clip(0, 2 ** 10 - 1)  # simulate ADC values

    # Convert ADC values to EDA in micro-Siemens
    eda_us = EDA.convert_adc_to_eda(synthetic_adc)

    # Preprocess the signal
    preprocessed_signal = EDA.preprocess_signal(eda_us, fs)

    # Extract tonic and phasic components
    tonic, phasic = EDA.extract_tonic_phasic(preprocessed_signal, fs)

    # Detect events in the phasic component
    events = EDA.detect_events(phasic)

    # Validate events (ECG-like logic)
    validated_events, smoothed_envelope = EDA.validate_events(
        phasic, events, envelope_smooth=15, envelope_threshold=0.5, amplitude_proximity=0.1
    )

    # Calculate EDA metrics
    metrics = EDA.calculate_metrics(preprocessed_signal, tonic, phasic, events, fs)
    print(f"EDA Metrics: {metrics}")
    print(f"Total detected events: {len(events)}")
    print(f"Validated events: {len(validated_events)}")
    if len(validated_events) > 0:
        print(f"First validated event at t = {t[validated_events[0]]:.3f} s")

    # Plot for visualization
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t, eda_us)
    plt.title("Synthetic EDA Signal (μS)")
    plt.subplot(4, 1, 2)
    plt.plot(t, preprocessed_signal)
    plt.title("Preprocessed EDA Signal")
    plt.subplot(4, 1, 3)
    plt.plot(t, tonic, label="Tonic")
    plt.plot(t, phasic, label="Phasic")
    plt.legend()
    plt.title("Tonic and Phasic Components")
    plt.subplot(4, 1, 4)
    plt.plot(t, phasic, label="Phasic")
    plt.plot(t, smoothed_envelope, label="Smoothed Envelope", color="orange")
    plt.scatter(t[events], phasic[events], color='green', label='Detected Events', marker='o')
    plt.scatter(t[validated_events], phasic[validated_events], color='magenta', label='Validated Events', marker='^')
    # Plot bars at event locations (beginning of phasic curves)
    for ev in validated_events:
        plt.axvline(t[ev], color='magenta', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Phasic Component, Envelope, and Events")
    plt.tight_layout()
    plt.show()
