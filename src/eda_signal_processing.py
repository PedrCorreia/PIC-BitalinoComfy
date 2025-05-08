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
        filtered_signal = NumpySignalProcessor.bandpass_filter(signal, 0.05, 1.0, fs)
        
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
    file_path = "/media/lugo/data/ComfyUI/custom_nodes/PIC_BitalinoComfy/report/test/EDA/signal_data.json"
    raw_signal = NumpySignalProcessor.load_signal(file_path)  # Use NumpySignalProcessor to load the signal
    
    fs = 1000
    # Convert ADC values to EDA in micro-Siemens
    eda_us = EDA.convert_adc_to_eda(raw_signal)
    
    # Preprocess the signal
    preprocessed_signal = EDA.preprocess_signal(eda_us, fs)
    
    # Extract tonic and phasic components
    tonic, phasic = EDA.extract_tonic_phasic(preprocessed_signal, fs)
    
    # Detect events in the phasic component
    events = EDA.detect_events(phasic)
    
    # Calculate EDA metrics
    metrics = EDA.calculate_metrics(preprocessed_signal, tonic, phasic, events, fs)
    print(f"EDA Metrics: {metrics}")
