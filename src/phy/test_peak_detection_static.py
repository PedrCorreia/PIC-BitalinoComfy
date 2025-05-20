import numpy as np
import matplotlib.pyplot as plt
from src.phy.ecg_signal_processing import ECG
from src.utils.signal_processing import NumpySignalProcessor

def test_peak_detection_on_static_signal():
    # Generate a synthetic ECG-like signal
    fs = 250  # Hz
    t = np.linspace(0, 10, fs*10)  # 10 seconds
    # Simulate a simple ECG: sum of sinusoids + noise
    ecg = 1.2 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + 0.1 * np.random.randn(len(t))
    # Bandpass filter
    filtered = NumpySignalProcessor.bandpass_filter(ecg, lowcut=5, highcut=15, fs=fs)
    # Detect peaks
    peaks = ECG.detect_r_peaks(filtered, fs=fs, mode="qrs")
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, filtered, label="Filtered ECG")
    plt.scatter(t[peaks], filtered[peaks], color='red', marker='x', label="Detected Peaks")
    plt.title("ECG Peak Detection Test (Static Signal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_peak_detection_on_static_signal()
