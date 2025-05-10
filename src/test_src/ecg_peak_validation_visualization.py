import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from ecg_signal_processing import ECG
import json
import os
from signal_processing import NumpySignalProcessor
import time

def bandpass_filter(signal, fs, lowcut, highcut):
    """
    Apply a band-pass filter to the signal.

    Parameters:
    - signal: The input signal.
    - fs: Sampling frequency in Hz.
    - lowcut: Low cutoff frequency in Hz.
    - highcut: High cutoff frequency in Hz.

    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def visualize_peak_validation(signal, fs, mode="qrs", smoothing_window=15, envelope_threshold=0.5, amplitude_proximity=0.1):
    """
    Visualizes the difference between detected peaks and validated peaks using envelope or smoothed envelope.

    Parameters:
    - signal: The input ECG signal.
    - fs: Sampling frequency in Hz.
    - mode: Detection mode ("qrs" or "all").
    - smoothing_window: Window size for smoothing the envelope.
    - envelope_threshold: Threshold for envelope-based validation.
    - amplitude_proximity: Proximity threshold for peak validation.
    """
    # Detect R-peaks
    detected_peaks = ECG.detect_r_peaks(signal, fs, mode=mode)

    # Compute the envelope and smoothed envelope
    envelope = np.abs(hilbert(signal))
    smoothed_envelope = np.convolve(envelope, np.ones(smoothing_window) / smoothing_window, mode='same')

    # Validate peaks using the envelope
    validated_peaks_envelope = np.array(
        ECG.validate_r_peaks(signal, detected_peaks, fs, envelope_threshold, int(1), amplitude_proximity), dtype=int
    )

    # Validate peaks using the smoothed envelope
    validated_peaks_smoothed = np.array(
        ECG.validate_r_peaks(signal, detected_peaks, fs, envelope_threshold, int(smoothing_window), amplitude_proximity), dtype=int
    )

    # Plot the results
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 8))

    # Original signal with detected peaks
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label="ECG Signal", color="blue")
    plt.scatter(time[detected_peaks], signal[detected_peaks], color="red", label="Detected Peaks", zorder=5)
    plt.title("Detected Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Signal with validated peaks using envelope
    plt.subplot(3, 1, 2)
    plt.plot(time, signal, label="ECG Signal", color="blue")
    plt.plot(time, envelope, label="Envelope", color="orange", linestyle="--")
    plt.scatter(time[validated_peaks_envelope], signal[validated_peaks_envelope], color="green", label="Validated Peaks (Envelope)", zorder=5)
    plt.title("Validated Peaks Using Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Signal with validated peaks using smoothed envelope
    plt.subplot(3, 1, 3)
    plt.plot(time, signal, label="ECG Signal", color="blue")
    plt.plot(time, smoothed_envelope, label="Smoothed Envelope", color="purple", linestyle="--")
    plt.scatter(time[validated_peaks_smoothed], signal[validated_peaks_smoothed], color="magenta", label="Validated Peaks (Smoothed Envelope)", zorder=5)
    plt.title("Validated Peaks Using Smoothed Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_np_methods(signal, fs, direct_peaks, envelope_signal, peaks_on_envelope):
    """
    Plot the results of NumpySignalProcessor direct and envelope peak detection.
    """
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 8))
    # 1. Original signal
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label="ECG Signal", color="blue")
    plt.title("NumpySignalProcessor: ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    # 2. Signal with direct peaks
    plt.subplot(3, 1, 2)
    plt.plot(time, signal, label="ECG Signal", color="blue")
    plt.scatter(time[direct_peaks], signal[direct_peaks], color="red", label="Direct Peaks", zorder=5)
    plt.title("NumpySignalProcessor: Direct Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    # 3. Envelope with peaks detected on envelope
    plt.subplot(3, 1, 3)
    plt.plot(time, envelope_signal, label="Envelope", color="orange")
    plt.scatter(time[peaks_on_envelope], envelope_signal[peaks_on_envelope], color="green", label="Peaks on Envelope", zorder=5)
    plt.title("NumpySignalProcessor: Peaks on Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope Amplitude")
    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    # Example usage
    fs = 1000  # Sampling frequency in Hz

    # Define the correct file path for the collarbone signal
    collarbone_file_path = os.path.join(os.path.dirname(__file__), "ECG", "collarbone_signal_data.json")

    # Load the collarbone signal using NumpySignalProcessor
    raw_collarbone_signal = NumpySignalProcessor.load_signal(collarbone_file_path)

    # Apply band-pass filter (e.g., 0.5-50 Hz for ECG signals)
    filtered_signal = bandpass_filter(raw_collarbone_signal, fs, lowcut=8, highcut=15)

    # Benchmark 1: Direct peak detection with NumpySignalProcessor
    start = time.time()
    direct_peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs)
    direct_time = time.time() - start

    # Benchmark 1.5: Apply envelope and then use NumpySignalProcessor to find peaks
    start = time.time()
    envelope_signal = np.abs(hilbert(filtered_signal))
    peaks_on_envelope = NumpySignalProcessor.find_peaks(envelope_signal, fs)
    envelope_np_time = time.time() - start

    # Benchmark 2: Peak validation with envelope
    start = time.time()
    detected_peaks = ECG.detect_r_peaks(filtered_signal, fs, mode="qrs")
    envelope = np.abs(hilbert(filtered_signal))
    validated_peaks_envelope = np.array(
        ECG.validate_r_peaks(filtered_signal, detected_peaks, fs, 0.5, int(1), 0.1), dtype=int
    )
    envelope_time = time.time() - start

    # Benchmark 3: Peak validation with smoothed envelope
    start = time.time()
    smoothed_envelope = np.convolve(envelope, np.ones(15) / 15, mode='same')
    validated_peaks_smoothed = np.array(
        ECG.validate_r_peaks(filtered_signal, detected_peaks, fs, 0.5, 15, 0.1), dtype=int
    )
    smoothed_time = time.time() - start

    print(f"Direct NumpySignalProcessor peak detection: {direct_time:.6f} seconds, {len(direct_peaks)} peaks")
    print(f"NumpySignalProcessor peak detection on envelope: {envelope_np_time:.6f} seconds, {len(peaks_on_envelope)} peaks")
    print(f"Envelope validation: {envelope_time:.6f} seconds, {len(validated_peaks_envelope)} peaks")
    print(f"Smoothed envelope validation: {smoothed_time:.6f} seconds, {len(validated_peaks_smoothed)} peaks")

    # Plot for NumpySignalProcessor methods
    plot_np_methods(filtered_signal, fs, direct_peaks, envelope_signal, peaks_on_envelope)

    # Visualize the filtered signal with peak validation (envelope methods)
    visualize_peak_validation(filtered_signal, fs, mode="qrs")
