import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

class SignalProcessing:
    def __init__(self, signal=None):
        self.signal = signal if signal is not None else deque(maxlen=1000)

    def add_sample(self, value):
        """
        Add a new sample to the signal deque.
        """
        self.signal.append(value)

    def compute_stft(self, window_size=256, overlap=128):
        """
        Compute the Short-Time Fourier Transform (STFT) of the signal using FFT.
        :param window_size: Size of the FFT window.
        :param overlap: Overlap between consecutive windows.
        :return: Frequencies, time segments, and STFT magnitudes.
        """
        signal_array = np.array(self.signal)
        step = window_size - overlap
        num_windows = (len(signal_array) - overlap) // step

        if num_windows <= 0:
            return None  # Not enough data to compute STFT

        stft_magnitudes = []
        for i in range(num_windows):
            start = i * step
            end = start + window_size
            windowed_signal = signal_array[start:end] * np.hanning(window_size)
            fft_result = np.fft.rfft(windowed_signal)
            stft_magnitudes.append(np.abs(fft_result))

        stft_magnitudes = np.array(stft_magnitudes).T
        frequencies = np.fft.rfftfreq(window_size, d=1.0)  # Assuming a sampling rate of 1 Hz
        time_segments = np.arange(num_windows) * step

        return frequencies, time_segments, stft_magnitudes

    def detect_peaks(self, signal=None, threshold=0.5, smooth_window=5, normalize=True):
        """
        Detect peaks in the signal with optional smoothing and normalization.
        :param signal: The signal array. If None, use the internal buffer.
        :param threshold: Minimum height of peaks.
        :param smooth_window: Window size for Savitzky-Golay smoothing (must be odd).
        :param normalize: Whether to normalize the signal before peak detection.
        :return: Indices of detected peaks.
        """
        signal_array = np.array(self.signal if signal is None else signal)

        # Optional normalization
        if normalize:
            signal_array = (signal_array - np.min(signal_array)) / (np.max(signal_array) - np.min(signal_array) + 1e-6)


        # Detect peaks
        peaks, _ = find_peaks(signal_array, height=threshold)
        return peaks

    def moving_average(self, window_size):
        """
        Apply a moving average filter to the signal.
        :param window_size: Size of the moving average window.
        :return: Filtered signal as a deque.
        """
        signal_array = np.array(self.signal)
        cumsum = np.cumsum(np.insert(signal_array, 0, 0))
        smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return deque(smoothed, maxlen=self.signal.maxlen)

    def filter_threshold(self, threshold):
        """
        Apply a threshold filter to the signal.
        :param threshold: Threshold value.
        :return: Filtered signal as a deque.
        """
        filtered = [value if value >= threshold else 0 for value in self.signal]
        return deque(filtered, maxlen=self.signal.maxlen)

    def low_pass_filter(self, cutoff_freq, sampling_freq):
        """
        Apply a low-pass filter to the signal.
        :param cutoff_freq: Cutoff frequency.
        :param sampling_freq: Sampling frequency.
        :return: Filtered signal as a deque.
        """
        b, a = butter(4, cutoff_freq / (0.5 * sampling_freq), btype='low')
        filtered = filtfilt(b, a, np.array(self.signal))
        return deque(filtered, maxlen=self.signal.maxlen)

    def high_pass_filter(self, cutoff_freq, sampling_freq):
        """
        Apply a high-pass filter to the signal.
        :param cutoff_freq: Cutoff frequency.
        :param sampling_freq: Sampling frequency.
        :return: Filtered signal as a deque.
        """
        b, a = butter(4, cutoff_freq / (0.5 * sampling_freq), btype='high')
        filtered = filtfilt(b, a, np.array(self.signal))
        return deque(filtered, maxlen=self.signal.maxlen)

    def band_pass_filter(self, low_cutoff, high_cutoff, sampling_freq):
        """
        Apply a band-pass filter to the signal.
        :param low_cutoff: Low cutoff frequency.
        :param high_cutoff: High cutoff frequency.
        :param sampling_freq: Sampling frequency.
        :return: Filtered signal as a deque.
        """
        b, a = butter(4, [low_cutoff / (0.5 * sampling_freq), high_cutoff / (0.5 * sampling_freq)], btype='band')
        filtered = filtfilt(b, a, np.array(self.signal))
        return deque(filtered, maxlen=self.signal.maxlen)

    def pll(self, reference_signal, loop_gain, initial_phase):
        """
        Apply a Phase-Locked Loop (PLL) to the signal.
        :param reference_signal: Reference signal as a deque.
        :param loop_gain: Loop gain for the PLL.
        :param initial_phase: Initial phase of the PLL.
        :return: PLL-processed signal as a deque.
        """
        signal_array = np.array(self.signal)
        reference_array = np.array(reference_signal)
        phase = initial_phase
        output = []

        for i in range(len(signal_array)):
            error = reference_array[i] - signal_array[i]
            phase += loop_gain * error
            output.append(np.sin(phase))

        return deque(output, maxlen=self.signal.maxlen)

    def calculate_group_delay(self, filter_coefficients):
        """
        Calculate the group delay of a filter for delay compensation.
        :param filter_coefficients: Tuple of (b, a) filter coefficients.
        :return: Group delay in samples.
        """
        from scipy.signal import group_delay
        b, a = filter_coefficients
        _, gd = group_delay((b, a))
        return int(np.mean(gd))  # Average group delay in samples
