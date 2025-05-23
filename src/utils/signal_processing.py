from scipy.signal import butter, filtfilt, welch  # Added welch import
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import time
try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.signal import detrend


class NumpySignalProcessor:
    @staticmethod
    def bandpass_filter(signal, lowcut, highcut, fs, order=4):
        """
        Applies a zero-phase bandpass filter to the signal using NumPy's butter and filtfilt.
        
        Parameters:
        - signal
        - lowcut: Low cutoff frequency in Hz.
        - highcut: High cutoff frequency in Hz.
        - fs: Sampling frequency in Hz.
        - order: Order of the Butterworth filter (default: 1).
        
        Returns:
        - filtered_signal: The filtered signal.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    @staticmethod
    def lowpass_filter(signal, cutoff, fs, order=1):
        """
        Applies a lowpass filter to the signal using a Butterworth filter.
        """
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low')
        start = time.time()
        filtered_signal = filtfilt(b, a, signal)
        elapsed = time.time() - start
        return filtered_signal

    @staticmethod
    def highpass_filter(signal, cutoff, fs, order=1):
        """
        Applies a highpass filter to the signal using a Butterworth filter.
        """
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='high')
       
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal

    @staticmethod
    def find_peaks(signal, fs, window=None, prominence=None, threshold=None,Live=False):
        """
        Finds peaks in the signal using NumPy, with an optional threshold.

        Parameters:
        - signal: Input signal (array).
        - fs: Sampling frequency (not used directly but kept for consistency).
        - window: Size of the sliding window for peak detection.
        - prominence: Minimum prominence of peaks.
        - threshold: Optional absolute threshold for peak selection.

        Returns:
        - indices: Indices of the detected peaks.
        """
        if Live:
            # For live signal processing, we need a more robust approach
            N = len(signal)
            if window is None:
                window = max(1, int(0.05 * N))  # Larger default window for live signals
            if prominence is None:
                prominence = 0.02 * np.std(signal)  # More sensitive prominence for live signals
                
            # Calculate adaptive statistics from recent signal history
            recent_mean = np.mean(signal[-min(N, 1000):])  # Use most recent 1000 samples or less
            recent_std = np.std(signal[-min(N, 1000):])
            
            # Adaptive threshold based on recent statistics
            adaptive_threshold = recent_mean + prominence * recent_std
            
            # For live processing, we only check if the last point could be a peak
            is_peak = False
            if N > window:
                # Check if the current point is a local maximum in its window
                window_start = max(0, N - window - 1)
                window_end = N
                window_signal = signal[window_start:window_end]
                if signal[-1] == max(window_signal) and signal[-1] > adaptive_threshold:
                    is_peak = True
            
            # Return the index of the detected peak (if found)
            return [N-1] if is_peak else []
        else:
            N = len(signal)
            if window is None:
                window = max(1, int(0.01 * N))  # Default window is 1% of signal length
            if prominence is None:
                prominence = 0.01 * np.std(signal)  # Default prominence is 10% of std deviation
            mean = np.mean(signal)
            # Pad the signal at both ends to handle edge effects
            padded = np.pad(signal, (window, window), mode='edge')
            # Create a sliding window view of the signal
            shape = (N, 2 * window + 1)
            strides = (padded.strides[0], padded.strides[0])
            windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
            # Find the maximum value in each window
            max_in_window = np.max(windows, axis=1)
            # A peak is a point that is the maximum in its window and above mean+prominence
            is_peak = (signal >= max_in_window) & (signal > mean + prominence)

            # If a threshold is provided, further require the peak to be above this value
            if threshold is not None:
                is_peak &= (signal > threshold)

            # Return the indices of detected peaks
            return np.flatnonzero(is_peak)

    @staticmethod
    def compute_psd_numpy(signal, fs, nperseg=None, noverlap=None, window='hann', detrend='constant'):
        """
        Computes the power spectral density using Welch's method with optional parameters for optimization.
        """
        if nperseg is None:
            # Use a larger segment for longer signals, but not more than 4096
            nperseg = min(4096, max(256, len(signal) // 8))
        if noverlap is None:
            noverlap = nperseg // 2  # 50% overlap is typical
        freqs, psd = welch(
            signal,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            detrend=detrend,
            scaling='density',
            average='mean'
        )
        return freqs, psd

    @staticmethod
    def normalize_signal(signal):
        """
        Normalizes the signal to a range between 0 and 1.
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val)

    @staticmethod
    def baseline_als_optimized(y, lam, p, niter=10):
        """
        Optimized baseline correction using Asymmetric Least Squares (ALS).

        Parameters:
        - y: Input signal (1D NumPy array).
        - lam: Smoothing parameter (higher values make the baseline smoother).
        - p: Asymmetry parameter (between 0 and 1, typically small, e.g., 0.01).
        - niter: Number of iterations (default: 10).

        Returns:
        - z: Estimated baseline (same shape as y).
        """
        L = len(y)
        # Construct the second-order difference matrix
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term for efficiency
        # Initialize weights to all ones for the first iteration
        w = np.ones(L)
        # Create a sparse diagonal matrix W with the current weights on the diagonal
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Update diagonal values with current weights
            Z = W + D
            # Solve the linear system to estimate the baseline
            z = spsolve(Z, w * y)
            # Update weights: p for points above baseline, (1-p) for below
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    @staticmethod
    def correct_baseline(signal, method="als", **kwargs):
        """
        Correct the baseline of the signal using the specified method.
        
        Parameters:
        - signal: The input signal.
        - method: The baseline correction method ("als", "polynomial", or "detrend").
        - kwargs: Additional parameters for the chosen method.
        
        Returns:
        - The signal with the baseline corrected.
        """
        if method == "als":
            lam = kwargs.get("lam", 1e6)
            p = kwargs.get("p", 0.01)
            niter = kwargs.get("niter", 10)
            baseline = NumpySignalProcessor.baseline_als_optimized(signal, lam, p, niter)
            return signal - baseline
        elif method == "polynomial":
            degree = kwargs.get("degree", 3)
            return NumpySignalProcessor._correct_baseline_polynomial(signal, degree)
        elif method == "detrend":
            return NumpySignalProcessor._correct_baseline_detrend(signal)
        else:
            raise ValueError("Unsupported baseline correction method: {}".format(method))

    @staticmethod
    def _correct_baseline_polynomial(signal, degree):
        """
        Corrects the baseline using polynomial fitting.
        
        Parameters:
        - signal: The input signal.
        - degree: The degree of the polynomial to fit.
        
        Returns:
        - The signal with the baseline corrected.
        """
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, degree)
        baseline = np.polyval(coeffs, x)
        return signal - baseline

    @staticmethod
    def _correct_baseline_detrend(signal):
        """
        Corrects the baseline using the detrend method from SciPy.
        
        Parameters:
        - signal: The input signal.
        
        Returns:
        - The signal with the baseline corrected.
        """
        return detrend(signal)

    @staticmethod
    def amplify_signal(signal, amplification_factor):
        """
        Amplifies the signal by a given factor.
        """
        return signal * amplification_factor

    @staticmethod
    def load_signal(file_path):
        """
        Loads a signal from a JSON file.
        
        Parameters:
        - file_path: Path to the JSON file containing the signal data.
        
        Returns:
        - A NumPy array containing the signal from the first channel.
        """
        import json
        with open(file_path, "r") as f:
            data = json.load(f)
        return np.array([frame["data"][0] for frame in data])  # Extract first channel

    @staticmethod
    def moving_average(signal, window_size):
        """
        Applies a moving average filter to the signal.
        
        Parameters:
        - signal: Input signal (array).
        - window_size: Size of the moving average window.
        
        Returns:
        - smoothed_signal: The smoothed signal.
        """
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def deque_to_numpy(deq):
        """
        Efficiently converts a deque to a NumPy array (static method).
        """
        return np.fromiter(deq, dtype=float, count=len(deq))


class TorchSignalProcessor:
    @staticmethod
    def bandpass_filter(signal, lowcut=0.1, highcut=0.5, fs=1000, fir_order=301):
        """
        Implements an FIR bandpass filter using PyTorch.
        Optimized with FFT-based convolution for long signals.
        """
        from scipy.signal import firwin

        # Design FIR filter
        nyquist = 0.5 * fs
        taps = firwin(fir_order, [lowcut / nyquist, highcut / nyquist], pass_zero=False)

        # Convert to PyTorch tensors
        signal = torch.tensor(signal, dtype=torch.float32, device="cuda")
        taps = torch.tensor(taps, dtype=torch.float32, device="cuda")

        # Use FFT-based convolution for long signals
        if len(signal) > 10_000:
            n_fft = 2 ** int(np.ceil(np.log2(len(taps) + len(signal) - 1)))
            signal_fft = torch.fft.rfft(signal, n=n_fft)
            taps_fft = torch.fft.rfft(taps, n=n_fft)
            filtered_signal = torch.fft.irfft(signal_fft * taps_fft)[:len(signal)]
        else:
            filtered_signal = torch.nn.functional.conv1d(
                signal.unsqueeze(0).unsqueeze(0),
                taps.view(1, 1, -1),
                padding=taps.size(0) // 2
            ).squeeze()

        return filtered_signal.cpu().numpy()

    @staticmethod
    def find_peaks(signal, fs, window=None, prominence=None):
        """
        Finds peaks in the signal using PyTorch.
        Optimized to use efficient tensor operations.
        """
        signal = torch.tensor(signal, dtype=torch.float32, device="cuda")
        N = len(signal)
        if window is None:
            window = max(1, int(0.01 * N))
        if prominence is None:
            prominence = 0.1 * torch.std(signal).item()
        mean = torch.mean(signal).item()

        # Pad the signal
        padded = torch.nn.functional.pad(signal, (window, window), mode='constant', value=mean - 10 * prominence)

        # Create sliding windows
        windows = padded.unfold(0, 2 * window + 1, 1)

        # Find peaks
        max_in_window, _ = torch.max(windows, dim=1)
        is_peak = (signal >= max_in_window) & (signal > mean + prominence)
        return torch.nonzero(is_peak, as_tuple=False).squeeze().cpu().numpy()

    @staticmethod
    def preprocess_signal(signal):
        """
        Preprocesses the signal by removing the baseline (mean).
        Optimized to leverage GPU acceleration and match CuPy's efficiency.
        """
        signal = torch.tensor(signal, dtype=torch.float32, device="cuda", requires_grad=False)
        mean = torch.mean(signal)  # Compute mean efficiently
        std = torch.std(signal)  # Compute standard deviation efficiently
        signal = (signal - mean) / std  # Normalize in a single operation
        return signal.cpu().numpy()

    @staticmethod
    def iir_filter(signal, b, a):
        """
        Applies an IIR filter using PyTorch.
        Parameters:
            signal: Input signal (1D array).
            b: Numerator coefficients of the IIR filter.
            a: Denominator coefficients of the IIR filter.
        """
        signal = torch.tensor(signal, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)

        # Initialize the output signal
        y = torch.zeros_like(signal)

        # Apply the IIR filter using the difference equation
        for i in range(len(signal)):
            y[i] = b[0] * signal[i]
            for j in range(1, len(b)):
                if i - j >= 0:
                    y[i] += b[j] * signal[i - j]
            for j in range(1, len(a)):
                if i - j >= 0:
                    y[i] -= a[j] * y[i - j]
        return y.numpy()

    @staticmethod
    def compute_fft(signal, fs, device="cuda"):
        """
        Computes the FFT of the signal using PyTorch.
        """
        torch_signal = torch.from_numpy(signal).float().to(device)
        torch.cuda.synchronize()
        start = time.time()
        fft_values = torch.fft.rfft(torch_signal)
        power = torch.abs(fft_values) ** 2
        torch.cuda.synchronize()
        elapsed = time.time() - start
        freqs = np.fft.rfftfreq(len(signal), d=1/fs)
        return freqs, power.cpu().numpy(), elapsed


class CudaSignalProcessor:
    @staticmethod
    def preprocess_signal(signal):
        """
        Preprocesses the signal by removing the baseline (mean) using CuPy.
        """
        signal = cp.asarray(signal, dtype=cp.float32)
        mean = cp.mean(signal)
        std = cp.std(signal)
        normalized_signal = (signal - mean) / std
        return cp.asnumpy(normalized_signal)

    @staticmethod
    def bandpass_filter(signal, lowcut, highcut, fs, fir_order=301):
        """
        Implements an FIR bandpass filter using CuPy.
        Optimized with overlap-save FFT convolution for long signals.
        """
        from scipy.signal import firwin

        nyquist = 0.5 * fs
        taps = firwin(fir_order, [lowcut / nyquist, highcut / nyquist], pass_zero=False)
        signal = cp.asarray(signal, dtype=cp.float32)
        taps = cp.asarray(taps, dtype=cp.float32)

        if len(signal) > 10_000:
            n_fft = 2 ** int(np.ceil(np.log2(len(taps) + len(signal) - 1)))
            signal_fft = cp.fft.rfft(signal, n=n_fft)
            taps_fft = cp.fft.rfft(taps, n=n_fft)
            filtered_signal = cp.fft.irfft(signal_fft * taps_fft)[:len(signal)]
        else:
            filtered_signal = cp.convolve(signal, taps, mode='same')

        return cp.asnumpy(filtered_signal)

    @staticmethod
    def find_peaks(signal, fs, window=None, prominence=None):
        """
        Finds peaks in the signal using CuPy.
        """
        signal = cp.asarray(signal, dtype=cp.float32)
        N = len(signal)
        if window is None:
            window = max(1, int(0.01 * N))
        if prominence is None:
            prominence = 0.1 * cp.std(signal).item()
        mean = cp.mean(signal).item()

        padded = cp.pad(signal, (window, window), mode='constant', constant_values=mean - 10 * prominence)
        shape = (N, 2 * window + 1)
        strides = (padded.strides[0], padded.strides[0])
        windows = cp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

        max_in_window = cp.max(windows, axis=1)
        is_peak = (signal >= max_in_window) & (signal > mean + prominence)
        return cp.flatnonzero(is_peak).get()

    @staticmethod
    def iir_filter(signal, b, a):
        """
        Applies an IIR filter using CuPy.
        Parameters:
            signal: Input signal (1D array).
            b: Numerator coefficients of the IIR filter.
            a: Denominator coefficients of the IIR filter.
        """
        signal = cp.asarray(signal, dtype=cp.float32)
        b = cp.asarray(b, dtype=cp.float32)
        a = cp.asarray(a, dtype=cp.float32)

        # Initialize the output signal
        y = cp.zeros_like(signal)

        # Apply the IIR filter using the difference equation
        for i in range(len(signal)):
            y[i] = b[0] * signal[i]
            for j in range(1, len(b)):
                if i - j >= 0:
                    y[i] += b[j] * signal[i - j]
            for j in range(1, len(a)):
                if i - j >= 0:
                    y[i] -= a[j] * y[i - j]
        return cp.asnumpy(y)

    @staticmethod
    def compute_fft(signal, fs):
        """
        Computes the FFT of the signal using CuPy.
        """
        cp_signal = cp.asarray(signal)
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        freqs = cp.fft.rfftfreq(len(cp_signal), d=1/fs)
        fft_values = cp.fft.rfft(cp_signal)
        power = cp.abs(fft_values) ** 2
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        return cp.asnumpy(freqs), cp.asnumpy(power), elapsed

    @staticmethod
    def benchmark_signal_processors(signal, fs):
        """
        Benchmarks Numpy, Torch, and CUDA implementations for preprocessing, bandpass_filter, and find_peaks.
        """
        results = {}

        # Numpy
        start_time = time.time()
        preprocessed_signal = NumpySignalProcessor.preprocess_signal(signal)
        preprocessing_time = time.time() - start_time

        filtered_signal, filtering_time = NumpySignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.1, fs)

        start_time = time.time()
        peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs)
        peak_detection_time = time.time() - start_time

        results['numpy'] = {
            'preprocessing_time': preprocessing_time,
            'filtering_time': filtering_time,
            'peak_detection_time': peak_detection_time,
            'total_time': preprocessing_time + filtering_time + peak_detection_time,
            'peaks': len(peaks)
        }

        # Torch
        start_time = time.time()
        preprocessed_signal = TorchSignalProcessor.preprocess_signal(signal)  # Use Torch preprocessing
        preprocessing_time = time.time() - start_time

        start_time = time.time()
        filtered_signal = TorchSignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.1, fs)
        filtering_time = time.time() - start_time

        start_time = time.time()
        peaks = TorchSignalProcessor.find_peaks(filtered_signal, fs)
        print(peaks)
        peak_detection_time = time.time() - start_time

        results['torch'] = {
            'preprocessing_time': preprocessing_time,
            'filtering_time': filtering_time,
            'peak_detection_time': peak_detection_time,
            'total_time': preprocessing_time + filtering_time + peak_detection_time,
            'peaks': len(peaks)
        }

        # CUDA
        start_time = time.time()
        preprocessed_signal = CudaSignalProcessor.preprocess_signal(signal)  # Use CUDA preprocessing
        preprocessing_time = time.time() - start_time

        filtered_signal = CudaSignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.1, fs)
        filtering_time = time.time() - start_time

        start_time = time.time()
        peaks = CudaSignalProcessor.find_peaks(filtered_signal, fs)
        peak_detection_time = time.time() - start_time

        results['cuda'] = {
            'preprocessing_time': preprocessing_time,
            'filtering_time': filtering_time,
            'peak_detection_time': peak_detection_time,
            'total_time': preprocessing_time + filtering_time + peak_detection_time,
            'peaks': len(peaks)
        }

        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a sample signal with 10 distinct frequencies, noise, and 4 seconds duration
    fs = 1000  # Sampling frequency (Hz)
    duration = 10  # Duration in seconds
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    frequencies = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]  # Distinct frequencies
    signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies) + np.random.normal(0, 0.5, t.shape)

    # Benchmark all processors
    results = CudaSignalProcessor.benchmark_signal_processors(signal, fs)

    # Print benchmark results
    print("Benchmark Results:")
    for processor, result in results.items():
        print(f"{processor.capitalize()} - Preprocessing: {result['preprocessing_time']:.6f}s, "
              f"Filtering: {result['filtering_time']:.6f}s, "
              f"Peak Detection: {result['peak_detection_time']:.6f}s, "
              f"Total: {result['total_time']:.6f}s, Peaks: {result['peaks']}")

    # Create subplots for visualization
    techniques = ['numpy', 'torch', 'cuda']
    processes = ['Raw FFT', 'Preprocessing', 'Filtering', 'Peak Detection', 'Filtered FFT']
    fig, axes = plt.subplots(len(techniques), len(processes), figsize=(25, 12), constrained_layout=True)

    for i, tech in enumerate(techniques):
        # Compute FFT of the raw signal
        start_time = time.time()
        if tech == 'numpy':
            raw_freqs = np.fft.rfftfreq(len(signal), d=1/fs)
            raw_fft_values = np.abs(np.fft.rfft(signal))
        elif tech == 'torch':
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            raw_fft_values = torch.fft.rfft(signal_tensor).abs().numpy()
            raw_freqs = torch.fft.rfftfreq(len(signal), d=1/fs).numpy()
        else:  # CUDA
            signal_gpu = cp.asarray(signal, dtype=cp.float32)
            raw_fft_values = cp.abs(cp.fft.rfft(signal_gpu)).get()
            raw_freqs = cp.fft.rfftfreq(len(signal), d=1/fs).get()
        raw_fft_time = time.time() - start_time

        # Preprocess the signal
        if tech == 'numpy':
            preprocessed_signal = NumpySignalProcessor.preprocess_signal(signal)
        elif tech == 'torch':
            preprocessed_signal = TorchSignalProcessor.preprocess_signal(signal)
        else:  # CUDA
            preprocessed_signal = CudaSignalProcessor.preprocess_signal(signal)

        # Filter the signal with a bandpass filter targeting 1 Hz
        if tech == 'numpy':
            filtered_signal, _ = NumpySignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.2, fs)
        elif tech == 'torch':
            filtered_signal = TorchSignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.1, fs)
        else:  # CUDA
            filtered_signal = CudaSignalProcessor.bandpass_filter(preprocessed_signal, 0.9, 1.1, fs)

        # Adjust the filtered signal to match the original signal length
        if len(filtered_signal) > len(t):
            filtered_signal = filtered_signal[:len(t)]
        elif len(filtered_signal) < len(t):
            filtered_signal = np.pad(filtered_signal, (0, len(t) - len(filtered_signal)), mode='constant')

        # Detect peaks
        if tech == 'numpy':
            peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs)
        elif tech == 'torch':
            peaks = TorchSignalProcessor.find_peaks(filtered_signal, fs)
        else:  # CUDA
            peaks = CudaSignalProcessor.find_peaks(filtered_signal, fs)  

        # Compute FFT of the filtered signal
        start_time = time.time()
        if tech == 'numpy':
            filtered_freqs = np.fft.rfftfreq(len(filtered_signal), d=1/fs)
            filtered_fft_values = np.abs(np.fft.rfft(filtered_signal))
        elif tech == 'torch':
            signal_tensor = torch.tensor(filtered_signal, dtype=torch.float32)
            filtered_fft_values = torch.fft.rfft(signal_tensor).abs().numpy()
            filtered_freqs = torch.fft.rfftfreq(len(filtered_signal), d=1/fs).numpy()
        else:  # CUDA
            signal_gpu = cp.asarray(filtered_signal, dtype=cp.float32)
            filtered_fft_values = cp.abs(cp.fft.rfft(signal_gpu)).get()
            filtered_freqs = cp.fft.rfftfreq(len(filtered_signal), d=1/fs).get()
        filtered_fft_time = time.time() - start_time

        # Print FFT times
        print(f"{tech.capitalize()} - Raw FFT Time: {raw_fft_time:.6f}s, Filtered FFT Time: {filtered_fft_time:.6f}s")

        # Plot Raw FFT
        axes[i, 0].plot(raw_freqs, raw_fft_values, label=f'{tech.capitalize()} Raw FFT', alpha=0.7)
        axes[i, 0].set_title(f'{tech.capitalize()} - Raw FFT')
        axes[i, 0].set_xlabel('Frequency (Hz)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].legend(loc='upper right')
        axes[i, 0].grid(True)

        # Plot Preprocessing
        axes[i, 1].plot(t, preprocessed_signal, label=f'{tech.capitalize()} Preprocessed', alpha=0.7)
        axes[i, 1].set_title(f'{tech.capitalize()} - Preprocessing')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].grid(True)

        # Plot Filtering
        axes[i, 2].plot(t, filtered_signal, label=f'{tech.capitalize()} Filtered', alpha=0.7)
        axes[i, 2].set_title(f'{tech.capitalize()} - Filtering')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].legend(loc='upper right')
        axes[i, 2].grid(True)

        # Plot Peak Detection
        axes[i, 3].plot(t, filtered_signal, label=f'{tech.capitalize()} Filtered', alpha=0.7)
        axes[i, 3].plot(t[peaks], filtered_signal[peaks], 'x', label=f'{tech.capitalize()} Peaks', markersize=8, color='red')
        axes[i, 3].set_title(f'{tech.capitalize()} - Peak Detection')
        axes[i, 3].set_xlabel('Time (s)')
        axes[i, 3].set_ylabel('Amplitude')
        axes[i, 3].legend(loc='upper right')
        axes[i, 3].grid(True)

        # Plot Filtered FFT
        axes[i, 4].plot(filtered_freqs, filtered_fft_values, label=f'{tech.capitalize()} Filtered FFT', alpha=0.7)
        axes[i, 4].set_title(f'{tech.capitalize()} - Filtered FFT')
        axes[i, 4].set_xlabel('Frequency (Hz)')
        axes[i, 4].set_ylabel('Amplitude')
        axes[i, 4].legend(loc='upper right')
        axes[i, 4].grid(True)

    plt.suptitle('Signal Processing Comparison Across Techniques', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.show()
