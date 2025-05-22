import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, savgol_filter
import time
from src.utils.signal_processing import NumpySignalProcessor
from scipy.signal import butter, filtfilt

# --- Improved Synthetic ECG waveform generator ---
def ecg_waveform(t, hr_mean=70, hr_std=1.5, noise_std=0.02, baseline_amp=0.01):
    # Simulate heart rate variability
    np.random.seed(42)
    rr_intervals = np.random.normal(60.0/hr_mean, hr_std/60.0*hr_mean, int(t[-1]*hr_mean/60)+10)
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < t[-1]+2.0]
    # Generate ECG as sum of Gaussian waves for P, Q, R, S, T
    ecg = np.zeros_like(t)
    for bt in beat_times:
        # P wave
        ecg += 0.12 * np.exp(-((t-bt-0.12)**2)/(2*0.012**2))
        # Q wave
        ecg += -0.15 * np.exp(-((t-bt-0.04)**2)/(2*0.008**2))
        # R wave (sharper)
        ecg += 0.9 * np.exp(-((t-bt)**2)/(2*0.010**2))
        # S wave
        ecg += -0.12 * np.exp(-((t-bt+0.03)**2)/(2*0.010**2))
        # T wave
        ecg += 0.25 * np.exp(-((t-bt+0.22)**2)/(2*0.030**2))
    # Add baseline wander and noise
    baseline = baseline_amp * np.sin(2 * np.pi * 0.33 * t) + baseline_amp * np.sin(2 * np.pi * 0.07 * t)
    noise = noise_std * np.random.randn(len(t))
    return ecg + baseline + noise

def find_peaks2(signal, fs, window=None, prominence=None, threshold=None):
    """
    Real-time (causal) peak detection using only NumPy. No stride tricks, no padding, no scipy.
    Each point is a peak if it is the maximum in its causal window and above mean+prominence (and threshold if given).
    """
    N = len(signal)
    if window is None:
        window = max(1, int(0.01 * N))  # Default window is 1% of signal length
    if prominence is None:
        prominence = 0.01 * np.std(signal)  # Default prominence is 1% of std deviation
    mean = np.mean(signal)
    # Causal window: for each point, look back 'window' samples (not forward)
    peaks = []
    for i in range(window, N):
        win = signal[i-window:i+1]  # causal window: includes current point
        if signal[i] == np.max(win):
            if signal[i] > mean + prominence:
                if threshold is None or signal[i] > threshold:
                    # To avoid flat peaks, require the peak to be strictly greater than all previous in window
                    if np.sum(win == signal[i]) == 1:
                        peaks.append(i)
    return np.array(peaks, dtype=int)

# --- Reference: Your original envelope-based peak detection ---
def find_peaks_moving_average_envelope(filtered_signal, fs, prominence=None):
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    threshold = 0.30 * np.max(smoothed_envelope)
    r_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs, threshold=threshold, prominence=prominence)
    return np.array(r_peaks, dtype=int)

# --- Improved: Savitzky-Golay envelope and adaptive window ---
def find_peaks_savgol_envelope(filtered_signal, fs, prominence=None):
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    # Adaptive window: 50 ms or at least 5 samples, must be odd
    window_size = max(5, int(0.05 * fs) | 1)
    smoothed_envelope = savgol_filter(envelope, window_length=window_size, polyorder=2, mode='interp')
    threshold = 0.30 * np.max(smoothed_envelope)
    r_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs, threshold=threshold, prominence=prominence)
    return np.array(r_peaks, dtype=int)

# --- Benchmark and plot ---
def main():
    fs = 1000
    t = np.arange(0, 5, 1/fs)
    raw = ecg_waveform(t)
    # Apply a bandpass filter (0.5-40 Hz is typical for ECG)
    
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Ensure valid band
        if not (0 < low < high < 1):
            print(f"Warning: Invalid bandpass range: low={low}, high={high}. Returning passthrough filter.")
            return np.array([1.0]), np.array([1.0])
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)
    
    # Apply the filter (0.5-40 Hz)
    filtered = apply_bandpass_filter(raw, 0.5, 40.0, fs)
    envelope = np.abs(np.asarray(hilbert(filtered)))
    prom_env = 0.2 * np.std(envelope)

    # Reference method
    t0 = time.time()
    peaks_ma = find_peaks_moving_average_envelope(filtered, fs, prominence=prom_env)
    t1 = time.time()
    # Improved method
    t2 = time.time()
    peaks_sg = find_peaks_savgol_envelope(filtered, fs, prominence=prom_env)
    t3 = time.time()

    # --- Subplot layout: 1. Filtered ECG, 2. Envelope, 3. Envelope+Peaks ---
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Filtered ECG
    axs[0].plot(t, filtered, color='C0', label='Filtered ECG')
    axs[0].set_title('Filtered ECG Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Hilbert Envelope
    axs[1].plot(t, envelope, color='C1', label='Envelope (Hilbert)')
    axs[1].set_title('Hilbert Envelope')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Envelope smoothed by moving average, with peaks from two methods
    smoothed_ma = NumpySignalProcessor.moving_average(envelope, window_size=5)
    # Pad smoothed_ma to match length if needed
    if len(smoothed_ma) < len(envelope):
        smoothed_ma = np.pad(smoothed_ma, (0, len(envelope)-len(smoothed_ma)), mode='edge')
    axs[2].plot(t, smoothed_ma, color='C2', label='Moving Average Envelope')
    # Peaks: method 1 (NumpySignalProcessor.find_peaks)
    threshold = 0.30 * np.max(smoothed_ma)
    peaks_np = NumpySignalProcessor.find_peaks(smoothed_ma, fs, threshold=threshold, prominence=prom_env)
    axs[2].scatter(t[peaks_np], smoothed_ma[peaks_np], marker='o', facecolors='none', edgecolors='C4', label='Peaks (NumpySignalProcessor)', s=60)
    # Peaks: method 2 (find_peaks2)
    peaks_np2 = find_peaks2(smoothed_ma, fs, window=20, prominence=prom_env, threshold=threshold)
    axs[2].scatter(t[peaks_np2], smoothed_ma[peaks_np2], marker='x', color='C5', label='Peaks (find_peaks2)', s=60)
    axs[2].set_title('Envelope + Peak Detection (Two Methods)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print benchmark
    print("\nBenchmark Table:")
    print(f"{'Method':<30} {'Peaks':>8}")
    print("-"*40)
    print(f"NumpySignalProcessor.find_peaks{'':<5} {len(peaks_np):>8}")
    print(f"find_peaks2{'':<25} {len(peaks_np2):>8}")

if __name__ == "__main__":
    main()
