import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import label
import time
from src.utils.signal_processing import NumpySignalProcessor

# --- Synthetic ECG waveform generator (from your code) ---
def ecg_waveform(t):
    hr = 70 + 5 * np.sin(2 * np.pi * 0.1 * t)
    rr = 60.0 / hr
    beat_times = [0.0]
    while beat_times[-1] < t[-1] + 2.0:
        beat_times.append(beat_times[-1] + rr[min(int(beat_times[-1]*len(t)/t[-1]), len(rr)-1)])
    beat_times = np.array(beat_times)
    idx = np.searchsorted(beat_times, t, side='right') - 1
    idx = np.clip(idx, 0, len(beat_times)-2)
    phase = (t - beat_times[idx]) / (beat_times[idx+1] - beat_times[idx])
    p = 0.15 * np.exp(-((phase - 0.10) ** 2) / (2 * 0.015 ** 2))
    q = -0.2 * np.exp(-((phase - 0.22) ** 2) / (2 * 0.005 ** 2))
    r = 0.4 * np.exp(-((phase - 0.28) ** 2) / (2 * 0.006 ** 2))
    s = -0.2 * np.exp(-((phase - 0.32) ** 2) / (2 * 0.006 ** 2))
    t_wave = 0.2 * np.exp(-((phase - 0.55) ** 2) / (2 * 0.025 ** 2))
    baseline = 0.01 * np.sin(2 * np.pi * 0.18 * t)
    noise = 0.03 * np.random.randn(len(t))
    
    # Add two outliers at approximately 30% and 70% of the total time
    if len(t) > 0:
        idx_30pct = int(len(t) * 0.3)
        idx_70pct = int(len(t) * 0.7)
        
        # Create outliers by adding spikes
        p[idx_30pct:idx_30pct+10] += 0.3  # Add a spike to the P-wave area
        r[idx_70pct:idx_70pct+10] += 0.6  # Add a large spike to the R-wave area
        
    # Combine all components to form the ECG signal
    ecg = p + q + r + s + t_wave + baseline + noise
    return ecg

def bandpass_filter(signal, fs, low=8, high=18, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

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
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    if mode == "qrs":
        threshold = 0.99 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks = NumpySignalProcessor.find_peaks(filtered_signal, fs, threshold=threshold, prominence=prominence)
    return np.array(r_peaks, dtype=int)

def detect_r_peaks2(filtered_signal, fs, mode="qrs", prominence=None):
    """
    
    Parameters:
    - filtered_signal: The filtered ECG signal.
    - fs: Sampling frequency in Hz.
    - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
    - prominence: Minimum prominence of peaks to detect (default: None).
    
    Returns:
    - r_peaks: Indices of detected R-peaks.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if mode == "qrs":
        # Dynamic threshold close to the maximum value
        threshold = 0.9 * np.max(smoothed_envelope)
        # Adjust based on signal variability
        threshold = min(threshold, np.median(smoothed_envelope) + 2.5 * np.std(smoothed_envelope))
    elif mode == "all":
        # Higher dynamic threshold for full complex, closer to maximum
        threshold = 0.85 * np.max(smoothed_envelope)
        # Adjust based on signal characteristics
        threshold = min(threshold, np.median(smoothed_envelope) + 3.0 * np.std(smoothed_envelope))
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks = NumpySignalProcessor.find_peaks(smoothed_envelope, fs, threshold=threshold, prominence=prominence)
    return np.array(r_peaks, dtype=int)

def detect_r_peaks3(filtered_signal, fs, mode="qrs", prominence=None):
    """
    
    Parameters:
    - filtered_signal: The filtered ECG signal.
    - fs: Sampling frequency in Hz.
    - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
    - prominence: Minimum prominence of peaks to detect (default: None).
    
    Returns:
    - r_peaks: Indices of detected R-peaks.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    # Find peaks on both signals
    peaks_env = NumpySignalProcessor.find_peaks(smoothed_envelope, fs, threshold=threshold, prominence=prominence)
    peaks_filt = NumpySignalProcessor.find_peaks(filtered_signal, fs, threshold=threshold, prominence=prominence)
    max_dist = int(0.05 * fs)
    valid = [fp for fp in peaks_filt if np.any(np.abs(peaks_env - fp) <= max_dist)]
    return np.array(valid, dtype=int)

def detect_r_peaks4(filtered_signal, fs, mode="qrs", prominence=None):
    """
    
    Parameters:
    - filtered_signal: The filtered ECG signal.
    - fs: Sampling frequency in Hz.
    - mode: Detection mode. "qrs" for QRS complex, "all" for the entire heart complex.
    - prominence: Minimum prominence of peaks to detect (default: None).
    
    Returns:
    - r_peaks: Indices of detected R-peaks.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    checksignal = np.abs(filtered_signal - smoothed_envelope)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks = NumpySignalProcessor.find_peaks(checksignal, fs, threshold=threshold, prominence=prominence)
    return np.array(r_peaks, dtype=int)

def detect_r_peaks_scipy(filtered_signal, fs, mode="qrs", prominence=None):
    """
    SciPy version: Detect R-peaks directly on the filtered ECG signal.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks, _ = find_peaks(filtered_signal, distance=int(0.2*fs), prominence=prominence)
    return np.array(r_peaks, dtype=int)

def detect_r_peaks2_scipy(filtered_signal, fs, mode="qrs", prominence=None):
    """
    SciPy version: Detect R-peaks on the smoothed envelope.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks, _ = find_peaks(smoothed_envelope, distance=int(0.2*fs), prominence=prominence)
    return np.array(r_peaks, dtype=int)

def detect_r_peaks3_scipy(filtered_signal, fs, mode="qrs", prominence=None):
    """
    SciPy version: Find peaks on both filtered and smoothed envelope, then optionally combine/validate.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks_env, _ = find_peaks(smoothed_envelope, distance=int(0.2*fs), prominence=prominence)
    r_peaks_filt, _ = find_peaks(filtered_signal, distance=int(0.2*fs), prominence=prominence)
    max_dist = int(0.05 * fs)
    valid = [fp for fp in r_peaks_filt if np.any(np.abs(r_peaks_env - fp) <= max_dist)]
    return np.array(valid, dtype=int)

def detect_r_peaks4_scipy(filtered_signal, fs, mode="qrs", prominence=None):
    """
    SciPy version: Find peaks on |filtered - smoothed_envelope|.
    """
    envelope = np.abs(np.asarray(hilbert(filtered_signal)))
    smoothed_envelope = NumpySignalProcessor.moving_average(envelope, window_size=5)
    checksignal = np.abs(filtered_signal - smoothed_envelope)
    if mode == "qrs":
        threshold = 0.30 * np.max(envelope)
    elif mode == "all":
        threshold = 0.95 * np.max(envelope)
    else:
        raise ValueError("Invalid mode. Use 'qrs' for QRS complex or 'all' for the entire heart complex.")
    r_peaks, _ = find_peaks(checksignal, distance=int(0.2*fs), prominence=prominence)
    return np.array(r_peaks, dtype=int)


def main():
    fs = 1000
    t = np.arange(0, 5, 1/fs)
    raw = ecg_waveform(t)
    filtered = bandpass_filter(raw, fs)*100000
    prom_filt = 0.2 * np.std(filtered)

    # --- Moving average BEFORE Hilbert envelope ---
    ma_before = NumpySignalProcessor.moving_average(filtered, window_size=5)
    # Pad to match length
    if len(ma_before) < len(filtered):
        ma_before = np.pad(ma_before, (0, len(filtered)-len(ma_before)), mode='edge')
    envelope_ma_before = np.abs(np.asarray(hilbert(ma_before)))
    peaks_ma_before = NumpySignalProcessor.find_peaks(envelope_ma_before, fs, threshold=0.3*np.max(envelope_ma_before), prominence=prom_filt)

    # --- Moving average AFTER Hilbert envelope ---
    envelope = np.abs(np.asarray(hilbert(filtered)))
    ma_after = NumpySignalProcessor.moving_average(envelope, window_size=5)
    if len(ma_after) < len(envelope):
        ma_after = np.pad(ma_after, (0, len(envelope)-len(ma_after)), mode='edge')
    peaks_ma_after = NumpySignalProcessor.find_peaks(ma_after, fs, threshold=0.3*np.max(ma_after), prominence=prom_filt)

    # --- Compute peaks for all four cases ---
    # 1. Peaks on filtered
    peaks_filt = NumpySignalProcessor.find_peaks(filtered, fs, threshold=0.7*np.max(filtered), prominence=prom_filt)
    # 2. Peaks on envelope (smoothed)
    peaks_env = NumpySignalProcessor.find_peaks(ma_after, fs, threshold=0.7*np.max(ma_after), prominence=prom_filt)
    # 3. Peaks on filtered that are near envelope peaks
    max_dist = int(0.05 * fs)
    peaks_near_env = [fp for fp in peaks_filt if np.any(np.abs(peaks_env - fp) <= max_dist)]
    # 4. Peaks on |filtered - envelope|
    checksignal = np.abs(filtered - ma_after)
    peaks_diff = NumpySignalProcessor.find_peaks(checksignal, fs, threshold=0.3*np.max(checksignal), prominence=prom_filt)

    # --- 4-panel plot ---
    fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    # 1. Peaks on filtered
    axs[0].plot(t, filtered, label='Filtered ECG', color='C0')
    axs[0].scatter(t[peaks_filt], filtered[peaks_filt], marker='o', facecolors='none', edgecolors='C3', label='Peaks (filtered)', s=60)
    axs[0].set_title('Peaks on Filtered Signal')
    axs[0].legend(); axs[0].grid(True)
    # 2. Peaks on envelope
    axs[1].plot(t, filtered, label='Filtered ECG', color='C0', alpha=0.5)
    axs[1].plot(t, ma_after, label='Smoothed Envelope', color='C2')
    axs[1].scatter(t[peaks_env], ma_after[peaks_env], marker='x', color='C4', label='Peaks (envelope)', s=60)
    axs[1].set_title('Peaks on Envelope (Smoothed)')
    axs[1].legend(); axs[1].grid(True)
    # 3. Peaks on filtered near envelope peaks
    axs[2].plot(t, filtered, label='Filtered ECG', color='C0')
    axs[2].scatter(t[peaks_near_env], filtered[peaks_near_env], marker='s', facecolors='none', edgecolors='C5', label='Peaks (filtered near envelope)', s=60)
    axs[2].set_title('Peaks on Filtered Near Envelope Peaks')
    axs[2].legend(); axs[2].grid(True)
    # 4. Peaks on |filtered - envelope|
    axs[3].plot(t, checksignal, label='|Filtered - Envelope|', color='C6')
    axs[3].scatter(t[peaks_diff], checksignal[peaks_diff], marker='^', color='C7', label='Peaks (|filtered-envelope|)', s=60)
    axs[3].set_title('Peaks on |Filtered - Envelope|')
    axs[3].legend(); axs[3].grid(True)
    plt.xlabel('Time (s)')
    plt.tight_layout(); plt.show()

    # --- Plot stepwise comparison ---
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    # 1. Filtered and moving averaged
    axs[0].plot(t, filtered, label='Filtered ECG', color='C0', alpha=0.7)
    axs[0].plot(t, ma_before, label='Moving Average (filtered)', color='C2')
    axs[0].set_title('Filtered ECG and Moving Average')
    axs[0].legend(); axs[0].grid(True)
    # 2. Envelope and envelope of moving averaged
    axs[1].plot(t, envelope, label='Envelope (Hilbert of filtered)', color='C1')
    axs[1].plot(t, envelope_ma_before, label='Envelope of MA(filtered)', color='C3')
    axs[1].set_title('Envelope (filtered) vs Envelope(MA(filtered))')
    axs[1].legend(); axs[1].grid(True)
    # 3. Moving average of envelope
    axs[2].plot(t, envelope, label='Envelope (Hilbert of filtered)', color='C1', alpha=0.5)
    axs[2].plot(t, ma_after, label='Moving Average of Envelope', color='C4')
    axs[2].set_title('Moving Average of Envelope (filtered)')
    axs[2].legend(); axs[2].grid(True)
    plt.xlabel('Time (s)')
    plt.tight_layout(); plt.show()

    print(f"Peaks (MA before envelope): {len(peaks_ma_before)}")
    print(f"Peaks (MA after envelope):  {len(peaks_ma_after)}")

if __name__ == "__main__":
    main()