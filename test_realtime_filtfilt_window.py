import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from scipy.signal import butter, lfilter, filtfilt
from src.phy.ecg_signal_processing import ECG
from src.utils.signal_processing import NumpySignalProcessor

# --- Synthetic ECG waveform generator (from your code) ---
def ecg_waveform(t):
    hr_min_calm = 60.0
    hr_max_calm = 75.0
    hr_min_stress = 85.0
    hr_max_stress = 120.0
    stress_cycle_duration = 15
    stress_phase = (t % stress_cycle_duration) / stress_cycle_duration
    stress_level = 0.5 - 0.5 * np.cos(2 * np.pi * stress_phase)
    hr_min = hr_min_calm + stress_level * (hr_min_stress - hr_min_calm)
    hr_max = hr_max_calm + stress_level * (hr_max_stress - hr_max_calm)
    hrv_factor = 0.08 * (1 - stress_level * 0.7)
    hr_variability = hrv_factor * np.sin(2 * np.pi * 0.05 * t)
    target_hr = hr_min + (hr_max - hr_min) * 0.5 * (1 + np.sin(2 * np.pi * 0.01 * t)) + hr_variability
    duration = t[-1] if hasattr(t, '__getitem__') else t
    t_max = np.max(t)
    beat_times = [0.0]
    current_hr = target_hr[0] if hasattr(target_hr, '__getitem__') else target_hr
    while beat_times[-1] < t_max + 2.0:
        elapsed = beat_times[-1]
        elapsed_stress_phase = (elapsed % stress_cycle_duration) / stress_cycle_duration
        elapsed_stress = 0.5 - 0.5 * np.cos(2 * np.pi * elapsed_stress_phase)
        elapsed_hr_min = hr_min_calm + elapsed_stress * (hr_min_stress - hr_min_calm)
        elapsed_hr_max = hr_max_calm + elapsed_stress * (hr_max_stress - hr_max_calm)
        elapsed_hrv = 0.08 * (1 - elapsed_stress * 0.7) * np.sin(2 * np.pi * 0.05 * elapsed)
        current_hr = elapsed_hr_min + (elapsed_hr_max - elapsed_hr_min) * 0.5 * (1 + np.sin(2 * np.pi * 0.01 * elapsed)) + elapsed_hrv
        rr_interval = 60.0 / current_hr
        beat_times.append(beat_times[-1] + rr_interval)
    beat_times = np.array(beat_times)
    idx = np.searchsorted(beat_times, t, side='right') - 1
    idx = np.clip(idx, 0, len(beat_times)-2)
    t_since_beat = t - beat_times[idx]
    rr_this = beat_times[idx+1] - beat_times[idx]
    phase = t_since_beat / rr_this
    stress_at_t = 0.5 - 0.5 * np.cos(2 * np.pi * (t % stress_cycle_duration) / stress_cycle_duration)
    p_amp = 0.15 * (1 + 0.1 * stress_at_t)
    t_amp = 0.2 * (1 - 0.2 * stress_at_t)
    qt_factor = 1.0 - (0.08 * stress_at_t)
    p = p_amp * np.exp(-((phase - 0.10) ** 2) / (2 * 0.015 ** 2))
    q = -0.2 * np.exp(-((phase - 0.22) ** 2) / (2 * 0.005 ** 2))
    r = 0.6 * np.exp(-((phase - 0.28) ** 2) / (2 * 0.006 ** 2))
    s = -0.2 * np.exp(-((phase - 0.32) ** 2) / (2 * 0.006 ** 2))
    t_wave = t_amp * np.exp(-((phase - (0.55 * qt_factor)) ** 2) / (2 * 0.025 ** 2))
    baseline = 0.005 * (1 - 0.3 * stress_at_t) * np.sin(2 * np.pi * 0.18 * t)
    noise_level = 0.02 + 0.02 * stress_at_t
    value = p + q + r + s + t_wave + baseline
    value = value + np.random.normal(0, noise_level, size=np.shape(t))
    return value

# --- Real-time filtfilt processing in a moving window (threaded) ---
class RealTimeFiltFiltProcessor:
    def __init__(self, fs, window_sec=2.0, step_sec=0.1):
        self.fs = fs
        self.window_size = int(window_sec * fs)
        self.step_size = int(step_sec * fs)
        self.signal = []
        self.filtered = []
        self.envelope = []
        self.peaks = []
        self.t = []
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def add_data(self, t_chunk, signal_chunk):
        with self.lock:
            self.t.extend(t_chunk)
            self.signal.extend(signal_chunk)

    def _process_loop(self):
        while self.running:
            with self.lock:
                if len(self.signal) < self.window_size:
                    time.sleep(0.05)
                    continue
                t_win = np.array(self.t[-self.window_size:])
                sig_win = np.array(self.signal[-self.window_size:])
            # Filtfilt processing on the window
            filtered, envelope, peaks = ECG.preprocess_signal(
                sig_win, self.fs, mode="qrs", bandpass_low=8, bandpass_high=15, envelope_smooth=5, visualization=True, gain=5
            )[:3]
            with self.lock:
                self.filtered = filtered
                self.envelope = envelope
                self.peaks = peaks
            time.sleep(0.01)

if __name__ == "__main__":
    fs = 1000
    duration = 30
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    ecg = ecg_waveform(t)
    processor = RealTimeFiltFiltProcessor(fs, window_sec=2.0, step_sec=0.1)
    processor.start()
    chunk_size = int(0.1 * fs)
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    for i in range(0, len(t), chunk_size):
        t_chunk = t[i:i+chunk_size]
        sig_chunk = ecg[i:i+chunk_size]
        processor.add_data(t_chunk.tolist(), sig_chunk.tolist())
        # Plot in main thread
        with processor.lock:
            if len(processor.t) >= processor.window_size:
                t_win = np.array(processor.t[-processor.window_size:])
                sig_win = np.array(processor.signal[-processor.window_size:])
                filtered = np.array(processor.filtered) if len(processor.filtered) == processor.window_size else None
                envelope = np.array(processor.envelope) if len(processor.envelope) == processor.window_size else None
                peaks = np.array(processor.peaks) if processor.peaks is not None else []
                ax.clear()
                ax.set_facecolor('white')
                ax.plot(t_win, sig_win, label='Raw', alpha=0.3)
                if filtered is not None:
                    ax.plot(t_win, filtered, label='FiltFilt', color='blue')
                if envelope is not None:
                    ax.plot(t_win, envelope, label='Envelope', color='orange')
                if len(peaks) > 0 and filtered is not None:
                    ax.scatter(t_win[peaks], filtered[peaks], color='red', label='Peaks', zorder=10)
                ax.legend()
                ax.set_title('Real-Time FiltFilt Processing (Moving Window)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (a.u.)')
        plt.pause(0.01)
        time.sleep(0.1)
    processor.stop()
    plt.ioff()
    plt.show()
    
def test_filtfilt_improvement():
    """Compare the effectiveness of filtfilt vs lfilter for live ECG processing"""
    print("\nTesting improved filtfilt implementation for live processing...")
    
    # Generate test ECG signal
    duration = 5  # 5 seconds
    fs = 1000     # 1000 Hz
    t = np.arange(0, duration, 1/fs)
    ecg_signal = ecg_waveform(t)
    
    # Add 50Hz power line noise + high frequency noise
    noise_50hz = 0.3 * np.sin(2 * np.pi * 50 * t)
    noise_high = 0.1 * np.sin(2 * np.pi * 120 * t)
    ecg_signal_noisy = ecg_signal + noise_50hz + noise_high
    
    # Configure filter parameters
    bandpass_low = 8
    bandpass_high = 15
    order = 4
    
    # Process with traditional lfilter
    zi_lfilter = None
    chunk_size = 200  # Simulate 200ms chunks
    lfilter_results = []
    
    for i in range(0, len(ecg_signal_noisy), chunk_size):
        chunk = ecg_signal_noisy[i:i+chunk_size]
        # Simulate old approach using lfilter
        nyquist = 0.5 * fs
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist
        b, a = butter(order, [low, high], btype='band')
        if zi_lfilter is None:
            zi_lfilter = np.zeros(max(len(a), len(b)) - 1)
        filtered_chunk, zi_lfilter = lfilter(b, a, chunk, zi=zi_lfilter)
        lfilter_results.append(filtered_chunk)
    
    lfilter_output = np.concatenate(lfilter_results)
    
    # Process with improved filtfilt approach
    filtfilt_results = []
    for i in range(0, len(ecg_signal_noisy), chunk_size):
        chunk = ecg_signal_noisy[i:i+chunk_size]
        # Get filtered chunk using our new implementation
        filtered_chunk, _ = NumpySignalProcessor.bandpass_filter(
            chunk, bandpass_low, bandpass_high, fs, order=order, live=True
        )
        filtfilt_results.append(filtered_chunk)
    
    filtfilt_output = np.concatenate(filtfilt_results)
    
    # Reference: Process full signal with traditional filtfilt
    reference_output, _ = NumpySignalProcessor.bandpass_filter(
        ecg_signal_noisy, bandpass_low, bandpass_high, fs, order=order, live=False
    )
    
    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Raw signal
    axes[0].plot(t, ecg_signal_noisy)
    axes[0].set_title('Raw ECG Signal with Noise')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # lfilter result
    axes[1].plot(t, lfilter_output)
    axes[1].set_title('Traditional lfilter (Live Processing)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    # filtfilt result
    axes[2].plot(t, filtfilt_output)
    axes[2].set_title('Improved filtfilt (Live Processing)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True)
    
    # reference result
    axes[3].plot(t, reference_output)
    axes[3].set_title('Reference filtfilt (Offline Processing)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].grid(True)
    
    plt.tight_layout()
    
    # Calculate metrics to quantify improvement
    from scipy.signal import correlate
    
    # Phase delay between live filtfilt and reference
    corr_filtfilt = correlate(filtfilt_output, reference_output, mode='full')
    lag_filtfilt = np.argmax(corr_filtfilt) - (len(reference_output) - 1)
    
    # Phase delay between live lfilter and reference
    corr_lfilter = correlate(lfilter_output, reference_output, mode='full')
    lag_lfilter = np.argmax(corr_lfilter) - (len(reference_output) - 1)
    
    print(f"Phase lag with traditional lfilter: {lag_lfilter} samples")
    print(f"Phase lag with improved filtfilt: {lag_filtfilt} samples")
    print(f"Improvement: {100 * (abs(lag_lfilter) - abs(lag_filtfilt)) / abs(lag_lfilter):.1f}% reduction in phase lag")
    
    # Calculate SNR for both methods
    def calculate_snr(signal, reference):
        noise = signal - reference
        return 10 * np.log10(np.sum(reference**2) / np.sum(noise**2))
    
    snr_lfilter = calculate_snr(lfilter_output, reference_output)
    snr_filtfilt = calculate_snr(filtfilt_output, reference_output)
    
    print(f"SNR with traditional lfilter: {snr_lfilter:.2f} dB")
    print(f"SNR with improved filtfilt: {snr_filtfilt:.2f} dB")
    print(f"SNR improvement: {snr_filtfilt - snr_lfilter:.2f} dB")
    
    plt.figure(figsize=(12, 6))
    
    # FFT of signals
    from scipy.fft import fft, fftfreq
    n = len(t)
    yf_raw = fft(ecg_signal_noisy)
    yf_lfilter = fft(lfilter_output)
    yf_filtfilt = fft(filtfilt_output)
    yf_ref = fft(reference_output)
    xf = fftfreq(n, 1/fs)[:n//2]
    
    plt.plot(xf, 2.0/n * np.abs(yf_raw[:n//2]), label='Raw Signal')
    plt.plot(xf, 2.0/n * np.abs(yf_lfilter[:n//2]), label='lfilter')
    plt.plot(xf, 2.0/n * np.abs(yf_filtfilt[:n//2]), label='Improved filtfilt')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Response Comparison')
    plt.legend()
    plt.xlim(0, 50)
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    # Uncomment to run the main streaming simulation
    # simulate_ecg_streaming()
    
    # Run the filter improvement test
    test_filtfilt_improvement()
