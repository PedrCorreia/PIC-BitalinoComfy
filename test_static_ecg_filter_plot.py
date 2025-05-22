import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from src.phy.ecg_signal_processing import ECG
from src.utils.signal_processing import NumpySignalProcessor
import os
import json

# --- ECG waveform function (static, from synthetic_functions.py) ---
def ecg_waveform(t):
    """Generate realistic ECG signal that cycles between calm and stressed states"""
    # More realistic heart rate ranges for healthy humans
    hr_min_calm = 60.0   # Minimum heart rate during calm (60 BPM)
    hr_max_calm = 75.0   # Maximum heart rate during calm (75 BPM)
    hr_min_stress = 85.0  # Minimum heart rate during stress (85 BPM)
    hr_max_stress = 120.0  # Maximum heart rate during stress (120 BPM)
    
    # Create longer stress-calm cycles (2-3 minutes per cycle)
    stress_cycle_duration = 15 # 2.5 minutes for full calm-to-stress-to-calm cycle
    stress_phase = (t % stress_cycle_duration) / stress_cycle_duration
    
    # Calculate stress level (0=calm, 1=stressed) with smooth transitions using cosine
    stress_level = 0.5 - 0.5 * np.cos(2 * np.pi * stress_phase)
    
    # Interpolate heart rate range based on stress level
    hr_min = hr_min_calm + stress_level * (hr_min_stress - hr_min_calm)
    hr_max = hr_max_calm + stress_level * (hr_max_stress - hr_max_calm)
    
    # Heart rate variability - higher during calm states
    hrv_factor = 0.08 * (1 - stress_level * 0.7)  
    hr_variability = hrv_factor * np.sin(2 * np.pi * 0.05 * t)
    
    # Current target heart rate with natural fluctuations
    target_hr = hr_min + (hr_max - hr_min) * 0.5 * (1 + np.sin(2 * np.pi * 0.01 * t)) + hr_variability
    
    # Generate beat times with this variable heart rate
    duration = t[-1] if hasattr(t, '__getitem__') else t
    t_max = np.max(t)
    beat_times = [0.0]
    current_hr = target_hr[0] if hasattr(target_hr, '__getitem__') else target_hr
    
    while beat_times[-1] < t_max + 2.0:
        # Get time since start
        elapsed = beat_times[-1]
        # Get stress level at this time
        elapsed_stress_phase = (elapsed % stress_cycle_duration) / stress_cycle_duration
        elapsed_stress = 0.5 - 0.5 * np.cos(2 * np.pi * elapsed_stress_phase)
        
        # Calculate HR at this time with variability
        elapsed_hr_min = hr_min_calm + elapsed_stress * (hr_min_stress - hr_min_calm)
        elapsed_hr_max = hr_max_calm + elapsed_stress * (hr_max_stress - hr_max_calm)
        elapsed_hrv = 0.08 * (1 - elapsed_stress * 0.7) * np.sin(2 * np.pi * 0.05 * elapsed)
        current_hr = elapsed_hr_min + (elapsed_hr_max - elapsed_hr_min) * 0.5 * (1 + np.sin(2 * np.pi * 0.01 * elapsed)) + elapsed_hrv
        
        # Add next beat time based on current HR
        rr_interval = 60.0 / current_hr
        beat_times.append(beat_times[-1] + rr_interval)
    
    beat_times = np.array(beat_times)
    
    # Find the beat interval for each time point
    idx = np.searchsorted(beat_times, t, side='right') - 1
    idx = np.clip(idx, 0, len(beat_times)-2)  # Ensure valid indices
    
    # Calculate phase within each beat
    t_since_beat = t - beat_times[idx]
    rr_this = beat_times[idx+1] - beat_times[idx]
    phase = t_since_beat / rr_this
    
    # Adjust ECG morphology based on stress level
    # (Higher stress = relatively smaller T wave, shorter QT interval, etc.)
    stress_at_t = 0.5 - 0.5 * np.cos(2 * np.pi * (t % stress_cycle_duration) / stress_cycle_duration)
    
    # P-QRS-T complex with stress-dependent morphology
    p_amp = 0.15 * (1 + 0.1 * stress_at_t)
    t_amp = 0.2 * (1 - 0.2 * stress_at_t)
    qt_factor = 1.0 - (0.08 * stress_at_t)
    
    # PQRST shape (normalized to each RR)
    p = p_amp * np.exp(-((phase - 0.10) ** 2) / (2 * 0.015 ** 2))
    q = -0.2 * np.exp(-((phase - 0.22) ** 2) / (2 * 0.005 ** 2))
    r = 0.6 * np.exp(-((phase - 0.28) ** 2) / (2 * 0.006 ** 2))
    s = -0.2 * np.exp(-((phase - 0.32) ** 2) / (2 * 0.006 ** 2))
    t_wave = t_amp * np.exp(-((phase - (0.55 * qt_factor)) ** 2) / (2 * 0.025 ** 2))
    
    # Baseline wander decreases during stress (sympathetic tone increases)
    baseline = 0.005 * (1 - 0.3 * stress_at_t) * np.sin(2 * np.pi * 0.18 * t)
    
    # Noise level increases slightly with stress (muscle artifacts)
    noise_level = 0.02 + 0.02 * stress_at_t
    
    # Combine all components
    value = p + q + r + s + t_wave + baseline
    value = value + np.random.normal(0, noise_level, size=np.shape(t))
    
    return value

if __name__ == "__main__":
    # Generate synthetic ECG data only (no JSON loading attempt)
    print("Generating synthetic ECG data...")
    fs = 1000  # Hz
    duration = 30  # seconds (showing 2 complete stress cycles)
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    ecg_signal = ecg_waveform(t)
    
    # Create time array based on signal length
    t = np.linspace(0, duration, len(ecg_signal), endpoint=False)
    
    # Simple processing pipeline using your existing methods:
    # Use the full ECG.preprocess_signal method for consistency
    # --- Static (filtfilt) filtering ---
    # Helper to unpack 3 or 4 return values
    def unpack_preprocess_signal(result):
        if len(result) == 4:
            return result[:3]
        return result

    # --- Static (filtfilt) filtering ---
    filtered_ecg_static, envelope_static, peaks_static = unpack_preprocess_signal(
        ECG.preprocess_signal(
            ecg_signal,
            fs, 
            mode="qrs",
            bandpass_low=8, 
            bandpass_high=15, 
            envelope_smooth=5,
            visualization=True,
            gain=5
        )
    )

    # --- Live (lfilter, rolling window) filtering ---
    window_sec = 2.0  # window size in seconds
    window_size = int(window_sec * fs)
    n_samples = len(ecg_signal)
    filtered_ecg_static_chunks = []
    envelope_static_chunks = []
    peaks_static_all = []
    filtered_ecg_live_chunks = []
    envelope_live_chunks = []
    peaks_live_all = []
    start = 0
    while start < n_samples:
        end = min(start + window_size, n_samples)
        chunk = ecg_signal[start:end]
        # Static (filtfilt) filtering on chunk
        result_static = ECG.preprocess_signal(
            chunk,
            fs,
            mode="qrs",
            bandpass_low=8,
            bandpass_high=15,
            envelope_smooth=5,
            visualization=True,
            gain=5
        )
        filtered_chunk_static, envelope_chunk_static, peaks_chunk_static = result_static[:3]
        filtered_ecg_static_chunks.append(filtered_chunk_static)
        envelope_static_chunks.append(envelope_chunk_static)
        if len(peaks_chunk_static) > 0:
            peaks_static_all.extend((np.array(peaks_chunk_static) + start).tolist())
        # Live (lfilter) filtering on chunk
        result_live = ECG.preprocess_signal(
            chunk,
            fs,
            mode="qrs",
            bandpass_low=8,
            bandpass_high=15,
            envelope_smooth=5,
            visualization=True,
            gain=5,
            live=True
        )
        filtered_chunk_live, envelope_chunk_live, peaks_chunk_live = result_live[:3]
        filtered_ecg_live_chunks.append(filtered_chunk_live)
        envelope_live_chunks.append(envelope_chunk_live)
        if len(peaks_chunk_live) > 0:
            peaks_live_all.extend((np.array(peaks_chunk_live) + start).tolist())
        start += window_size
    # Concatenate all chunks
    filtered_ecg_static = np.concatenate(filtered_ecg_static_chunks)
    envelope_static = np.concatenate(envelope_static_chunks)
    peaks_static = np.array(peaks_static_all)
    filtered_ecg_live = np.concatenate(filtered_ecg_live_chunks)
    envelope_live = np.concatenate(envelope_live_chunks)
    peaks_live = np.array(peaks_live_all)

    # Calculate heart rate from detected peaks (static and live)
    heart_rate_static = ECG.extract_heart_rate(filtered_ecg_static, fs, r_peaks=peaks_static)
    heart_rate_live = ECG.extract_heart_rate(filtered_ecg_live, fs, r_peaks=peaks_live)

    # Calculate instantaneous heart rate for trend visualization (static and live)
    def calc_inst_hr(peaks, t, fs):
        inst_hr = []
        inst_hr_times = []
        if len(peaks) > 1:
            for i in range(1, len(peaks)):
                rr_sec = (peaks[i] - peaks[i-1]) / fs
                inst_hr.append(60 / rr_sec)
                inst_hr_times.append(t[peaks[i-1]] + rr_sec/2)
        return inst_hr, inst_hr_times

    inst_hr_static, inst_hr_times_static = calc_inst_hr(peaks_static, t, fs)
    inst_hr_live, inst_hr_times_live = calc_inst_hr(peaks_live, t, fs)

    print(f"Static (filtfilt) Heart Rate: {heart_rate_static:.2f} bpm, Detected {len(peaks_static)} peaks")
    print(f"Live (lfilter) Heart Rate: {heart_rate_live:.2f} bpm, Detected {len(peaks_live)} peaks")

    # Plot with envelopes, peaks, and heart rate trend for both filter types
    plt.figure(figsize=(14, 14))

    # Plot 1: Raw signal with overlays of both filtered signals and stress level
    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_signal, label='Raw ECG', alpha=0.5, color='gray')
    plt.plot(t, filtered_ecg_static, label='Filtered (Static/filtfilt)', alpha=0.8, color='blue')
    plt.plot(t, filtered_ecg_live, label='Filtered (Live/lfilter)', alpha=0.8, color='orange', linestyle='--')
    # Stress level visualization
    stress_cycle_duration = 15.0
    stress_level = 0.5 - 0.5 * np.cos(2 * np.pi * (t % stress_cycle_duration) / stress_cycle_duration)
    scaled_stress = stress_level * 0.4
    plt.plot(t, scaled_stress - 0.5, label='Stress Level', color='red', linewidth=1.5, alpha=0.7)
    for i in range(1, int(duration/stress_cycle_duration) + 1):
        plt.axvline(i * stress_cycle_duration, color='k', linestyle='--', alpha=0.3)
        plt.text(i * stress_cycle_duration - 0.5, 0.8, f"Cycle {i}", fontsize=8)
    plt.title('Raw ECG with Both Filtered Outputs and Stress Cycles')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()

    # Plot 2: Heart rate trend for both filter types
    plt.subplot(3, 1, 2)
    if inst_hr_times_static:
        plt.plot(inst_hr_times_static, inst_hr_static, 'o-', color='blue', markersize=4, label='Inst. HR (Static)')
    if inst_hr_times_live:
        plt.plot(inst_hr_times_live, inst_hr_live, 'o-', color='orange', markersize=4, label='Inst. HR (Live)')
    plt.axhline(y=60, color='b', linestyle=':', alpha=0.5, label='Resting HR (60 BPM)')
    plt.axhline(y=120, color='r', linestyle=':', alpha=0.5, label='Peak Stress HR (120 BPM)')
    for i in range(1, int(duration/stress_cycle_duration) + 1):
        plt.axvline(i * stress_cycle_duration, color='k', linestyle='--', alpha=0.3)
    plt.title('Heart Rate Trend: Static vs Live Filtering')
    plt.ylabel('Heart Rate (BPM)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Envelope and peaks for both filter types
    plt.subplot(3, 1, 3)
    plt.plot(t, envelope_static, label='Envelope (Static)', color='blue', alpha=0.7)
    plt.plot(t, envelope_live, label='Envelope (Live)', color='orange', alpha=0.7, linestyle='--')
    if len(peaks_static) > 0:
        plt.scatter(t[peaks_static], envelope_static[peaks_static], color='blue', marker='x', s=80, label='Peaks (Static)')
    if len(peaks_live) > 0:
        plt.scatter(t[peaks_live], envelope_live[peaks_live], color='orange', marker='o', s=40, label='Peaks (Live)', facecolors='none', edgecolors='orange')
    plt.title(f'Envelope and Peak Detection: Static vs Live (Static HR: {heart_rate_static:.1f} BPM, Live HR: {heart_rate_live:.1f} BPM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()

    plt.tight_layout()
    plt.show()