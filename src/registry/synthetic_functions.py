"""
synthetic_functions.py
- Place to define custom synthetic signal generator functions for use with SyntheticSignalGenerator.
- Add your own functions here and import as needed.
"""
import numpy as np
from scipy.signal import chirp
from scipy.signal import sweep_poly

def ecg_waveform(t, sample_index, stress_period=30.0):
    # Heart rate modulation to simulate stress/calm transitions
    # Baseline: 65 bpm, Stress: up to 80 bpm
    base_hr = 65
    stress_hr = 80
    # Use a smooth sinusoidal modulation for HR between base_hr and stress_hr
    stress_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / stress_period)
    current_hr = base_hr + (stress_hr - base_hr) * stress_factor

    # Clamp HR to avoid overshooting due to floating point errors
    current_hr = np.clip(current_hr, base_hr, stress_hr)

    # Convert HR to frequency in Hz
    freq = current_hr / 60.0

    # Calculate phase with heart rate variability
    hrv = 0.005 * np.random.normal()  # Smaller random variation in timing
    phase = ((t + hrv) * freq) % 1.0

    # ECG components - mainly focused on R peak with subtle P and T waves
    p_wave = 0.12 * np.exp(-70 * (phase - 0.08)**2)  # Subtle P wave
    qrs_complex = 0.9 * np.exp(-400 * (phase - 0.22)**2)  # Sharper R peak
    t_wave = 0.18 * np.exp(-60 * (phase - 0.35)**2)  # Subtle T wave

    # Combine components and add noise
    baseline = -0.01 * np.cos(2 * np.pi * phase)  # Subtle baseline drift
    noise = 0.01 * np.random.normal()  # Electrode noise

    value = baseline + p_wave + qrs_complex + t_wave + noise
    return value

def eda_waveform(t, sample_index):
    # Very slow baseline drift
    slow_component = 5 + 5 * np.sin(2 * np.pi * 0.005 * t)
    # Occasional EDA responses with realistic rise/decay times
    peak_interval = 8.0  # seconds between possible peaks
    amplitude = 3
    rise_time = 0.5
    decay_time = 3.0
    
    # Generate peaks at regular intervals, but jitter their timing
    value = slow_component
    for i in range(-2, 3):
        peak_time = (np.floor(t / peak_interval) + i) * peak_interval 
        
        # Create asymmetric peak with specified rise and decay times
        if t >= peak_time:
            # Decay phase (exponential decay)
            time_since_peak = t - peak_time
            peak_value = amplitude * np.exp(-time_since_peak / decay_time)
        else:
            # Rise phase (exponential rise)
            time_to_peak = peak_time - t
            if time_to_peak <= rise_time * 3:  # Only include rise within reasonable range
                peak_value = amplitude * (1 - np.exp(-((rise_time * 3 - time_to_peak) / rise_time)))
            else:
                peak_value = 0
        
        value += peak_value
    

    return value

def rr_waveform(t, sample_index, noise_level=0.05, period=30.0):
    # Make the waveform continuous by symmetric extension (mirror at period)
    t_mod = t % (2 * period)
    if t_mod < period:
        value = chirp(t_mod, f0=0.05, f1=0.7, t1=period, method='linear')
    else:
        # Mirror: play the chirp backward for the second half
        value = chirp(2 * period - t_mod, f0=0.7, f1=0.05, t1=period, method='linear')
    value += np.random.normal(0, noise_level)
    return value


def sine_waveform(t, sample_index, frequency=0.5, noise_level=0.1):
    value = np.sin(2 * np.pi * frequency * t)
    value = value * 0.9 + np.random.normal(0, noise_level)
    return value


# Add more custom synthetic functions below as needed.
