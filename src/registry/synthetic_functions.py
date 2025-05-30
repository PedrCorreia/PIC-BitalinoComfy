"""
synthetic_functions.py
- Place to define custom synthetic signal generator functions for use with SyntheticSignalGenerator.
- Add your own functions here and import as needed.
"""
import numpy as np
from scipy.signal import chirp

def ecg_waveform(t, sample_index, stress_period=30.0):
    # Heart rate modulation to simulate stress/calm transitions
    # Baseline: 65 bpm, Stress: up to 95 bpm
    base_hr = 50.0
    stress_hr =80
    stress_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / stress_period)
    current_hr = base_hr + (stress_hr - base_hr) * stress_factor
    
    # Convert HR to frequency in Hz
    freq = current_hr / 60.0
    
    # Calculate phase with heart rate variability
    hrv = 0.03 * np.random.normal()  # Small random variation in timing
    phase = (t * freq + hrv) % 1.0
    
    # ECG components - mainly focused on R peak with subtle P and T waves
    p_wave = 0.15 * np.exp(-70 * (phase - 0.08)**2)  # Subtle P wave
    qrs_complex = 0.8 * np.exp(-300 * (phase - 0.22)**2)  # Prominent R peak
    t_wave = 0.2 * np.exp(-70 * (phase - 0.35)**2)  # Subtle T wave
    
    # Combine components and add noise
    baseline = -0.02 * np.cos(2 * np.pi * phase)  # Subtle baseline drift
    noise = 0.02 * np.random.normal()  # Electrode noise
    
    value = baseline + p_wave + qrs_complex + t_wave + noise
    return value

def eda_waveform(t, sample_index):
    slow_component = 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    random_walk = np.sin(0.5 * t) * np.random.normal(0, 0.02)
    value = slow_component + random_walk
    return value

def rr_waveform(t, sample_index, noise_level=0.05):
    # Continuous chirp frequency from 0.1 Hz up to 1 Hz and back down over a 60s period
    cycle_period = 60.0
    # Calculate the position within the cycle
    phase = (t % cycle_period) / cycle_period
    duration = cycle_period / 2
    # Use a single chirp function with a sawtooth time base for continuity
    if phase < 0.5:
        # Up-chirp: 0.1 Hz to 1.0 Hz
        t_chirp = phase * cycle_period
        value = chirp(t_chirp, f0=0.1, f1=1.0, t1=duration, method='linear')
    else:
        # Down-chirp: 1.0 Hz to 0.1 Hz
        t_chirp = (phase - 0.5) * cycle_period
        value = chirp(t_chirp, f0=1.0, f1=0.1, t1=duration, method='linear')
    # Make the waveform continuous at the cycle boundaries by matching the start/end values
    # Optionally, you can use a cosine window to smooth the transitions
    value += np.random.normal(0, noise_level)
    return value


def sine_waveform(t, sample_index, frequency=0.5, noise_level=0.1):
    value = np.sin(2 * np.pi * frequency * t)
    value = value * 0.9 + np.random.normal(0, noise_level)
    return value


# Add more custom synthetic functions below as needed.
