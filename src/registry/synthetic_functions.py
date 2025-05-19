"""
synthetic_functions.py
- Place to define custom synthetic signal generator functions for use with SyntheticSignalGenerator.
- Add your own functions here and import as needed.
"""
import numpy as np

def ecg_waveform(t, sample_index):
    base_freq = 1.2
    value = np.sin(2 * np.pi * base_freq * t)
    spike = np.exp(-80 * ((t * base_freq) % 1 - 0.2) ** 2)
    value = value * 0.3 + spike * 0.7
    value = value + np.random.normal(0, 0.05)
    return value

def eda_waveform(t, sample_index):
    slow_component = 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    random_walk = np.sin(0.5 * t) * np.random.normal(0, 0.02)
    value = slow_component + random_walk
    return value

def sine_waveform(t, sample_index, frequency=1.0, noise_level=0.1):
    value = np.sin(2 * np.pi * frequency * t)
    value = value * 0.9 + np.random.normal(0, noise_level)
    return value


# Add more custom synthetic functions below as needed.
