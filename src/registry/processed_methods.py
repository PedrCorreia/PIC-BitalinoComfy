"""
processed_methods.py
- Place to define custom signal processing methods for use with ProcessedSignalGenerator.
- Add your own processing functions here and import as needed.
"""
import numpy as np

def moving_average(signal, window_size=5):
    if len(signal) < window_size:
        return signal
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def normalize(signal):
    if len(signal) == 0:
        return signal
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

# Add more custom processing methods below as needed.
