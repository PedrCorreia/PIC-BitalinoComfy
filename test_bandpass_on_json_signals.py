import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load signals from JSON files
def load_signal(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Handle empty file
    if not data:
        print(f"{json_path} is empty.")
        return None
    # If list of dicts with 'data' key, extract first value from each 'data' list
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        arr = []
        for d in data:
            if isinstance(d, dict) and "data" in d and isinstance(d["data"], list):
                arr.append(d["data"][0])
        return np.array(arr)
    # If list of dicts, extract first value from each dict (fallback)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        arr = []
        for d in data:
            if isinstance(d, dict):
                arr.append(list(d.values())[0])
        return np.array(arr)
    # If dict, try common keys
    if isinstance(data, dict):
        for key in ['signal', 'ecg', 'data', 'v', 'values']:
            if key in data:
                return np.array(data[key])
        # If dict but no known key, try first value
        first_val = list(data.values())[0]
        if isinstance(first_val, list):
            return np.array(first_val)
        elif isinstance(first_val, dict):
            # Nested dict, try to extract values
            return np.array(list(first_val.values()))
        else:
            return np.array(first_val)
    print(f"Unknown data format in {json_path}")
    return None

# File paths
heart_path = "src/phy/ECG/heart_signal_data.json"
collarbone_path = "src/phy/ECG/collarbone_signal_data.json"

# Load signals
try:
    heart_signal = load_signal(heart_path)
except Exception as e:
    print(f"Could not load heart signal: {e}")
    heart_signal = None
try:
    collarbone_signal = load_signal(collarbone_path)
except Exception as e:
    print(f"Could not load collarbone signal: {e}")
    collarbone_signal = None

# Bandpass filter parameters
fs = 1000  # Hz (adjust if known)
lowcut = 8
highcut = 15
order = 4
b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

# Plot and filter each signal if loaded
for name, signal in [("Heart", heart_signal), ("Collarbone", collarbone_signal)]:
    if signal is not None and len(signal) > 10:
        filtered = filtfilt(b, a, signal)
        # Normalize for comparison
        signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) if np.max(signal) != np.min(signal) else signal
        filtered_norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered)) if np.max(filtered) != np.min(filtered) else filtered
        t = np.arange(len(signal)) / fs
        plt.figure(figsize=(14, 6))
        plt.plot(t, signal_norm, label=f'{name} Raw (normalized)', alpha=0.5)
        plt.plot(t, filtered_norm, label=f'{name} Bandpassed (8-20 Hz, normalized)', color='red', linewidth=1.2)
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Signal')
        plt.title(f'{name} Signal: Raw vs Bandpassed (8-20 Hz, Butterworth, filtfilt)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"No valid data for {name} signal or too short.")
