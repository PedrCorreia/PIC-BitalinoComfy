import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from src.utils.bitalino_receiver_PIC import BitalinoReceiver
from src.utils.bitalino_receiver_LUNA import BitalinoReceiver as LunaReceiver
from scipy.signal import butter, filtfilt
import json
import os

# BITalino acquisition setup
bitalino_mac_address = "BTH20:16:07:18:17:02"  # Change to your device MAC
acquisition_duration = 20  # seconds
sampling_freq = 1000
channel_code = 0x01  # Only first channel (ECG)
buffer_size = 5000
bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)

# Filtering parameters for QRS accentuation - MATCH EXACTLY WITH JSON TEST
lowcut = 8
highcut = 15
order = 4
fs = 1000  # Match sampling frequency 
b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

# Acquisition
start_time = time.time()
raw_deque = deque()
timestamps_deque = deque()
last_sampled_ts = None
sample_counter = 0

print("Starting BITalino ECG acquisition...")

# --- NEW: Collect ALL samples from the buffer, not just the latest ---
while time.time() - start_time < acquisition_duration:
    buffers = bitalino.get_buffers()
    if not buffers or len(buffers) <= 0 or len(buffers[0]) == 0:
        time.sleep(0.01)
        continue

    # Iterate over all new samples in the buffer (not just the latest)
    for sample in list(buffers[0]):
        ts = sample[0]
        # Only append if this timestamp is new (to avoid duplicates)
        if last_sampled_ts is None or ts > last_sampled_ts:
            sample_counter += 1
            if sample_counter % 50 == 0:
                print(f"Sample #{sample_counter}: ts={ts}, value={sample[1]}")
            value = float(sample[1])
            if sample_counter < 5 or sample_counter % 1000 == 0:
                print(f"DEBUG: Raw sample type: {type(sample[1])}, value: {sample[1]}")
                print(f"DEBUG: Converted value type: {type(value)}, value: {value}")
            raw_deque.append(value)
            timestamps_deque.append(ts)
            last_sampled_ts = ts
    time.sleep(0.001)

# After acquisition, process the data in two ways
print(f"Total samples collected: {len(raw_deque)}")

# DIAGNOSTICS: Print first few and last few raw values for inspection
print("\n--- RAW DATA DIAGNOSTICS ---")
print(f"First 10 values: {list(raw_deque)[:10]}")
print(f"Last 10 values: {list(raw_deque)[-10:]}")
print(f"Data type of first element: {type(next(iter(raw_deque), None))}")

# 1. DIRECT METHOD: Process directly as we've been doing
raw_values = list(raw_deque)
timestamps = list(timestamps_deque)

# Convert to numpy arrays for direct processing
raw_arr = np.array(raw_values, dtype=np.float64)  # Explicitly use float64 for consistent precision
t_arr = np.array(timestamps)
if len(t_arr) > 0:
    t_arr = t_arr - t_arr[0]  # Normalize time to start at 0

print(f"Direct data type: {raw_arr.dtype}, shape: {raw_arr.shape}")
print(f"Direct method - Min: {np.min(raw_arr)}, Max: {np.max(raw_arr)}, Mean: {np.mean(raw_arr)}")

# 2. JSON TEST METHOD: Format data like in the JSON test
# Create a structure similar to what the JSON test would load
json_format_data = {"signal": raw_values}

# Optional: Save to actual JSON file to verify
json_file_path = os.path.join(os.path.dirname(__file__), "ecg_live_capture.json")
with open(json_file_path, 'w') as f:
    json.dump(json_format_data, f)
print(f"Saved raw data to {json_file_path}")

# Load from the JSON-like structure (mimic the JSON test function)
def load_from_json_format(data):
    if isinstance(data, dict):
        for key in ['signal', 'ecg', 'data', 'v', 'values']:
            if key in data:
                return np.array(data[key], dtype=np.float64)
    return None

# Process data like the JSON test would
json_arr = load_from_json_format(json_format_data)
print(f"JSON method data type: {json_arr.dtype}, shape: {json_arr.shape}")
print(f"JSON method - Min: {np.min(json_arr)}, Max: {np.max(json_arr)}, Mean: {np.mean(json_arr)}")

# Apply filter to both datasets
padlen = 3 * max(len(a), len(b))
if len(raw_arr) > padlen:
    # Apply filters
    filtered_direct = filtfilt(b, a, raw_arr)
    filtered_json = filtfilt(b, a, json_arr)
    
    # Normalize both for visualization
    direct_norm = (raw_arr - np.min(raw_arr)) / (np.max(raw_arr) - np.min(raw_arr)) if np.max(raw_arr) != np.min(raw_arr) else raw_arr
    direct_filtered_norm = (filtered_direct - np.min(filtered_direct)) / (np.max(filtered_direct) - np.min(filtered_direct)) if np.max(filtered_direct) != np.min(filtered_direct) else filtered_direct
    
    json_norm = (json_arr - np.min(json_arr)) / (np.max(json_arr) - np.min(json_arr)) if np.max(json_arr) != np.min(json_arr) else json_arr
    json_filtered_norm = (filtered_json - np.min(filtered_json)) / (np.max(filtered_json) - np.min(filtered_json)) if np.max(filtered_json) != np.min(filtered_json) else filtered_json
    
    # Check if filters produced identical results
    filter_diff = np.abs(filtered_direct - filtered_json).mean()
    print(f"Average difference between filtering methods: {filter_diff}")
    
    # Create comparative plots
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Raw signals comparison (scatter, no lines)
    plt.subplot(3, 1, 1)
    plt.scatter(t_arr, direct_norm, color='b', alpha=0.5, label='Direct Raw (normalized)', s=8)
    plt.scatter(t_arr, json_norm, color='g', alpha=0.5, label='JSON Raw (normalized)', s=8)
    plt.title('Raw Signal Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    
    # Plot 2: Filtered signals comparison (scatter, no lines)
    plt.subplot(3, 1, 2)
    plt.scatter(t_arr, direct_filtered_norm, color='r', label='Direct Filtered (normalized)', s=8)
    plt.scatter(t_arr, json_filtered_norm, color='c', label='JSON Filtered (normalized)', s=8)
    plt.title('Filtered Signal Comparison (8-15 Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    
    # Plot 3: Difference between filters (scatter, no lines)
    plt.subplot(3, 1, 3)
    plt.scatter(t_arr, filtered_direct - filtered_json, color='k', s=8)
    plt.title(f'Difference Between Filtering Methods (Avg: {filter_diff:.8f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Difference')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.tight_layout()
    plt.show()
else:
    print(f"Not enough samples for filtfilt: got {len(raw_arr)}, need > {padlen}")

# --- LUNA ACQUISITION BLOCK ---
def acquire_luna(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code):
    bitalino = LunaReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code)
    samples = []
    timestamps = []
    start = time.time()
    while time.time() - start < acquisition_duration:
        val = bitalino.get_last_value()
        samples.append(float(val))
        timestamps.append(time.time() - start)
        # No sleep here: collect as fast as possible
    return np.array(samples, dtype=np.float64), np.array(timestamps, dtype=np.float64)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- LUNA acquisition and plotting ---
    print("\n--- LUNA ACQUISITION ---")
    luna_mac = bitalino_mac_address
    luna_duration = acquisition_duration
    luna_freq = sampling_freq
    luna_channel_code = channel_code
    luna_samples, luna_timestamps = acquire_luna(luna_mac, luna_duration, luna_freq, luna_channel_code)

    print(f"LUNA: Collected {len(luna_samples)} samples")
    print(f"LUNA: Min={np.min(luna_samples)}, Max={np.max(luna_samples)}, Mean={np.mean(luna_samples)}")

    # Filter LUNA data
    padlen = 3 * max(len(a), len(b))
    if len(luna_samples) > padlen:
        luna_filtered = filtfilt(b, a, luna_samples)
        luna_norm = (luna_samples - np.min(luna_samples)) / (np.max(luna_samples) - np.min(luna_samples)) if np.max(luna_samples) != np.min(luna_samples) else luna_samples
        luna_filtered_norm = (luna_filtered - np.min(luna_filtered)) / (np.max(luna_filtered) - np.min(luna_filtered)) if np.max(luna_filtered) != np.min(luna_filtered) else luna_filtered
    else:
        print(f"LUNA: Not enough samples for filtfilt: got {len(luna_samples)}, need > {padlen}")
        luna_filtered = np.zeros_like(luna_samples)
        luna_norm = luna_samples
        luna_filtered_norm = luna_filtered

    # --- Plot LUNA vs PIC (direct deque) ---
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(luna_timestamps, luna_norm, label='LUNA Raw (normalized)', alpha=0.5, s=8)
    plt.scatter(luna_timestamps, luna_filtered_norm, label='LUNA Filtered (normalized)', color='red', s=8)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Signal')
    plt.title('LUNA Acquisition: Raw vs Bandpassed (8-15 Hz)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(t_arr, direct_norm, label='PIC Raw (normalized)', alpha=0.5, s=8)
    plt.scatter(t_arr, direct_filtered_norm, label='PIC Filtered (normalized)', color='red', s=8)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Signal')
    plt.title('PIC Acquisition: Raw vs Bandpassed (8-15 Hz)')
    plt.legend()

    plt.tight_layout()
    plt.show()
