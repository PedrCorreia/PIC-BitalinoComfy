import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from src.utils.signal_processing import NumpySignalProcessor
from src.registry.synthetic_functions import ecg_waveform
import json

# --- Parameters ---
fs = 1000
T = 1000  # total duration (seconds)
chunk_size = 50  # samples per chunk (50 ms)
window_secs = 10  # rolling window for plotting (seconds)
window_size = int(window_secs * fs)

# --- Load collarbone signal data ---
with open('src/phy/ECG/collarbone_signal_data.json', 'r') as f:
    collarbone_data = json.load(f)
# Flatten the data array (assuming each entry has a single value in 'data')
signal_array = np.array([entry['data'][0] for entry in collarbone_data])

# --- Rolling buffers ---
t_deque = deque(maxlen=window_size)
sig_deque = deque(maxlen=window_size)
filtered_static_deque = deque(maxlen=window_size)
filtered_live_deque = deque(maxlen=window_size)
env_static_deque = deque(maxlen=window_size)
env_live_deque = deque(maxlen=window_size)
env_ma_static_deque = deque(maxlen=window_size)
env_ma_live_deque = deque(maxlen=window_size)

peaks_static_global = []
peaks_live_global = []

# --- Live processing state ---
zi = None
ma_state = None
thresh_state = None

# --- Matplotlib setup ---
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# --- Real-time signal generation and processing loop ---
for chunk_start in range(0, min(len(signal_array), T * fs), chunk_size):
    # Generate time and ECG chunk
    t_chunk = np.arange(chunk_start, chunk_start + chunk_size) / fs
    sig_chunk = signal_array[chunk_start:chunk_start + chunk_size]

    # Append to rolling deques
    t_deque.extend(t_chunk)
    sig_deque.extend(sig_chunk)

    # --- Static (offline) processing on current window ---
    sig_arr = np.array(sig_deque)
    if len(sig_arr) < chunk_size:
        continue  # wait until enough data
    filtered_static = NumpySignalProcessor.bandpass_filter(sig_arr, 8, 18, fs, order=4)
    env_static = np.abs(filtered_static)
    env_ma_static = NumpySignalProcessor.moving_average(env_static, window_size=25)
    filtered_static_deque.clear(); filtered_static_deque.extend(filtered_static)
    env_static_deque.clear(); env_static_deque.extend(env_static)
    env_ma_static_deque.clear(); env_ma_static_deque.extend(env_ma_static)
    # Static peaks (local to window)
    peaks_static = NumpySignalProcessor.find_peaks(env_ma_static, fs, window=25)
    peaks_static_global = [i for i in peaks_static]

    # --- Live (causal, streaming) processing chunk-by-chunk ---
    filtered_live_chunk, zi = NumpySignalProcessor.bandpass_filter(sig_chunk, 8, 18, fs, order=4, live=True, zi=zi)
    filtered_live_deque.extend(filtered_live_chunk)
    env_live_chunk = np.abs(filtered_live_chunk)
    env_live_deque.extend(env_live_chunk)
    env_ma_live_chunk, ma_state = NumpySignalProcessor.causal_moving_average(env_live_chunk, window_size=25, state=ma_state)
    env_ma_live_deque.extend(env_ma_live_chunk)
    # Live peaks (local to chunk)
    peaks_live, thresh_state = NumpySignalProcessor.live_adaptive_peak_detect(
        np.array(env_ma_live_chunk), window=25, threshold_state=thresh_state, alpha=0.15, min_dist=25)
    # Offset to global index in window
    peaks_live_global = [len(env_ma_live_deque) - len(env_ma_live_chunk) + p for p in peaks_live]

    # --- Dynamic plotting ---
    axes[0].cla(); axes[1].cla(); axes[2].cla()
    t_arr = np.array(t_deque)
    axes[0].plot(t_arr, sig_deque, label='Raw ECG', alpha=0.5)
    axes[0].plot(t_arr, filtered_static_deque, label='Filtered (static)')
    axes[0].plot(t_arr, filtered_live_deque, '--', label='Filtered (live)')
    axes[0].legend(); axes[0].set_title('Filtered Signal (ECG)')

    axes[1].plot(t_arr, env_ma_static_deque, label='Envelope MA (static)')
    axes[1].plot(t_arr, env_ma_live_deque, '--', label='Envelope MA (live)')
    axes[1].legend(); axes[1].set_title('Envelope (Moving Average)')

    axes[2].plot(t_arr, env_ma_live_deque, label='Envelope MA (live)', alpha=0.7)
    if len(peaks_static_global) > 0:
        axes[2].scatter(t_arr[peaks_static_global], np.array(env_ma_static_deque)[peaks_static_global], marker='o', color='C1', label='Peaks (static)')
    if len(peaks_live_global) > 0:
        axes[2].scatter(t_arr[peaks_live_global], np.array(env_ma_live_deque)[peaks_live_global], marker='x', color='C3', label='Peaks (live)')
    axes[2].legend(); axes[2].set_title('Peak Detection')
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.pause(0.01)

plt.ioff()
plt.show()
