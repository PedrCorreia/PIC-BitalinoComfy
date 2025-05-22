import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from src.utils.bitalino_receiver_PIC import BitalinoReceiver
from scipy.signal import butter, filtfilt

# BITalino acquisition setup
bitalino_mac_address = "BTH20:16:07:18:17:02"  # Change to your device MAC
acquisition_duration = 20  # seconds
sampling_freq = 1000
channel_code = 0x01  # Only first channel (ECG)
buffer_size = 5000
bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)

# Filtering parameters for QRS accentuation
lowcut = 8
highcut = 18
order = 4
butter_result = butter(order, [lowcut, highcut], btype='bandpass', fs=sampling_freq)
if isinstance(butter_result, tuple) and len(butter_result) >= 2:
    b, a = butter_result[:2]
else:
    raise RuntimeError(f"Unexpected butter() output: {butter_result}")

# Acquisition
start_time = time.time()
raw_deque = deque()
timestamps_deque = deque()
last_sampled_ts = None

print("Starting BITalino ECG acquisition...")

while time.time() - start_time < acquisition_duration:
    buffers = bitalino.get_buffers()
    if not buffers or len(buffers) <= 0 or len(buffers[0]) == 0:
        time.sleep(0.01)
        continue
    channel_data = list(buffers[0])
    for sample in channel_data:
        ts = sample[0]
        if last_sampled_ts is None or ts > last_sampled_ts:
            print(f"Sample: ts={ts}, value={sample[1]}")
            raw_deque.append(sample[1])
            timestamps_deque.append(ts)
            last_sampled_ts = ts
    time.sleep(0.001)

# After acquisition, process the data
print(f"Final raw_deque: {list(raw_deque)}")
print(f"Final timestamps_deque: {list(timestamps_deque)}")
raw_arr = np.array(raw_deque, dtype=np.float64)
t_arr = np.array(timestamps_deque, dtype=np.float64)
if len(t_arr) > 0:
    t_arr = t_arr - t_arr[0]

# Minimal: filter and plot (only after acquisition is complete)
padlen = 3 * max(len(a), len(b))
if len(raw_arr) > padlen:
    filtered = filtfilt(b, a, raw_arr)
else:
    print(f"Not enough samples for filtfilt: got {len(raw_arr)}, need > {padlen}")
    filtered = np.zeros_like(raw_arr)

plt.figure(figsize=(14, 6))
plt.plot(t_arr, raw_arr, label='Raw ECG', alpha=0.5)
plt.plot(t_arr, filtered, label='Bandpassed ECG (8-18 Hz)', color='red', linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('ECG (raw and filtered)')
plt.title('BITalino ECG: Raw vs Bandpassed (8-18 Hz)')
plt.legend()
plt.tight_layout()
plt.show()
