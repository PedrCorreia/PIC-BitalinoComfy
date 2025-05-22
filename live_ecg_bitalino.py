import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time
import sys
from scipy.signal import bessel, filtfilt, lfilter
sys.path.append(r'c:/Users/corre/ComfyUI/custom_nodes/PIC-2025')
from src.utils.signal_processing import NumpySignalProcessor
from src.utils.bitalino_receiver_PIC import BitalinoReceiver
from collections import deque

# BITalino acquisition setup
bitalino_mac_address = "BTH20:16:07:18:17:02"  # Change to your device MAC
acquisition_duration = 9999  # Long duration for live
sampling_freq = 1000
channel_code = 0x01  # Only first channel (ECG)
buffer_size = 5000  # Large enough for rolling window
bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)

def normalize(x):
    x = np.asarray(x)
    if np.ptp(x) == 0:
        return x
    return (x - np.mean(x)) / np.ptp(x)

# Real-time plotting setup
fs = sampling_freq
window_sec = 5  # Show 5 seconds
window_size = fs * window_sec
buffer = np.zeros(window_size)
causal_buffer = np.zeros(window_size)

acquisition_time_sec = 1000  # How long to acquire/process (seconds)
start_time = time.time()

lowcut = 5
highcut = 20
order_causal = 2
order_filtfilt = 2
lowcut_wide = 2
highcut_wide = 40
order_wide = 2

# Initialize buffers
raw_buffer = np.zeros(window_size)
causal_buffer = np.zeros(window_size)
filtfilt_buffer = np.zeros(window_size)
wide_buffer = np.zeros(window_size)

downsample_factor = 4  # e.g., 1000 Hz -> 250 Hz

plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# Initialize lines with correct x and y data for 5s window
init_x = np.arange(window_size // downsample_factor) / (fs // downsample_factor)
init_y = np.zeros(window_size // downsample_factor)
lines = [
    axs[0].plot(init_x, init_y)[0],
    axs[1].plot(init_x, init_y)[0],
]
axs[0].set_ylabel('Raw')
axs[1].set_ylabel('Live Filtered')
axs[1].set_xlabel('Time (s)')
fig.suptitle('Live ECG Processing Results')
plt.show(block=False)
plt.pause(0.1)

class LiveFilter:
    """Base class for live filters."""
    def process(self, x):
        if np.isnan(x):
            return x
        return self._process(x)
    def __call__(self, x):
        return self.process(x)
    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")

class LiveLFilter(LiveFilter):
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)
    def _process(self, x):
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)
        return y

# Design a live bandpass filter (Butterworth, like your previous settings)
from scipy.signal import iirfilter
b, a = iirfilter(N=order_causal, Wn=[lowcut, highcut], fs=fs, btype="bandpass", ftype="butter", output="ba")
live_bandpass = LiveLFilter(b, a)

# Rolling buffers for true live window
raw_rolling = np.zeros(window_size)
live_rolling = np.zeros(window_size)

try:
    channel_index = 0
    min_samples = 1  # process every new sample
    while time.time() - start_time < acquisition_time_sec:
        buffers = bitalino.get_buffers()
        if not buffers or len(buffers) <= channel_index or len(buffers[channel_index]) < 1:
            print(f"[WARN] No data or not enough data in channel {channel_index}. Buffer length: {len(buffers[channel_index]) if buffers and len(buffers) > channel_index else 0}")
            time.sleep(0.01)
            continue
        channel_data = list(buffers[channel_index])
        # Only process the latest sample
        latest_sample = channel_data[-1][1]
        # Update rolling raw buffer
        raw_rolling = np.roll(raw_rolling, -1)
        raw_rolling[-1] = latest_sample
        # Live filter: process one sample at a time, keep state
        filtered_sample = live_bandpass(latest_sample)
        live_rolling = np.roll(live_rolling, -1)
        live_rolling[-1] = filtered_sample
        # Downsample for plotting
        raw_ds = raw_rolling[::downsample_factor]
        live_ds = live_rolling[::downsample_factor]
        t_ds = np.arange(len(raw_ds)) / (fs // downsample_factor)
        # Update plots with both x and y data
        lines[0].set_xdata(t_ds)
        lines[0].set_ydata(normalize(raw_ds))
        lines[1].set_xdata(t_ds)
        lines[1].set_ydata(normalize(live_ds))
        for ax in axs:
            ax.set_xlim(t_ds[0], t_ds[-1])
            ax.set_ylim(-1.2, 1.2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
except KeyboardInterrupt:
    print('Stopped.')
finally:
    plt.ioff()
    plt.show()
