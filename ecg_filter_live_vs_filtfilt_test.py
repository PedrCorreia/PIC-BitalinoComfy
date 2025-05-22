import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from src.utils.bitalino_receiver_PIC import BitalinoReceiver
from src.utils.signal_processing import NumpySignalProcessor
from scipy.signal import iirfilter, lfilter, filtfilt, sosfilt, butter

# BITalino acquisition setup
bitalino_mac_address = "BTH20:16:07:18:17:02"  # Change to your device MAC
acquisition_duration = 20  # seconds
sampling_freq = 1000
channel_code = 0x01  # Only first channel (ECG)
buffer_size = 5000
bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)

# Filtering parameters
lowcut = 5
highcut = 20
order = 4

# Design bandpass filter
b, a = iirfilter(N=order, Wn=[lowcut, highcut], fs=sampling_freq, btype="bandpass", ftype="butter", output="ba")
sos = iirfilter(N=order, Wn=[lowcut, highcut], fs=sampling_freq, btype="bandpass", ftype="butter", output="sos")

# Acquisition and processing
window_sec = 5
window_size = sampling_freq * window_sec
raw_buffer = np.zeros(window_size)
live_lfilter_buffer = np.zeros(window_size)
filtfilt_buffer = np.zeros(window_size)
past_lfilter_buffer = np.zeros(window_size)

# For live lfilter state
class LiveLFilter:
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)
    def __call__(self, x):
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)
        return y

# Live SOS filter (sample-by-sample, numerically stable)
class LiveSosFilter:
    """Live implementation of digital filter with second-order sections."""
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos
        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def __call__(self, x):
        """Filter incoming data with cascaded second-order sections."""
        for s in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]
            y = b0 * x + self.state[s, 0]
            self.state[s, 0] = b1 * x - a1 * y + self.state[s, 1]
            self.state[s, 1] = b2 * x - a2 * y
            x = y
        return y

# Use a Butterworth bandpass for live lfilter (minimal phase distortion, good for ECG)
live_b, live_a = b, a
live_lfilter = LiveLFilter(live_b, live_a)
live_sosfilter = LiveSosFilter(sos)

# Use a moving window for live windowed filtering (with minimal latency)
def moving_window_lfilter(signal_window, b, a):
    # Use scipy lfilter for causal, minimal-latency filtering on the window
    return lfilter(b, a, signal_window)

def moving_window_filtfilt(signal_window, b, a):
    # Use NumpySignalProcessor.bandpass_filter for zero-phase (no mode argument, always filtfilt)
    return NumpySignalProcessor.bandpass_filter(signal_window, lowcut, highcut, sampling_freq, order=order)

acquisition_time_sec = 20
start_time = time.time()
raw_all = []
live_lfilter_all = []
filtfilt_all = []
past_lfilter_all = []
timestamps_all = []

min_samples = 1000

# Use deques for rolling window and live acquisition/processing
raw_deque = deque(maxlen=window_size * 10)  # store more for post-hoc plot
live_lfilter_deque = deque(maxlen=window_size * 10)
live_sosfilter_deque = deque(maxlen=window_size * 10)
live_window_lfilter_deque = deque(maxlen=window_size * 10)
live_window_filtfilt_deque = deque(maxlen=window_size * 10)
timestamps_deque = deque(maxlen=window_size * 10)

print("Starting BITalino ECG acquisition and filtering test...")

while time.time() - start_time < acquisition_time_sec:
    buffers = bitalino.get_buffers()
    if not buffers or len(buffers) <= 0 or len(buffers[0]) < min_samples:
        time.sleep(0.05)
        continue
    channel_data = list(buffers[0])
    latest = channel_data[-1]
    latest_val = latest[1]
    latest_ts = latest[0]
    # Update deques
    raw_deque.append(latest_val)
    filtered_val = live_lfilter(latest_val)
    live_lfilter_deque.append(filtered_val)
    sos_val = live_sosfilter(latest_val)
    live_sosfilter_deque.append(sos_val)
    # Live windowed lfilter and filtfilt (moving window, minimal latency)
    if len(raw_deque) >= window_size:
        window_arr = np.array(list(raw_deque)[-window_size:])
        # Causal lfilter on window
        window_lfilter = moving_window_lfilter(window_arr, b, a)
        live_window_lfilter_deque.append(window_lfilter[-1])
        # Zero-phase filtfilt on window (using numpy version)
        window_filtfilt = moving_window_filtfilt(window_arr, b, a)
        live_window_filtfilt_deque.append(window_filtfilt[-1])
    else:
        live_window_lfilter_deque.append(0)
        live_window_filtfilt_deque.append(0)
    timestamps_deque.append(latest_ts)
    time.sleep(0.01)

# After acquisition, process the full arrays for windowed lfilter and filtfilt
raw_arr = np.array(raw_deque)
live_lfilter_arr = np.array(live_lfilter_deque)
live_sosfilter_arr = np.array(live_sosfilter_deque)
t_arr = np.array(timestamps_deque)
if len(t_arr) > 0:
    t_arr = t_arr - t_arr[0]

# --- Static bandpass filtering of the entire raw signal after acquisition ---
static_bandpass_full = lfilter(b, a, raw_arr)
static_bandpass_filtfilt_full = filtfilt(b, a, raw_arr)

# Sliding window processing for the entire signal
windowed_lfilter_full = np.zeros_like(raw_arr)
windowed_filtfilt_full = np.zeros_like(raw_arr)
for i in range(window_size, len(raw_arr)):
    window = raw_arr[i-window_size:i]
    windowed_lfilter_full[i] = lfilter(b, a, window)[-1]
    windowed_filtfilt_full[i] = moving_window_filtfilt(window, b, a)[-1]

# Optionally, accentuate peaks by squaring the live filter output
accentuated_live = live_lfilter_arr ** 2
accentuated_sos = live_sosfilter_arr ** 2

# Trim all arrays to the minimum length
minlen = min(len(raw_arr), len(live_lfilter_arr), len(windowed_lfilter_full), len(windowed_filtfilt_full), len(t_arr), len(accentuated_live), len(live_sosfilter_arr), len(accentuated_sos), len(static_bandpass_full), len(static_bandpass_filtfilt_full))
raw_arr = raw_arr[-minlen:]
live_lfilter_arr = live_lfilter_arr[-minlen:]
live_sosfilter_arr = live_sosfilter_arr[-minlen:]
windowed_lfilter_full = windowed_lfilter_full[-minlen:]
windowed_filtfilt_full = windowed_filtfilt_full[-minlen:]
t_arr = t_arr[-minlen:]
accentuated_live = accentuated_live[-minlen:]
accentuated_sos = accentuated_sos[-minlen:]
static_bandpass_full = static_bandpass_full[-minlen:]
static_bandpass_filtfilt_full = static_bandpass_filtfilt_full[-minlen:]

plt.figure(figsize=(14, 21))
plt.subplot(9, 1, 1)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(raw_arr), label='Raw')
plt.ylabel('Raw')
plt.subplot(9, 1, 2)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(live_lfilter_arr), label='Live Butter lfilter', color='orange')
plt.ylabel('Live Butter lfilter')
plt.subplot(9, 1, 3)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(accentuated_live), label='Live Butter lfilter (accentuated)', color='magenta')
plt.ylabel('Live Butter lfilter (accentuated)')
plt.subplot(9, 1, 4)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(live_sosfilter_arr), label='Live SOS filter', color='red')
plt.ylabel('Live SOS filter')
plt.subplot(9, 1, 5)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(accentuated_sos), label='Live SOS filter (accentuated)', color='brown')
plt.ylabel('Live SOS (accentuated)')
plt.subplot(9, 1, 6)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(windowed_lfilter_full), label='Live window lfilter', color='blue')
plt.ylabel('Live window lfilter')
plt.subplot(9, 1, 7)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(windowed_filtfilt_full), label='Live window filtfilt (numpy)', color='green')
plt.ylabel('Live window filtfilt')
plt.subplot(9, 1, 8)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(static_bandpass_full), label='Static lfilter (full signal)', color='purple')
plt.ylabel('Static lfilter')
plt.subplot(9, 1, 9)
plt.plot(t_arr, NumpySignalProcessor.normalize_signal(static_bandpass_filtfilt_full), label='Static filtfilt (full signal)', color='teal')
plt.ylabel('Static filtfilt')
plt.xlabel('Time (s)')
plt.suptitle('ECG Filtering: Raw, Live/Windowed/Static Filters (BITalino)')
plt.tight_layout()
plt.show()
