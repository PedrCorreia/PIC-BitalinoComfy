import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Signal parameters
fs = 1000  # Hz
T = 10     # seconds
N = fs * T

t = np.linspace(0, T, N, endpoint=False)
# Create three sinusoids
sig_2hz = np.sin(2 * np.pi * 2 * t)
sig_10hz = np.sin(2 * np.pi * 10 * t)
sig_30hz = np.sin(2 * np.pi * 30 * t)
sum_sig = sig_2hz + sig_10hz + sig_30hz

# Bandpass filter for QRS (8-20 Hz)
lowcut = 8
highcut = 20
order = 4
b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
filtered_sum = filtfilt(b, a, sum_sig)

plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, sig_2hz, label='2 Hz')
plt.ylabel('Amplitude')
plt.title('2 Hz Component')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(t, sig_10hz, label='10 Hz')
plt.ylabel('Amplitude')
plt.title('10 Hz Component')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(t, sig_30hz, label='30 Hz')
plt.ylabel('Amplitude')
plt.title('30 Hz Component')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(t, sum_sig, label='Sum (2+10+30 Hz)', alpha=0.5)
plt.plot(t, filtered_sum, label='Filtered Sum (8-20 Hz)', color='red', linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum and Bandpass Filtered Output')
plt.legend()
plt.tight_layout()
plt.show()
