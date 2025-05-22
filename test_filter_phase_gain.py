import numpy as np
import matplotlib.pyplot as plt
from src.utils.signal_processing import NumpySignalProcessor

# Parameters
ds = 1000  # Hz
T = 2      # seconds
N = ds * T

t = np.linspace(0, T, N, endpoint=False)
# Test signal: sum of two sines, one in-band, one out-of-band
f1 = 10  # Hz (in band)
f2 = 100 # Hz (out of band)
sig = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Filter settings
lowcut = 5
highcut = 20
order = 4

# Static (zero-phase) filtering
filtered_static, _ = NumpySignalProcessor.bandpass_filter(sig, lowcut, highcut, ds, order=order, live=False)

# Live (causal) filtering (simulate chunked processing)
chunk_size = 200
zi = None
filtered_live = np.zeros_like(sig)
for i in range(0, len(sig), chunk_size):
    chunk = sig[i:i+chunk_size]
    filtered_chunk, zi = NumpySignalProcessor.bandpass_filter(chunk, lowcut, highcut, ds, order=order, live=True, zi=zi)
    filtered_live[i:i+chunk_size] = filtered_chunk

# Plot
plt.figure(figsize=(12, 8))
plt.plot(t, sig, label='Original Signal', alpha=0.5)
plt.plot(t, filtered_static, label='Static (filtfilt)', linewidth=2)
plt.plot(t, filtered_live, label='Live (lfilter, causal)', linewidth=2, linestyle='--')
plt.title('Filter Phase/Gain Test')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
