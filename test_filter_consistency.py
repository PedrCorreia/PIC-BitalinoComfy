import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Generate a test signal with known components
fs = 1000  # sampling frequency in Hz
t = np.arange(0, 5, 1/fs)  # 5 seconds of data
f1, f2, f3, f4 = 1, 10, 20, 50  # frequency components (Hz)

# Create test signal with multiple frequency components
signal = (
    np.sin(2*np.pi*f1*t) +     # 1 Hz component (should be filtered out)
    np.sin(2*np.pi*f2*t) * 2 +  # 10 Hz component (should pass through)
    np.sin(2*np.pi*f3*t) * 0.5 + # 20 Hz component (should be attenuated)
    np.sin(2*np.pi*f4*t) * 0.3   # 50 Hz component (should be filtered out)
)

# Add some noise
np.random.seed(42)  # For reproducibility
signal_noisy = signal + 0.2 * np.random.randn(len(t))

# Create the filter
lowcut = 8
highcut = 15
order = 4
b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

# Apply filter using float32 data
signal_float32 = signal_noisy.astype(np.float32)
filtered_float32 = filtfilt(b, a, signal_float32)

# Apply filter using float64 data
signal_float64 = signal_noisy.astype(np.float64)
filtered_float64 = filtfilt(b, a, signal_float64)

# Apply filter using Python float list
signal_list = [float(x) for x in signal_noisy]
filtered_list = filtfilt(b, a, signal_list)

# Check differences
diff_32_64 = np.abs(filtered_float32 - filtered_float64).mean()
diff_64_list = np.abs(filtered_float64 - filtered_list).mean()

print(f"Data type differences:")
print(f"float32: {signal_float32.dtype}")
print(f"float64: {signal_float64.dtype}")
print(f"list converted to: {np.array(filtered_list).dtype}")
print(f"Average difference between float32 and float64 filtered: {diff_32_64:.10f}")
print(f"Average difference between float64 and list filtered: {diff_64_list:.10f}")

# Plot the results
plt.figure(figsize=(15, 10))

# Original signal
plt.subplot(3, 1, 1)
plt.plot(t[:500], signal_noisy[:500])
plt.title('Noisy Signal (First 500 samples)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Filtered signals
plt.subplot(3, 1, 2)
plt.plot(t[:500], filtered_float32[:500], 'r-', label='float32', alpha=0.7)
plt.plot(t[:500], filtered_float64[:500], 'g-', label='float64', alpha=0.7)
plt.plot(t[:500], filtered_list[:500], 'b--', label='Python list', alpha=0.7)
plt.title('Filtered Signals (First 500 samples)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Differences
plt.subplot(3, 1, 3)
plt.plot(t[:500], filtered_float32[:500] - filtered_float64[:500], 'r-', label='float32 - float64', alpha=0.7)
plt.plot(t[:500], filtered_float64[:500] - filtered_list[:500], 'g-', label='float64 - list', alpha=0.7)
plt.title('Differences Between Filtering Methods')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude Difference')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()
