from scipy.signal import butter
b, a = butter(4, [0.1, 0.2], btype='bandpass')
print("b:", b)
print("a:", a)