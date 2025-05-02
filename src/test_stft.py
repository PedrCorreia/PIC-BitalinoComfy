import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft  # Import STFT from scipy.signal

def test_stft():
    # Create a pure signal: combination of 10 Hz and 20 Hz sine waves
    sampling_rate = 1000  # Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = 100*np.sin(2 * np.pi * 200 * t) + 100*np.sin(2 * np.pi * 400 * t)  # 10 Hz + 20 Hz

    # Perform the STFT with a moving window of 3 seconds
    f, t_stft, Zxx = stft(signal, fs=sampling_rate, nperseg=3000)

    # Plot the STFT result
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_stft, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.show()

if __name__ == "__main__":
    test_stft()
