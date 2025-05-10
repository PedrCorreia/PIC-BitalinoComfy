import threading
import time
import numpy as np
from collections import deque

class SyntheticDataNode:
    def __init__(self):
        self.data_deque = deque(maxlen=1000)
        self.running = False
        self.thread = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_type": (["EDA", "ECG", "RR"], {"default": "EDA"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "sampling_rate": ("INT", {"default": 100, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("fx", "y")
    FUNCTION = "generate"

    CATEGORY = "Synthetic Data"

    def _generate_signal(self, signal_type, duration, sampling_rate):
        t = np.linspace(0, duration, int(duration * sampling_rate))
        if signal_type == "EDA":
            # Large baseline drift (slow sine), with mid-amplitude Gaussian peaks
            baseline = 0.5 * np.sin(2 * np.pi * 0.01 * t)  # slow drift
            y = baseline + 0.05 * np.random.randn(len(t))
            # Add mid-amplitude Gaussian peaks at random locations
            num_peaks = int(duration // 2)
            for _ in range(num_peaks):
                peak_center = np.random.uniform(0, duration)
                peak_width = np.random.uniform(0.1, 0.5)
                peak_height = np.random.uniform(0.2, 0.4)
                y += peak_height * np.exp(-0.5 * ((t - peak_center) / peak_width) ** 2)
        elif signal_type == "ECG":
            # Simulate a more realistic ECG: sum of harmonics + sharp R-peaks
            y = 0.1 * np.sin(2 * np.pi * 1.0 * t)  # baseline wander
            y += 0.5 * np.sin(2 * np.pi * 1.2 * t)  # P-wave
            y += 0.2 * np.sin(2 * np.pi * 1.7 * t)  # T-wave
            # Add sharp R-peaks at regular intervals (simulate heartbeats)
            heart_rate = 60  # bpm
            rr_interval = 60.0 / heart_rate  # seconds between beats
            for beat in np.arange(0, duration, rr_interval):
                r_peak_width = 0.03  # seconds
                r_peak_height = 1.0
                y += r_peak_height * np.exp(-0.5 * ((t - beat) / r_peak_width) ** 2)
            y += 0.05 * np.random.randn(len(t))  # noise
        elif signal_type == "RR":
            y = 60 + 5 * np.sin(2 * np.pi * 0.1 * t) + np.random.randn(len(t))
        else:
            y = np.zeros_like(t)
        for xi, yi in zip(t, y):
            self.data_deque.append((xi, yi))
            time.sleep(1.0 / sampling_rate)

    def generate(self, signal_type, duration, sampling_rate):
        self.data_deque.clear()
        self.running = True
        self.thread = threading.Thread(target=self._generate_signal, args=(signal_type, duration, sampling_rate))
        self.thread.start()
        # Wait for data generation to finish
        self.thread.join()
        fx, y = zip(*self.data_deque) if self.data_deque else ([], [])
        return list(fx), list(y)

# Node registration
NODE_CLASS_MAPPINGS = {
    "SyntheticDataNode": SyntheticDataNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyntheticDataNode": "Synthetic Data Generator"
}
