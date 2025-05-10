import threading
import time
import numpy as np
from collections import deque

class SyntheticDataNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.data_deque = deque(maxlen=1000)
        self.running = False
        self.thread = None
        self.signal_type = "EDA"
        self.sampling_rate = 100
        self.duration = 10
        print("SyntheticDataNode initialized")

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
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "Synthetic Data"

    def _background_generator(self):
        t0 = time.time()
        i = 0
        last_printed_second = -1
        while self.running:
            with self.lock:
                t = i / self.sampling_rate
                if self.signal_type == "EDA":
                    baseline = 0.5 * np.sin(2 * np.pi * 0.01 * t)
                    y = baseline + 0.05 * np.random.randn()
                    # Add random peaks
                    if np.random.rand() < 0.01:
                        y += np.random.uniform(0.2, 0.4)
                elif self.signal_type == "ECG":
                    # Simulate a more realistic ECG waveform (PQRST complex)
                    heart_rate = 60  # bpm
                    rr_interval = 60.0 / heart_rate  # seconds per beat
                    t_mod = t % rr_interval

                    # P wave
                    p_wave = 0.1 * np.exp(-((t_mod - 0.1) ** 2) / (2 * 0.01 ** 2))
                    # Q wave
                    q_wave = -0.15 * np.exp(-((t_mod - 0.2) ** 2) / (2 * 0.008 ** 2))
                    # R wave
                    r_wave = 1.0 * np.exp(-((t_mod - 0.22) ** 2) / (2 * 0.005 ** 2))
                    # S wave
                    s_wave = -0.25 * np.exp(-((t_mod - 0.24) ** 2) / (2 * 0.008 ** 2))
                    # T wave
                    t_wave = 0.3 * np.exp(-((t_mod - 0.35) ** 2) / (2 * 0.02 ** 2))

                    y = p_wave + q_wave + r_wave + s_wave + t_wave
                    y += 0.005 * np.random.randn()  # Add noise
                elif self.signal_type == "RR":
                    y = 60 + 5 * np.sin(2 * np.pi * 0.1 * t) + np.random.randn()
                else:
                    y = 0.0
                self.data_deque.append((t, y))
                i += 1

            # # Print once per second, similar to the example
            # current_second = int(t)
            # if current_second != last_printed_second:
            #     realtime_diff = t - current_second
            #     print(f"{current_second} s -> {y:.4f} (realtime diff: {realtime_diff:+.3f}s)")
            #     last_printed_second = current_second

            time.sleep(1.0 / self.sampling_rate)

    def _ensure_thread(self, signal_type, duration, sampling_rate):
        restart = (
            signal_type != self.signal_type or
            sampling_rate != self.sampling_rate or
            duration != self.duration or
            not self.running
        )
        if restart:
            self.signal_type = signal_type
            self.sampling_rate = sampling_rate
            self.duration = duration
            self.data_deque = deque(maxlen=int(sampling_rate * duration))
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._background_generator, daemon=True)
            self.thread.start()

    def generate(self, signal_type, duration, sampling_rate):
        self._ensure_thread(signal_type, duration, sampling_rate)
        # Wait a moment for buffer to fill if just started
        time.sleep(0.05)
        with self.lock:
            fx, y = zip(*self.data_deque) if self.data_deque else ([], [])
        #print(f"fx: {fx}, y: {y},time: {time.time()}")       
        return list(fx), list(y)

# Node registration
NODE_CLASS_MAPPINGS = {
    "SyntheticDataNode": SyntheticDataNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyntheticDataNode": "Synthetic Data Generator"
}
