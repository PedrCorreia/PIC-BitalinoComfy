"""
Signal Generators for RAW and PROCESSED signals

Each generator registers a signal with a unique ID and metadata to the SignalRegistry.
"""

import numpy as np
import time
try:
    from src.registry.signal_registry import SignalRegistry
except ImportError:
    from registry.signal_registry import SignalRegistry

class SquareRawGenerator:
    """Generator for SQUARE_RAW signal."""
    def __init__(self):
        self.signal_id = "SQUARE_RAW"
        self.metadata = {
            'name': 'Square Raw',
            'color': (255, 150, 100),
            'type': 'raw',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10  # seconds
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        duty = 0.5 + 0.3 * np.sin(0.2 * t)
        data = np.where((t % 1) < duty, 0.8, -0.8)
        data += np.random.normal(0, 0.1, size=self.n_samples)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

class SquareProcessedGenerator:
    """Generator for SQUARE_PROCESSED signal."""
    def __init__(self):
        self.signal_id = "SQUARE_PROCESSED"
        self.metadata = {
            'name': 'Square Processed',
            'color': (100, 150, 255),
            'type': 'processed',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        duty = 0.5 + 0.3 * np.sin(0.2 * t)
        data = np.where((t % 1) < duty, 0.8, -0.8)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

class SawtoothRawGenerator:
    """Generator for SAWTOOTH_RAW signal."""
    def __init__(self):
        self.signal_id = "SAWTOOTH_RAW"
        self.metadata = {
            'name': 'Sawtooth Raw',
            'color': (255, 200, 100),
            'type': 'raw',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        data = 0.7 * (2 * (t * 0.4 % 1) - 1)
        data += np.random.normal(0, 0.07, size=self.n_samples)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

class SawtoothProcessedGenerator:
    """Generator for SAWTOOTH_PROCESSED signal."""
    def __init__(self):
        self.signal_id = "SAWTOOTH_PROCESSED"
        self.metadata = {
            'name': 'Sawtooth Processed',
            'color': (100, 200, 255),
            'type': 'processed',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        data = 0.7 * (2 * (t * 0.4 % 1) - 1)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

# Update Sine generators to output (t, data)
class SineRawGenerator:
    """Generator for SINE_RAW signal."""
    def __init__(self):
        self.signal_id = "SINE_RAW"
        self.metadata = {
            'name': 'Sine Raw',
            'color': (255, 100, 100),
            'type': 'raw',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        data = 0.9 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.08, size=self.n_samples)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

class SineProcessedGenerator:
    """Generator for SINE_PROCESSED signal."""
    def __init__(self):
        self.signal_id = "SINE_PROCESSED"
        self.metadata = {
            'name': 'Sine Processed',
            'color': (100, 100, 255),
            'type': 'processed',
            'sampling_rate': 100,
            'source': 'synthetic',
            'timestamp': time.time()
        }
        self.t0 = time.time()
        self.window = 10
        self.n_samples = 1000

    def generate(self):
        now = time.time()
        t = np.linspace(now - self.window, now, self.n_samples)
        t = np.asarray(t).flatten()
        data = 0.9 * np.sin(2 * np.pi * 0.5 * t + 0.2)
        data = np.asarray(data).flatten()
        return t, data

    def register(self):
        registry = SignalRegistry.get_instance()
        t, data = self.generate()
        registry.register_signal(self.signal_id, (t, data), self.metadata)

# Example usage:
if __name__ == "__main__":
    SineRawGenerator().register()
    SineProcessedGenerator().register()
    SquareRawGenerator().register()
    SquareProcessedGenerator().register()
    SawtoothRawGenerator().register()
    SawtoothProcessedGenerator().register()
    print("Signals registered to SignalRegistry.")
