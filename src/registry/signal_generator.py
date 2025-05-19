#!/usr/bin/env python
"""
Signal Generator for PIC-2025 Registry System
- Contains only signal generation and registration logic.
- No UI or adapter logic.
"""
import numpy as np
import threading
import time
import collections
from src.registry.plot_registry import PlotRegistry

class RegistrySignalGenerator:
    """
    Generator for synthetic signals to be registered with the SignalRegistry.
    Handles only signal creation, updating, and registration.
    """
    def __init__(self):
        self.plot_registry = PlotRegistry.get_instance()
        self.sampling_rate = 10  # Hz (set to 10Hz for debug)
        self.signals = {}
        self.generators = {
            'raw': {
                'ECG': {'created': False, 'id': None},
                'EDA': {'created': False, 'id': None},
                'RAW_SINE': {'created': False, 'id': None}
            },
            'processed': {
                'WAVE1': {'created': False, 'id': None},
                'WAVE2': {'created': False, 'id': None},
                'WAVE3': {'created': False, 'id': None}
            }
        }
        self.buffer_seconds = 10
        self.max_buffer_size = int(self.buffer_seconds * self.sampling_rate)
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()
        self.last_generated_values = {}
        self.start_time = time.time()  # Store generator initialization time
        print(f"RegistrySignalGenerator initialized with {self.buffer_seconds}s buffer")

    def set_buffer_seconds(self, seconds):
        self.buffer_seconds = max(1, seconds)
        self.max_buffer_size = int(self.buffer_seconds * self.sampling_rate)
        print(f"Buffer size set to {self.buffer_seconds} seconds ({self.max_buffer_size} samples)")
        return self.buffer_seconds

    def start(self):
        if self.running:
            print("Signal generator already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self._generate_signals)
        self.thread.daemon = True
        self.thread.start()
        print("Signal generator started")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("Signal generator stopped")

    def create_ecg_signal(self):
        if self.generators['raw']['ECG']['created']:
            return self.generators['raw']['ECG']['id']
        signal_id = "ECG"
        initial_data = {'t': [], 'v': [], 'meta': {}}
        metadata = {
            'name': 'ECG Signal',
            'color': (255, 0, 0),
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_ecg,
            'type': 'raw'
        }
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        self.signals[signal_id] = metadata
        self.generators['raw']['ECG']['created'] = True
        self.generators['raw']['ECG']['id'] = signal_id
        return signal_id

    def create_eda_signal(self):
        if self.generators['raw']['EDA']['created']:
            return self.generators['raw']['EDA']['id']
        signal_id = "EDA"
        initial_data = {'t': [], 'v': [], 'meta': {}}
        metadata = {
            'name': 'EDA Signal',
            'color': (220, 120, 0),
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_eda,
            'type': 'raw'
        }
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        self.signals[signal_id] = metadata
        self.generators['raw']['EDA']['created'] = True
        self.generators['raw']['EDA']['id'] = signal_id
        return signal_id

    def create_sine_signal(self, processed=False, name=None):
        if not name:
            name = "WAVE1" if processed else "RAW_SINE"
        category = 'processed' if processed else 'raw'
        if name not in self.generators[category]:
            return None
        if self.generators[category][name]['created']:
            return self.generators[category][name]['id']
        prefix = "PROC" if processed else "RAW"
        signal_id = f"{prefix}_{name}"
        initial_data = {'t': [], 'v': [], 'meta': {}}
        freq_map = {"WAVE1": 0.5, "WAVE2": 0.25, "WAVE3": 0.1, "RAW_SINE": 1.0}
        frequency = freq_map.get(name, 0.5)
        def custom_generator(elapsed_time, sample_index):
            t = elapsed_time
            value = np.sin(2 * np.pi * frequency * t)
            noise_level = 0.05 if processed else 0.1
            value = value * 0.9 + np.random.normal(0, noise_level)
            return value
        generator_func = custom_generator
        metadata = {
            'name': f'{"Processed" if processed else "Raw"} {name}',
            'color': (0, 180, 220) if processed else (220, 180, 0),
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': generator_func,
            'type': 'processed' if processed else 'raw'
        }
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        self.signals[signal_id] = metadata
        self.generators[category][name]['created'] = True
        self.generators[category][name]['id'] = signal_id
        return signal_id

    def _generate_signals(self):
        # Create initial signals (only once)
        self.create_ecg_signal()
        self.create_eda_signal()
        self.create_sine_signal(processed=True, name="WAVE1")
        sample_index = 0
        next_sample_time = time.time()
        # Use deques for each signal's t and v
        signal_buffers = {}
        while self.running:
            try:
                with self.data_lock:
                    current_time = time.time() - self.start_time  # Use time relative to initialization
                    for signal_id, metadata in self.signals.items():
                        generate_func = metadata.get('generator_func')
                        if generate_func:
                            new_value = generate_func(current_time, sample_index)
                            self.last_generated_values[signal_id] = new_value
                            # Initialize deques if not present
                            if signal_id not in signal_buffers:
                                signal_buffers[signal_id] = {
                                    't': collections.deque(maxlen=self.max_buffer_size),
                                    'v': collections.deque(maxlen=self.max_buffer_size)
                                }
                            buf = signal_buffers[signal_id]
                            buf['t'].append(current_time)
                            buf['v'].append(new_value)
                            plot_data = {
                                't': np.array(buf['t']),
                                'v': np.array(buf['v']),
                                'meta': metadata
                            }
                            self.plot_registry.register_signal(
                                signal_id=signal_id,
                                signal_data=plot_data,
                                metadata=metadata
                            )
                sample_index += 1
                # Real-time sampling: wait until next sample time
                next_sample_time += 1.0 / self.sampling_rate
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_sample_time = time.time()
            except Exception as e:
                print(f"Error in signal generator thread: {e}")
                import traceback; traceback.print_exc()
                time.sleep(1.0)

    def _generate_ecg(self, elapsed_time, sample_index):
        t = elapsed_time
        base_freq = 1.2
        value = np.sin(2 * np.pi * base_freq * t)
        spike = np.exp(-80 * ((t * base_freq) % 1 - 0.2) ** 2)
        value = value * 0.3 + spike * 0.7
        value = value + np.random.normal(0, 0.05)
        return value

    def _generate_eda(self, elapsed_time, sample_index):
        t = elapsed_time
        slow_component = 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
        random_walk = np.sin(0.5 * t) * np.random.normal(0, 0.02)
        value = slow_component + random_walk
        return value

    def _generate_sine(self, elapsed_time, sample_index):
        t = elapsed_time
        frequency = 0.5
        value = np.sin(2 * np.pi * frequency * t)
        value = value * 0.8 + np.random.normal(0, 0.1)
        return value

    def add_custom_signal(self, name, processed=False):
        category = 'processed' if processed else 'raw'
        if name in self.generators[category]:
            return self.create_sine_signal(processed=processed, name=name)
        for gen_name, info in self.generators[category].items():
            if not info['created']:
                return self.create_sine_signal(processed=processed, name=gen_name)
        return None

    def get_active_signal_ids_by_type(self, signal_type):
        return {
            signal_id for signal_id, metadata in self.signals.items()
            if metadata.get('type') == signal_type
        }
