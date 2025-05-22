#!/usr/bin/env python
"""
Modular Signal Generator System for PIC-2025
- Supports synthetic, processed, and hardware (Bitalino) signal generators.
- Each generator registers signals and metadata in the SignalRegistry.
- Easily extensible for new generator types.
"""
import numpy as np
import threading
import time
import collections
from src.registry.plot_registry import PlotRegistry
from src.registry.signal_registry import SignalRegistry
from src.utils.bitalino_receiver_PIC import BitalinoReceiver

class BaseSignalGenerator:
    """
    Abstract base class for all signal generators.
    """
    def __init__(self, signal_registry=None, plot_registry=None):
        self.signal_registry = signal_registry or SignalRegistry.get_instance()
        self.plot_registry = plot_registry or PlotRegistry.get_instance()

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_signal_ids(self):
        raise NotImplementedError

class SyntheticSignalGenerator(BaseSignalGenerator):
    """
    Generates synthetic signals (ECG, EDA, Sine, etc.).
    """
    def __init__(self, sampling_rate=10, buffer_seconds=10, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.buffer_seconds = buffer_seconds
        self.max_buffer_size = int(self.buffer_seconds * self.sampling_rate)
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()
        self.signal_defs = {}
        self.signal_buffers = {}
        self.last_generated_values = {}
        self.start_time = time.time()

    def add_signal(self, signal_id, generator_func, metadata):
        """
        Register a new synthetic signal with a generator function and metadata.
        """
        self.signal_defs[signal_id] = {
            'generator_func': generator_func,
            'metadata': metadata
        }
        self.signal_buffers[signal_id] = {
            't': collections.deque(maxlen=self.max_buffer_size),
            'v': collections.deque(maxlen=self.max_buffer_size)
        }
        self.signal_registry.register_signal(signal_id, {'t': [], 'v': []}, metadata)
        self.plot_registry.register_signal(signal_id, {'t': [], 'v': []}, metadata)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self):
        sample_index = 0
        next_sample_time = time.time()
        while self.running:
            try:
                with self.data_lock:
                    current_time = time.time() - self.start_time
                    for signal_id, sigdef in self.signal_defs.items():
                        gen_func = sigdef['generator_func']
                        value = gen_func(current_time, sample_index)
                        self.last_generated_values[signal_id] = value
                        buf = self.signal_buffers[signal_id]
                        buf['t'].append(current_time)
                        buf['v'].append(value)
                        plot_data = {
                            't': np.array(buf['t']),
                            'v': np.array(buf['v'])
                        }
                        self.signal_registry.register_signal(signal_id, plot_data, sigdef['metadata'])
                        self.plot_registry.register_signal(signal_id, plot_data, sigdef['metadata'])
                sample_index += 1
                next_sample_time += 1.0 / self.sampling_rate
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_sample_time = time.time()
            except Exception as e:
                print(f"Error in synthetic signal generator: {e}")
                import traceback; traceback.print_exc()
                time.sleep(1.0)

    def get_signal_ids(self):
        return list(self.signal_defs.keys())

class RegistrySignalGenerator:
    """
    Manager for all signal generators (synthetic, processed, hardware).
    Allows adding/removing/starting/stopping any generator type.
    """
    def __init__(self):
        self.generators = []

    def add_generator(self, generator):
        self.generators.append(generator)

    def start_all(self):
        for gen in self.generators:
            gen.start()

    def stop_all(self):
        for gen in self.generators:
            gen.stop()

    def get_all_signal_ids(self):
        ids = []
        for gen in self.generators:
            ids.extend(gen.get_signal_ids())
        return ids

class ProcessedSignalGenerator(BaseSignalGenerator):
    """
    Processes an existing signal by applying a function, and registers the result as a new processed signal.
    """
    def __init__(self, input_signal_id, process_func, output_signal_id, output_metadata, sampling_rate=10, **kwargs):
        super().__init__(**kwargs)
        self.input_signal_id = input_signal_id
        self.process_func = process_func
        self.output_signal_id = output_signal_id
        self.output_metadata = output_metadata
        self.sampling_rate = sampling_rate
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self):
        next_sample_time = time.time()
        while self.running:
            try:
                input_data = self.signal_registry.get_signal(self.input_signal_id)
                if input_data and len(input_data['v']) > 0:
                    processed_v = self.process_func(np.array(input_data['v']))
                    processed_t = np.array(input_data['t'])
                    plot_data = {'t': processed_t, 'v': processed_v}
                    self.signal_registry.register_signal(self.output_signal_id, plot_data, self.output_metadata)
                    self.plot_registry.register_signal(self.output_signal_id, plot_data, self.output_metadata)
                next_sample_time += 1.0 / self.sampling_rate
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_sample_time = time.time()
            except Exception as e:
                print(f"Error in processed signal generator: {e}")
                import traceback; traceback.print_exc()
                time.sleep(1.0)

    def get_signal_ids(self):
        return [self.output_signal_id]

class BitalinoSignalGenerator(BaseSignalGenerator):
    """
    Wraps a BitalinoReceiver to register hardware signals in the registry system.
    """
    def __init__(self, bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size, signal_ids, metadata_map, **kwargs):
        super().__init__(**kwargs)
        self.receiver = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channel_code, buffer_size)
        self.signal_ids = signal_ids  # List of signal ids to register (e.g., ["BITALINO_CH0", ...])
        self.metadata_map = metadata_map  # Dict mapping signal id to metadata
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.receiver.stop()

    def _run(self):
        while self.running:
            try:
                buffers = self.receiver.get_buffers()
                for i, signal_id in enumerate(self.signal_ids):
                    if i < len(buffers):
                        buf = buffers[i]
                        t = np.array([x[0] for x in buf])
                        v = np.array([x[1] for x in buf])
                        plot_data = {'t': t, 'v': v}
                        metadata = self.metadata_map.get(signal_id, {})
                        self.signal_registry.register_signal(signal_id, plot_data, metadata)
                        self.plot_registry.register_signal(signal_id, plot_data, metadata)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in Bitalino signal generator: {e}")
                import traceback; traceback.print_exc()
                time.sleep(1.0)

    def get_signal_ids(self):
        return self.signal_ids
