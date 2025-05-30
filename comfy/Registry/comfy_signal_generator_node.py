"""
ComfyUI Node: Comfy Signal Generator
- Lets user select signal type, sampling frequency, and frequency
- Registers a single signal in the generator registry
- Output is a single signal id (for use with Signal Connector)
"""
import numpy as np
from ...src.registry.signal_registry import SignalRegistry
from ...src.registry import synthetic_functions
import threading
import time

class ComfySignalGeneratorNode:
    _registered_signals = set()  # Class-level set to track registered signals
    _generator_threads = {}      # Track background threads by signal_id
    _stop_flags = {}            # Track stop flags for each signal_id

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_type": ("STRING", {"default": "sine_waveform", "options": [
                    "sine_waveform", "ecg_waveform", "eda_waveform"
                ]}),
                "sampling_freq": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "freq": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0}),
                "signal_id": ("STRING", {"default": "GEN_0"}),
                "duration_sec": ("INT", {"default": 10, "min": 1, "max": 3600}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Registry/SignalGen"
    OUTPUT_NODE = True

    def _background_generate(self, signal_type, sampling_freq, freq, signal_id, duration_sec, stop_flag):
        import collections
        registry = SignalRegistry.get_instance()
        t_window = 10  # seconds
        maxlen = int(t_window * sampling_freq)
        t_deque = collections.deque(maxlen=maxlen)
        v_deque = collections.deque(maxlen=maxlen)
        start_time = None
        while not stop_flag['stop']:
            now = time.time()
            if start_time is None:
                start_time = now
            t_new = now - start_time
            if t_new > duration_sec:
                break
            t_deque.append(t_new)
            sample_index = len(t_deque) - 1
            if signal_type == "sine_waveform":
                v_new = synthetic_functions.sine_waveform(np.array([t_new]), np.array([sample_index]), frequency=freq)[0]
            elif signal_type == "ecg_waveform":
                v_new = synthetic_functions.ecg_waveform(np.array([t_new]), np.array([sample_index]))[0]
            elif signal_type == "eda_waveform":
                v_new = synthetic_functions.eda_waveform(np.array([t_new]), np.array([sample_index]))[0]
            elif signal_type == "rr_waveform":
                v_new = synthetic_functions.rr_waveform(np.array([t_new]), np.array([sample_index]))[0]
            else:
                raise ValueError(f"Unknown signal_type: {signal_type}")
            v_deque.append(v_new)
            t_arr = np.array(t_deque)
            v_arr = np.array(v_deque)
            meta = {"id": signal_id, "sampling_rate": sampling_freq, "freq": freq, "start_time": start_time}
            registry.register_signal(signal_id, {"t": t_arr, "v": v_arr}, meta)
            time.sleep(1/sampling_freq)
        # After finished, keep last data in registry so UI doesn't crash
        registry.register_signal(signal_id, {"t": t_arr, "v": v_arr}, meta)

    def generate(self, signal_type, sampling_freq, freq, signal_id, duration_sec):
        if signal_id in self._generator_threads:
            # Already running in background, just return the ID
            return (signal_id,)
        stop_flag = {'stop': False}
        self._stop_flags[signal_id] = stop_flag
        thread = threading.Thread(target=self._background_generate, args=(signal_type, sampling_freq, freq, signal_id, duration_sec, stop_flag), daemon=True)
        self._generator_threads[signal_id] = thread
        thread.start()
        self._registered_signals.add(signal_id)
        return (signal_id,)

    def __del__(self):
        # Attempt to stop all background threads gracefully
        for stop_flag in self._stop_flags.values():
            stop_flag['stop'] = True

NODE_CLASS_MAPPINGS = {"ComfySignalGeneratorNode": ComfySignalGeneratorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfySignalGeneratorNode": "Comfy Signal Generator"}
