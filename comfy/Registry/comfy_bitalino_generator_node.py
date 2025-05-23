"""
ComfyUI Node: Comfy Signal Generator
- Lets user select signal type, sampling frequency, and frequency
- Registers a single signal in the generator registry
- Output is a single signal id (for use with Signal Connector)
"""
import numpy as np
from ...src.registry.signal_registry import SignalRegistry
import threading
import time
from ...src.utils.bitalino_receiver_PIC import BitalinoReceiver
from ...src.registry.signal_registry import SignalRegistry

class BITSignalGeneratorNode:
    _registered_signals = set()
    _generator_threads = {}
    _stop_flags = {}
    _bitalino_instances = {}
    _last_registry_index = {}  # Track last index per (key, channel)

    def __init__(self):
        # Step 19: Initializing LRBitalinoReceiver
        print("Step 19: Initializing LRBitalinoReceiver")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampling_freq": (["10", "100", "1000"], {"default": "100"}),
                "duration_sec": ("INT", {"default": 10, "min": 1, "max": 36000000}),
                "bitalino_mac_address": ("STRING", {"default": "BTH20:16:07:18:17:02"}),
                "channel_1": ("BOOLEAN", {"default": True}),
                "channel_2": ("BOOLEAN", {"default": True}),
                "channel_3": ("BOOLEAN", {"default": True}),
                "channel_4": ("BOOLEAN", {"default": False}),
                "channel_5": ("BOOLEAN", {"default": False}),
                "channel_6": ("BOOLEAN", {"default": False}),
                "buffer_size": ("INT", {"default": 1000, "min": 10, "max": 10000}),
                "signal_id_1": ("STRING", {"default": "BIT_1"}),
                "signal_id_2": ("STRING", {"default": "BIT_2"}),
                "signal_id_3": ("STRING", {"default": "BIT_3"}),
                "signal_id_4": ("STRING", {"default": "BIT_4"}),
                "signal_id_5": ("STRING", {"default": "BIT_5"}),
                "signal_id_6": ("STRING", {"default": "BIT_6"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("signal_id_1", "signal_id_2", "signal_id_3", "signal_id_4", "signal_id_5", "signal_id_6")
    FUNCTION = "generate_bitalino"
    CATEGORY = "Registry/SignalGen"
    OUTPUT_NODE = True

    def _background_update(self, key, channel_bools, signal_ids, update_interval=None):
        """
        Update signal registry with BITalino data in real-time.
        Adapts update rate to sampling frequency to prevent data loss.
        """
        registry = SignalRegistry.get_instance()
        bitalino = self._bitalino_instances[key]
        stop_flag = self._stop_flags[key]
        
        if key not in self._last_registry_index:
            self._last_registry_index[key] = [0] * 6  # up to 6 channels
            
        # Set appropriate update interval based on sampling frequency
        if update_interval is None:
            # Get sampling_freq from key (it's the second element in the tuple)
            sampling_freq = key[1]
            if sampling_freq >= 1000:
                update_interval = 0.01  # 100Hz updates for 1000Hz sampling
            elif sampling_freq >= 100:
                update_interval = 0.02  # 50Hz updates for 100Hz sampling
            else:
                update_interval = 0.04  # 25Hz updates for slower sampling
        
        # Pre-allocate these lists for efficiency
        t_rel_lists = [[] for _ in range(len(signal_ids))]
        v_lists = [[] for _ in range(len(signal_ids))]
        
        while not stop_flag['stop']:
            update_start = time.perf_counter()
            
            buffers = bitalino.get_buffers()
            if len(buffers) < len(signal_ids):
                buffers.extend([[] for _ in range(len(signal_ids) - len(buffers))])
                
            # Batch process all data for all signals
            for i, sid in enumerate(signal_ids):
                buf = buffers[i]
                if buf:
                    # Clear previous lists
                    t_rel_lists[i].clear()
                    v_lists[i].clear()
                    
                    # Extract all time-value pairs
                    for ts, val in buf:
                        t_rel_lists[i].append(ts)
                        v_lists[i].append(val)
                    
                    # Register the data in a single operation
                    if t_rel_lists[i]:
                        registry.register_signal(sid, {"t": t_rel_lists[i], "v": v_lists[i]})
                else:
                    registry.register_signal(sid, {"t": [], "v": []})
            
            # Adaptive sleep - only sleep for remaining time in the interval
            elapsed = time.perf_counter() - update_start
            sleep_time = max(0.001, update_interval - elapsed)  # Minimum 1ms sleep
            time.sleep(sleep_time)

    def generate_bitalino(self, sampling_freq, duration_sec, bitalino_mac_address,
                        channel_1, channel_2, channel_3, channel_4, channel_5, channel_6,
                        buffer_size,
                        signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6):
        # Convert sampling_freq from string to integer
        sampling_freq = int(sampling_freq)
        # Compute buffer size in samples based on duration_sec
        # Compute channel_code from booleans
        channel_bools = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6]
        channel_code = 0
        for i, active in enumerate(channel_bools):
            if active:
                channel_code |= (1 << i)
        key = (bitalino_mac_address, sampling_freq, channel_code)
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        if key not in self._bitalino_instances:
            self._bitalino_instances[key] = BitalinoReceiver(
                bitalino_mac_address, duration_sec, sampling_freq, channel_code, buffer_size
            )
            # Wait for device to initialize (max 5s)
            receiver = self._bitalino_instances[key]
            if hasattr(receiver, 'device_initialized'):
                receiver.device_initialized.wait(timeout=5.0)
            self._stop_flags[key] = {'stop': False}
            thread = threading.Thread(target=self._background_update, args=(key, channel_bools, ids), daemon=True)
            self._generator_threads[key] = thread
            thread.start()
        bitalino = self._bitalino_instances[key]
        buffers = bitalino.get_buffers()
        while len(buffers) < 6:
            buffers.append([])
        registry = SignalRegistry.get_instance()
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        for i, sid in enumerate(ids):
            buf = buffers[i]
            if buf:
                t, v = zip(*buf)
                t0 = t[0]
                t_rel = [float(ts) - float(t0) for ts in t]
                registry.register_signal(sid, {"t": t_rel, "v": list(v)})
            else:
                registry.register_signal(sid, {"t": [], "v": []})
        return tuple(ids)

    def __del__(self):
        for stop_flag in self._stop_flags.values():
            stop_flag['stop'] = True


