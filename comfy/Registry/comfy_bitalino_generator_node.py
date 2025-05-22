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
    FUNCTION = "generate_bitalino"
    CATEGORY = "Registry/SignalGen"
    OUTPUT_NODE = True

    def _background_update(self, key, channel_bools, signal_ids, update_interval=0.05):
        registry = SignalRegistry.get_instance()
        bitalino = self._bitalino_instances[key]
        stop_flag = self._stop_flags[key]
        while not stop_flag['stop']:
            buffers = bitalino.get_buffers()
            while len(buffers) < 6:
                buffers.append([])
            for i, sid in enumerate(signal_ids):
                buf = buffers[i]
                if buf:
                    t, v = zip(*buf)
                    registry.register_signal(sid, {"t": list(t), "v": list(v)})
                else:
                    registry.register_signal(sid, {"t": [], "v": []})
            time.sleep(update_interval)

    def generate_bitalino(self, sampling_freq, duration_sec, bitalino_mac_address,
                        channel_1, channel_2, channel_3, channel_4, channel_5, channel_6,
                        buffer_size,
                        signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6):
        # Convert sampling_freq from string to integer
        sampling_freq = int(sampling_freq)
        print(f"[BITSignalGeneratorNode] Called with: sampling_freq={sampling_freq}, duration_sec={duration_sec}, mac={bitalino_mac_address}, buffer_size={buffer_size}")
        print(f"[BITSignalGeneratorNode] Channels: {[channel_1, channel_2, channel_3, channel_4, channel_5, channel_6]}")
        print(f"[BITSignalGeneratorNode] Signal IDs: {[signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]}")
        # Compute channel_code from booleans
        channel_bools = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6]
        channel_code = 0
        for i, active in enumerate(channel_bools):
            if active:
                channel_code |= (1 << i)
        print(f"[BITSignalGeneratorNode] Computed channel_code: {hex(channel_code)}")
        key = (bitalino_mac_address, sampling_freq, channel_code)
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        if key not in self._bitalino_instances:
            print(f"[BITSignalGeneratorNode] Initializing new BitalinoReceiver for key: {key}")
            self._bitalino_instances[key] = BitalinoReceiver(
                bitalino_mac_address, duration_sec, sampling_freq, channel_code, buffer_size
            )
            # Wait for device to initialize (max 5s)
            receiver = self._bitalino_instances[key]
            if hasattr(receiver, 'device_initialized'):
                print("[BITSignalGeneratorNode] Waiting for BITalino device to initialize...")
                if not receiver.device_initialized.wait(timeout=5.0):
                    print("[BITSignalGeneratorNode] Warning: BITalino device did not initialize in time.")
            self._stop_flags[key] = {'stop': False}
            thread = threading.Thread(target=self._background_update, args=(key, channel_bools, ids), daemon=True)
            self._generator_threads[key] = thread
            thread.start()
        bitalino = self._bitalino_instances[key]
        buffers = bitalino.get_buffers()
        print(f"[BITSignalGeneratorNode] Buffers lengths: {[len(b) for b in buffers]}")
        while len(buffers) < 6:
            buffers.append([])
        registry = SignalRegistry.get_instance()
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        for i, sid in enumerate(ids):
            buf = buffers[i]
            if buf:
                t, v = zip(*buf)
                print(f"[BITSignalGeneratorNode] Registering {sid}: {len(t)} samples")
                registry.register_signal(sid, {"t": list(t), "v": list(v)})
            else:
                print(f"[BITSignalGeneratorNode] Registering {sid}: EMPTY")
                registry.register_signal(sid, {"t": [], "v": []})
        return tuple(ids)

    def __del__(self):
        for stop_flag in self._stop_flags.values():
            stop_flag['stop'] = True


