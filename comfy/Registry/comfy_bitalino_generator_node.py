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
import atexit
from ...src.utils.bitalino_receiver_PIC import BitalinoReceiver
from ...src.registry.signal_registry import SignalRegistry

class BITSignalGeneratorNode:
    _registered_signals = set()
    _generator_threads = {}
    _stop_flags = {}
    _bitalino_instances = {}
    _last_registry_index = {}  # Track last index per (key, channel)
    _signal_buffer_sizes = {}  # Track individual buffer sizes per signal
    _cleanup_registered = False  # Track if cleanup is registered
    
    @classmethod
    def _register_cleanup(cls):
        """Register cleanup on exit if not already registered"""
        if not cls._cleanup_registered:
            atexit.register(cls._cleanup_all)
            cls._cleanup_registered = True
    
    @classmethod  
    def _cleanup_all(cls):
        """Cleanup all instances on program exit"""
        print("[BitalinoGenerator] Program exit cleanup")
        # Create a temporary instance to call cleanup
        temp_instance = cls()
        temp_instance.cleanup()

    def __init__(self):
        # Step 19: Initializing LRBitalinoReceiver
        print("Step 19: Initializing LRBitalinoReceiver")
        # Register cleanup on first instance creation
        self._register_cleanup()


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
                "buffer_size_1": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
                "buffer_size_2": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
                "buffer_size_3": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
                "buffer_size_4": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
                "buffer_size_5": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
                "buffer_size_6": ("INT", {"default": 1000, "min": 10, "max": 1000000}),
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

    def _background_update(self, key, channel_bools, signal_ids, buffer_sizes, update_interval=None):
        """
        Update signal registry with BITalino data in real-time.
        Adapts update rate to sampling frequency to prevent data loss.
        """
        registry = SignalRegistry.get_instance()
        bitalino = self._bitalino_instances[key]
        stop_flag = self._stop_flags[key]
        
        if key not in self._last_registry_index:
            self._last_registry_index[key] = [0] * 6  # up to 6 channels
            
        # Set faster update intervals for better responsiveness
        if update_interval is None:
            # Get sampling_freq from key (it's the second element in the tuple)
            sampling_freq = key[1]
            if sampling_freq >= 1000:
                update_interval = 0.01  # 100Hz updates for 1000Hz sampling
            elif sampling_freq >= 100:
                update_interval = 0.02  # 50Hz updates for 100Hz sampling
            else:
                update_interval = 0.05  # 20Hz updates for slower sampling
            
        # Get sampling frequency from key for metadata
        sampling_freq = key[1]
        
        start_time = None
        last_buffer_sizes = [0] * len(signal_ids)  # Track buffer sizes to detect significant changes
        last_forced_update = [0.0] * len(signal_ids)  # Track last forced update time per signal
        forced_update_interval = 0.1  # Force update every 100ms when buffer is rotating
        
        while not stop_flag['stop']:
            update_start = time.perf_counter()
            
            buffers = bitalino.get_buffers()
            if len(buffers) < len(signal_ids):
                buffers.extend([[] for _ in range(len(signal_ids) - len(buffers))])
                
            # Only update signals when buffer size changes significantly to prevent flickering
            for i, sid in enumerate(signal_ids):
                buf = buffers[i]
                current_size = len(buf) if buf else 0
                
                # Individual buffer size for this signal
                signal_buffer_size = buffer_sizes[i]
                
                # More sensitive updates for faster response
                size_change_threshold = max(2, last_buffer_sizes[i] * 0.005)  # 0.5% change threshold
                significant_change = abs(current_size - last_buffer_sizes[i]) > size_change_threshold
                
                # CRITICAL FIX: When deque is at max capacity, it rotates data but size stays constant
                # We need to check if the buffer content has changed, not just size
                at_max_capacity = current_size >= signal_buffer_size * 0.95  # 95% of max capacity
                buffer_is_rotating = current_size == signal_buffer_size  # Deque at exact max size means rotation
                
                # Time-based forced update for rotating buffers
                time_since_last_forced = update_start - last_forced_update[i]
                needs_forced_update = buffer_is_rotating and time_since_last_forced > forced_update_interval
                
                # Always update when:
                # 1. Significant size change (buffer growing)
                # 2. Buffer is rotating and needs time-based update
                # 3. Buffer has any data and we haven't updated recently
                should_update = (significant_change or needs_forced_update) and current_size > 0
                
                if needs_forced_update:
                    last_forced_update[i] = update_start
                
                if should_update:
                    
                    # NEVER downsample during acquisition - preserve all raw data
                    # Send all data from the buffer
                    t_arr = [ts for ts, val in buf]
                    v_arr = [val for ts, val in buf]
                    
                    # Set start_time at the moment the first sample is received
                    if start_time is None and t_arr:
                        start_time = time.time()
                        
                    if t_arr:  # Only register if we have data
                        meta = {
                            "id": sid, 
                            "sampling_rate": sampling_freq, 
                            "start_time": start_time,
                            "type": "raw",
                            "actual_buffer_size": current_size,
                            "configured_buffer_size": signal_buffer_size,
                            "all_data_preserved": True  # No downsampling in acquisition
                        }
                        registry.register_signal(sid, {"t": t_arr, "v": v_arr}, meta)
                    
                    last_buffer_sizes[i] = current_size
                elif current_size == 0 and last_buffer_sizes[i] != 0:
                    # Buffer was cleared, register empty signal only once
                    registry.register_signal(sid, {"t": [], "v": []})
                    last_buffer_sizes[i] = 0
            
            # Very fast updates for maximum responsiveness
            elapsed = time.perf_counter() - update_start
            sleep_time = max(0.001, update_interval - elapsed)  # Minimum 1ms sleep for maximum speed
            time.sleep(sleep_time)

    def generate_bitalino(self, sampling_freq, duration_sec, bitalino_mac_address,
                        channel_1, channel_2, channel_3, channel_4, channel_5, channel_6,
                        buffer_size_1, buffer_size_2, buffer_size_3, buffer_size_4, buffer_size_5, buffer_size_6,
                        signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6):
        # Convert sampling_freq from string to integer
        sampling_freq = int(sampling_freq)
        # Compute channel_code from booleans
        channel_bools = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6]
        buffer_sizes = [buffer_size_1, buffer_size_2, buffer_size_3, buffer_size_4, buffer_size_5, buffer_size_6]
        channel_code = 0
        for i, active in enumerate(channel_bools):
            if active:
                channel_code |= (1 << i)
        key = (bitalino_mac_address, sampling_freq, channel_code)
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        
        if key not in self._bitalino_instances:
            self._bitalino_instances[key] = BitalinoReceiver(
                bitalino_mac_address, duration_sec, sampling_freq, channel_code, buffer_sizes
            )
            # Wait for device to initialize (max 5s)
            receiver = self._bitalino_instances[key]
            if hasattr(receiver, 'device_initialized'):
                receiver.device_initialized.wait(timeout=5.0)
            self._stop_flags[key] = {'stop': False}
            
            # Store individual buffer sizes for each signal
            self._signal_buffer_sizes[key] = buffer_sizes
            
            thread = threading.Thread(target=self._background_update, args=(key, channel_bools, ids, buffer_sizes), daemon=True)
            self._generator_threads[key] = thread
            thread.start()
            
        # The background thread handles continuous signal registration
        # Just return the signal IDs - initial registration happens in background thread
        registry = SignalRegistry.get_instance()
        ids = [signal_id_1, signal_id_2, signal_id_3, signal_id_4, signal_id_5, signal_id_6]
        
        # Only do initial registration if signals don't exist yet
        for i, sid in enumerate(ids):
            if registry.get_signal(sid) is None:
                # Initial registration with empty values and individual buffer size
                meta = {
                    "id": sid, 
                    "sampling_rate": sampling_freq, 
                    "type": "raw",
                    "configured_buffer_size": buffer_sizes[i]
                }
                registry.register_signal(sid, {"t": [], "v": []}, meta)
                
        return tuple(ids)

    def __del__(self):
        self.cleanup()
    
    def cleanup(self):
        """Properly cleanup all Bitalino instances and stop all threads"""
        print("[BitalinoGenerator] Cleanup called - stopping all instances")
        
        # Stop all background threads first
        for stop_flag in self._stop_flags.values():
            stop_flag['stop'] = True
        
        # Stop all Bitalino receiver instances
        for key, bitalino_instance in self._bitalino_instances.items():
            try:
                print(f"[BitalinoGenerator] Stopping Bitalino instance: {key}")
                if hasattr(bitalino_instance, 'stop'):
                    bitalino_instance.stop()
                if hasattr(bitalino_instance, 'data_compiler') and bitalino_instance.data_compiler:
                    bitalino_instance.data_compiler.stop()
            except Exception as e:
                print(f"[BitalinoGenerator] Error stopping Bitalino instance {key}: {e}")
        
        # Clear all instances
        self._bitalino_instances.clear()
        self._stop_flags.clear()
        print("[BitalinoGenerator] All instances stopped")


