import platform
import sys
from collections import deque

from ..src.bitalino_receiver import BitalinoReceiver
import numpy as np

class LRBitalinoReceiver:
    def __init__(self):
        # Step 19: Initializing LRBitalinoReceiver
        print("Step 19: Initializing LRBitalinoReceiver")
        self.bitalino = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bitalino_mac_address": ("STRING", {"default": "BTH20:16:07:18:17:02"}),
                "acquisition_duration": ("INT", {"default": 3600*24}),  # duration of bitalino readout (seconds)
                "sampling_freq": ("INT", {"default": 10}),  # how many sample per second
                "channels 1": (BOLEAN, {"default": True}),
                "channels 2": (BOLEAN, {"default": True}),
                "channels 3": (BOLEAN, {"default": True}),
                "channels 4": (BOLEAN, {"default": True}),
                "channels 5": (BOLEAN, {"default": True}),
                "channels 6": (BOLEAN, {"default": True}),
                "buffer_period": ("INT", {"default": 2}),  # size of the buffer
            }
        }
        
    @classmethod 
    def IS_CHANGED(self, bitalino_mac_address, acquisition_duration, sampling_freq, channels, buffer_period):
        return float("NaN")
            
    RETURN_TYPES = (
        "ARRAY", "ARRAY", "ARRAY", "ARRAY", "ARRAY", "ARRAY", "INT",  # Buffers
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT"          # Last values
    )
    RETURN_NAMES = (
        "Buffer_1", "Buffer_2", "Buffer_3", "Buffer_4", "Buffer_5", "Buffer_6", "Sampling_freq",
        "Last_1", "Last_2", "Last_3", "Last_4", "Last_5", "Last_6"
    )
    FUNCTION = "get_value"
    OUTPUT_NODE = True
    CATEGORY = "Pedro_PIC/black_boxes"
    
    def get_value(self, bitalino_mac_address, acquisition_duration, sampling_freq, channels, buffer_period):
        # Step 20: get_value called
        #print("Step 20: get_value called")
        if self.bitalino is None:
            # Step 21: Initializing BitalinoReceiver
            #print("Step 21: Initializing BitalinoReceiver")
            buffer_size = sampling_freq * buffer_period
            self.bitalino = BitalinoReceiver(bitalino_mac_address, acquisition_duration, sampling_freq, channels, buffer_size)
            #print(f"bitalino initialized: {self.bitalino}")
        
        # Step 22: Retrieving buffers
        #print("Step 22: Retrieving buffers")
        buffers = self.bitalino.get_buffers()  # Retrieve all buffers
        buffers = buffers[:channels + 1]  # Limit buffers to the selected number of channels
        
        # Ensure all six buffers are returned, even if some are empty
        while len(buffers) < 6:
            buffers.append(deque())
        
        # Step 23: Returning buffers
        #print("Step 23: Returning buffers")
        buffers_np = [np.array(list(buffer)) for buffer in buffers]
        # Get last value from each buffer, handling tuples/arrays
        last_values = []
        for buffer in buffers_np:
            if len(buffer) > 0:
                last = buffer[-1]
                # If last is an array or tuple, extract the first element
                if isinstance(last, (np.ndarray, list, tuple)):
                    # If it's a 1-element array, extract the scalar
                    if hasattr(last, 'shape') and last.shape == ():  # 0-dim array
                        last_values.append(float(last))
                    elif len(last) > 0:
                        last_values.append(float(last[0]))
                    else:
                        last_values.append(float("nan"))
                else:
                    last_values.append(float(last))
            else:
                last_values.append(float("nan"))
        return (
            buffers_np[0], buffers_np[1], buffers_np[2], buffers_np[3], buffers_np[4], buffers_np[5], sampling_freq,
            last_values[0], last_values[1], last_values[2], last_values[3], last_values[4], last_values[5]
        )