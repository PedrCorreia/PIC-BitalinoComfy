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
                "channels": ("INT", {"default": 1}),  # from 0 to 6
                "buffer_period": ("INT", {"default": 2}),  # size of the buffer
            }
        }
        
    @classmethod 
    def IS_CHANGED(self, bitalino_mac_address, acquisition_duration, sampling_freq, channels, buffer_period):
        return float("NaN")
            
    RETURN_TYPES = ("ARRAY", "ARRAY", "ARRAY", "ARRAY", "ARRAY", "ARRAY", "INT") 
    RETURN_NAMES = ("Buffer_0", "Buffer_1", "Buffer_2", "Buffer_3", "Buffer_4", "Buffer_5", "Sampling_freq") 
    FUNCTION = "get_value"
    OUTPUT_NODE = True
    CATEGORY = "Biosiglas/black_boxes"
    
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
        i = 0
        #while i < channels:
            #print(f"Buffer {i}: {buffers[i]}")
            #i += 1
        buffers_np = [np.array(list(buffer)) for buffer in buffers]
        return (buffers_np[0], buffers_np[1], buffers_np[2], buffers_np[3], buffers_np[4], buffers_np[5], sampling_freq)
