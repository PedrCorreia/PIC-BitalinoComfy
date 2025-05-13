from ..src.utils.signal_processing import NumpySignalProcessor  # Use NumpySignalProcessor class
import numpy as np
from collections import deque
import json  # For loading JSON files

class MovingAverageFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input buffer as a deque of [timestamp, value] pairs
                "Window_size": ("INT", {"default": 3}),  # Window size for moving average
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the filtered data as a deque
    RETURN_NAMES = ("Filtered_Data",)
    FUNCTION = "apply_filter"
    CATEGORY = "Pedro_PIC/🔬 Processing"

    def apply_filter(self, Buffer, Window_size):
        # Extract values from the deque
        data = np.array(Buffer)
        timestamps, values = data[:, 0], data[:, 1]

        # Apply moving average to the values
        signal_processor = NumpySignalProcessor(signal=values)
        filtered_values = signal_processor.moving_average(Window_size)

        # Combine timestamps with filtered values
        filtered_data = np.column_stack((timestamps, filtered_values))
        return deque(filtered_data.tolist())


class SignalFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input buffer as a deque of [timestamp, value] pairs
                "Filter_Type": ("STRING", {  # Select filter type
                    "default": "low",
                    "choices": ["low", "high", "band"]
                }),
                "Cutoff_Freq": ("FLOAT", {"default": 0.1}),  # Cutoff frequency for low/high-pass
                "Band_Cutoff_Freqs": ("LIST", {  # Cutoff frequencies for band-pass
                    "default": [0.1, 0.5],
                    "min_length": 2,
                    "max_length": 2
                }),
                "Sampling_Freq": ("INT",),  # Sampling frequency
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the filtered data as a deque
    RETURN_NAMES = ("Filtered_Data",)
    FUNCTION = "apply_filter"
    CATEGORY = "Pedro_PIC/🔬 Processing"

    def apply_filter(self, Buffer, Filter_Type, Cutoff_Freq, Band_Cutoff_Freqs, Sampling_Freq):
        # Extract values from the deque
        data = np.array(Buffer)
        timestamps, values = data[:, 0], data[:, 1]

        # Apply the selected filter to the values
        signal_processor = NumpySignalProcessor(signal=values)
        if Filter_Type == "low":
            filtered_values = NumpySignalProcessor.lowpass_filter(values, Cutoff_Freq, Sampling_Freq)
        elif Filter_Type == "high":
            filtered_values = NumpySignalProcessor.highpass_filter(Cutoff_Freq, Sampling_Freq)
        elif Filter_Type == "band":
            filtered_values = NumpySignalProcessor.bandpass_filter(Band_Cutoff_Freqs[0], Band_Cutoff_Freqs[1], Sampling_Freq)
        else:
            raise ValueError(f"Invalid Filter_Type: {Filter_Type}")

        # Combine timestamps with filtered values
        filtered_data = np.column_stack((timestamps, filtered_values))
        return deque(filtered_data.tolist())


class LoadSignalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "File_Path": ("STRING",),  # Path to the JSON file
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the loaded signal as a deque
    RETURN_NAMES = ("Loaded_Signal",)
    FUNCTION = "load_signal"
    CATEGORY = "Pedro_PIC/🔬 Processing"

    def load_signal(self, File_Path):
        # Load the JSON file
        with open(File_Path, 'r') as file:
            data = json.load(file)

        # Ensure the data is in the format [[timestamp, value], ...]
        if not isinstance(data, list) or not all(isinstance(item, list) and len(item) == 2 for item in data):
            raise ValueError("Invalid JSON format. Expected a list of [timestamp, value] pairs.")

        # Convert to deque
        return deque(data)

NODE_CLASS_MAPPINGS = {
    "MovingAverageFilter": MovingAverageFilter,
    "SignalFilter": SignalFilter,
    "LoadSignalNode": LoadSignalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MovingAverageFilter": "📉 Moving Average Filter",
    "SignalFilter": "🔍 Signal Filter",
    "LoadSignalNode": "📂 Load Signal"
}