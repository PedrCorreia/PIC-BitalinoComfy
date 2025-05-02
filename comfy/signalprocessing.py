from ..src.signalprocessing import SignalProcessing  # Use SignalProcessing class


class MovingAverageFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input buffer as a deque
                "Window_size": ("INT", {"default": 3}),  # Window size for moving average
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the filtered data as a deque
    RETURN_NAMES = ("Filtered_Data",)
    FUNCTION = "apply_filter"
    CATEGORY = "PIC/Filters"

    def apply_filter(self, Buffer, Window_size):
        signal_processor = SignalProcessing(signal=Buffer)
        return signal_processor.moving_average(Window_size)

class SignalThresholdFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input buffer as a deque
                "Threshold": ("FLOAT", {"default": 0.0}),  # Threshold value
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the filtered data as a deque
    RETURN_NAMES = ("Filtered_Data",)
    FUNCTION = "apply_filter"
    CATEGORY = "PIC/Filters"

    def apply_filter(self, Buffer, Threshold):
        signal_processor = SignalProcessing(signal=Buffer)
        return signal_processor.filter_threshold(Threshold)
class SignalPLL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input signal buffer as a deque
                "Reference_Signal": ("DEQUE",),  # Reference signal buffer as a deque
                "Loop_Gain": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "Initial_Phase": ("FLOAT", {"default": 0.0, "min": -np.pi, "max": np.pi, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DEQUE",)  # Return the PLL-processed signal as a deque
    RETURN_NAMES = ("PLL_Output",)
    FUNCTION = "apply_pll"
    CATEGORY = "PIC/Filters"

    def apply_pll(self, Buffer, Reference_Signal, Loop_Gain, Initial_Phase):
        signal_processor = SignalProcessing(signal=Buffer)
        return signal_processor.pll(reference_signal=Reference_Signal, loop_gain=Loop_Gain, initial_phase=Initial_Phase)

class SignalFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Buffer": ("DEQUE",),  # Input buffer as a deque
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
    CATEGORY = "PIC/Filters"

    def apply_filter(self, Buffer, Filter_Type, Cutoff_Freq, Band_Cutoff_Freqs, Sampling_Freq):
        signal_processor = SignalProcessing(signal=Buffer)
        if Filter_Type == "low":
            return signal_processor.low_pass_filter(Cutoff_Freq, Sampling_Freq)
        elif Filter_Type == "high":
            return signal_processor.high_pass_filter(Cutoff_Freq, Sampling_Freq)
        elif Filter_Type == "band":
            return signal_processor.band_pass_filter(Band_Cutoff_Freqs[0], Band_Cutoff_Freqs[1], Sampling_Freq)
        else:
            raise ValueError(f"Invalid Filter_Type: {Filter_Type}")