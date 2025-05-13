import numpy as np
from  ..src.utils.utils import Utils_Arrays

class CombineNode:
    """
    Node to combine timestamp, value, and is_peak arrays into a single array of tuples.
    Uses the combine_components utility function.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timestamps": ("ARRAY",),
                "values": ("ARRAY",),
                "is_peaks": ("ARRAY", {"default": None, "optional": True}),
            }
        }

    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("Data",)
    FUNCTION = "combine"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def combine(self, timestamps, values, is_peaks=None):
        """Combine separate arrays into [(timestamp, value, is_peak), ...] format"""
        return (Utils_Arrays.combine_components(timestamps, values, is_peaks),)


class SeparateNode:
    """
    Node to separate an array of (timestamp, value, is_peak) tuples into separate arrays.
    Uses the separate_components utility function.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ARRAY",),  # Array of (timestamp, value, is_peak) tuples
            }
        }

    RETURN_TYPES = ("ARRAY", "ARRAY", "ARRAY")
    RETURN_NAMES = ("timestamps", "values", "is_peaks")
    FUNCTION = "separate"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def separate(self, data):
        """Separate [(timestamp, value, is_peak), ...] format into component arrays"""
        timestamps, values, is_peaks = Utils_Arrays.separate_components(data)
        return timestamps, values, is_peaks
    
# ...existing code...

class GetLastValueNode:
    """
    Node that extracts the last element from a signal array.
    Uses the get_last utility function.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ARRAY",),  # Input array of (timestamp, value, is_peak) tuples
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("last_timestamp", "last_value", "is_peak")
    FUNCTION = "get_last"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def get_last(self, data):
        """Extract the last element from the signal array"""

        last_element = Utils_Arrays.get_last(np.array(data))
        if last_element is None:
            return float("nan"), float("nan"), False
        
        # Return the individual components of the last element
        return float(last_element[0]), float(last_element[1]), bool(last_element[2])


class IsPeakNode:
    """
    Node that checks if the last element in the signal array is a peak.
    Features buffer mode to extend peak detection over time.
    """
    
    def __init__(self):
        self.last_peak_time = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ARRAY",),  # Input array of (timestamp, value, is_peak) tuples
                "use_buffer": ("BOOLEAN", {"default": False}),
                "buffer_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_peak",)
    FUNCTION = "check_peak"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def check_peak(self, data, use_buffer=False, buffer_duration=0.5):
        """
        Check if the last element in the signal array is a peak
        When use_buffer is True, output remains True for buffer_duration seconds after peak
        """
        from ..src.utils.utils import Utils_Arrays
        import numpy as np
        import time
        
        # Get the raw peak detection from the utility
        is_peak = Utils_Arrays.peak(np.array(data))
        
        # If not using buffer, just return the peak status
        if not use_buffer:
            return (is_peak,)
            
        # If we detected a new peak, update the timestamp
        if is_peak:
            self.last_peak_time = time.time()
            
        # Check if we're still within the buffer window of the last peak
        time_since_peak = time.time() - self.last_peak_time
        buffered_peak = time_since_peak <= buffer_duration
        
        return (buffered_peak,)