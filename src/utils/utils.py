import numpy as np
class Arousal:
    def weighted_hr(signal, min_val=50, max_val=120):
        arr = np.array(signal)
        if arr.size == 0 or arr.shape[1] < 2:
            return float("nan")
        # Normalize values between min_val and max_val
        vals = arr[:, 1]
        normalized_vals = (vals - vals.min()) / (vals.ptp() + 1e-8) * (max_val - min_val) + min_val
        return float(normalized_vals.mean())

    def weighted_rr(signal, min_val=12, max_val=20):
        arr = np.array(signal)
        if arr.size == 0 or arr.shape[1] < 2:
            return float("nan")
        vals = arr[:, 1]
        # Normalize values between min_val and max_val
        normalized_vals = (vals - vals.min()) / (vals.ptp() + 1e-8) * (max_val - min_val) + min_val
        return float(normalized_vals.mean())

    def weighted_scl(signal, min_val=0.05, max_val=20):
        arr = np.array(signal)
        if arr.size == 0 or arr.shape[1] < 2:
            return float("nan")
        vals = arr[:, 1]
        # Normalize values between min_val and max_val
        normalized_vals = (vals - vals.min()) / (vals.ptp() + 1e-8) * (max_val - min_val) + min_val
        return float(normalized_vals.mean())

class Utils_Arrays:
    def get_last(array):
        if array.size == 0:
            return None
        return array[-1]
    
    def peak(array):
        is_peak = False
        arr = np.array(array)
        if arr[-1, 2] == 1:
            is_peak=True
        
        return is_peak
        

    def separate_components(data_array):
        """Convert [(timestamp, value, is_peak), ...] to (timestamps[], values[], is_peaks[])"""
        if not data_array or len(data_array) == 0:
            return [], [], []
        
        data_array = np.array(data_array)
        timestamps = data_array[:, 0]
        values = data_array[:, 1]
        is_peaks = data_array[:, 2] if data_array.shape[1] > 2 else np.zeros(len(data_array))
    
        return timestamps, values, is_peaks
        
    def combine_components(timestamps, values, is_peaks=None):
        """Convert separate arrays back to [(timestamp, value, is_peak), ...]"""
        if is_peaks is None:
            is_peaks = np.zeros(len(timestamps))
        
        return np.column_stack((timestamps, values, is_peaks)).tolist()
    
