import numpy as np

class Arousal:
    @staticmethod
    def hr_arousal_level(hr):
        """
        Return a discrete arousal level for heart rate:
        0 = sleep (<= 50 bpm)
        1 = low (51-70 bpm)
        2 = medium (71-100 bpm)
        3 = high (101-120 bpm)
        4 = panic (> 120 bpm)
        """
        if hr <= 50:
            return 0
        elif hr <= 70:
            return 1
        elif hr <= 100:
            return 2
        elif hr <= 120:
            return 3
        else:
            return 4

    @staticmethod
    def hr_arousal(hr):
        """
        Normalize heart rate (HR) to a value between 0 and 1,
        with each arousal level mapped to a segment of the range.
        """
        # Define boundaries for each level
        levels = [(40, 50), (51, 70), (71, 100), (101, 120), (121, 140)]
        min_hr, max_hr = 40.0, 140.0
        # Find which level hr is in
        for i, (low, high) in enumerate(levels):
            if low <= hr <= high:
                # Normalize within this segment
                segment_min = low
                segment_max = high
                segment_range = segment_max - segment_min
                if segment_range == 0:
                    norm = 0
                else:
                    norm = (hr - segment_min) / segment_range
                # Each segment is 0.2 wide in normalized space
                return float(np.clip(i * 0.2 + norm * 0.2, 0, 1))
        # If out of bounds
        return float(np.clip((hr - min_hr) / (max_hr - min_hr), 0, 1))

    @staticmethod
    def rr_arousal_level(rr):
        """
        Return a discrete arousal level for respiration rate:
        0 = sleep (<= 10 br/min)
        1 = low (11-15 br/min)
        2 = medium (16-22 br/min)
        3 = high (23-27 br/min)
        4 = panic (> 27 br/min)
        """
        if rr <= 10:
            return 0
        elif rr <= 15:
            return 1
        elif rr <= 22:
            return 2
        elif rr <= 27:
            return 3
        else:
            return 4

    @staticmethod
    def rr_arousal(rr):
        """
        Normalize respiration rate (RR) to a value between 0 and 1,
        with each arousal level mapped to a segment of the range.

        RR ranges and their normalized arousal values:
        - 8-10   : sleep    (normalized 0.0 - 0.2)
        - 11-15  : low      (normalized 0.2 - 0.4)
        - 16-22  : medium   (normalized 0.4 - 0.6)
        - 23-27  : high     (normalized 0.6 - 0.8)
        - 28-30  : panic    (normalized 0.8 - 1.0)
        """
        levels = [(8, 10), (11, 15), (16, 22), (23, 27), (28, 30)]
        min_rr, max_rr = 8.0, 30.0
        for i, (low, high) in enumerate(levels):
            if low <= rr <= high:
                segment_min = low
                segment_max = high
                segment_range = segment_max - segment_min
                if segment_range == 0:
                    norm = 0
                else:
                    norm = (rr - segment_min) / segment_range
                return float(np.clip(i * 0.2 + norm * 0.2, 0, 1))
        #print("RR Arousal", float(np.clip((rr - min_rr) / (max_rr - min_rr), 0, 1)))
        return float(np.clip((rr - min_rr) / (max_rr - min_rr), 0, 1))

    @staticmethod
    def scl_arousal_level(scl):
        """
        Return a discrete arousal level for skin conductance level:
        0 = sleep (<= 0.1 μS)
        1 = low (0.1-1 μS)
        2 = medium (1-5 μS)
        3 = high (5-12 μS)
        4 = panic (> 12 μS)
        """
        if scl <= 0.1:
            return 0
        elif scl <= 1:
            return 1
        elif scl <= 5:
            return 2
        elif scl <= 12:
            return 3
        else:
            return 4

    @staticmethod
    def scl_arousal(scl):
        """
        Normalize skin conductance level (SCL) to a value between 0 and 1,
        with each arousal level mapped to a segment of the range.
        """
        levels = [(0.05, 0.1), (0.1, 1), (1, 5), (5, 12), (12, 20)]
        min_scl, max_scl = 0.05, 20.0
        for i, (low, high) in enumerate(levels):
            if low <= scl <= high:
                segment_min = low
                segment_max = high
                segment_range = segment_max - segment_min
                if segment_range == 0:
                    norm = 0
                else:
                    norm = (scl - segment_min) / segment_range
                return float(np.clip(i * 0.2 + norm * 0.2, 0, 1))
        return float(np.clip((scl - min_scl) / (max_scl - min_scl), 0, 1))

    @staticmethod
    def scr_arousal_level(scr):
        """
        Return a discrete arousal level for skin conductance response:
        0 = sleep (<= 0.1 μS)
        1 = low (0.1-0.5 μS)
        2 = medium (0.5-2 μS)
        3 = high (2-4 μS)
        4 = panic (> 4 μS)
        """
        if scr <= 0.1:
            return 0
        elif scr <= 0.5:
            return 1
        elif scr <= 2:
            return 2
        elif scr <= 4:
            return 3
        else:
            return 4

    @staticmethod
    def scr_arousal(scr):
        """
        Normalize skin conductance response (SCR) to a value between 0 and 1,
        with each arousal level mapped to a segment of the range.
        """
        levels = [(0.0, 0.1), (0.1, 0.5), (0.5, 2), (2, 4), (4, 5)]
        min_scr, max_scr = 0.0, 5.0
        for i, (low, high) in enumerate(levels):
            if low <= scr <= high:
                segment_min = low
                segment_max = high
                segment_range = segment_max - segment_min
                if segment_range == 0:
                    norm = 0
                else:
                    norm = (scr - segment_min) / segment_range
                return float(np.clip(i * 0.2 + norm * 0.2, 0, 1))
        return float(np.clip((scr - min_scr) / (max_scr - min_scr), 0, 1))

    def overall_arousal(hr=None, rr=None, scl=None, scr=None):
        """
        Compute overall arousal level based on available HR, RR, SCL, and SCR.
        Uses only the metrics provided (not None). If none are provided, returns 0.5.
        """
        scores = []
        if hr is not None:
            scores.append(Arousal.hr_arousal(hr))
        if rr is not None:
            scores.append(Arousal.rr_arousal(rr))
        if scl is not None:
            scores.append(Arousal.scl_arousal(scl))
        if scr is not None:
            scores.append(Arousal.scr_arousal(scr))
        if not scores:
            return 0.5
        return sum(scores) / len(scores)
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
    
