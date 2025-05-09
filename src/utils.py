import numpy as np

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

