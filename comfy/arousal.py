import numpy as np
from ..src.utils import Arousal

class PhysioNormalizeNode:
    """
    Node to compute weighted/normalized values for HR, RR, SCR, and arousal.
    Each output can be enabled/disabled via a boolean input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hr_signal": ("DEQUE",),
                "rr_signal": ("DEQUE",),
                "scr_signal": ("DEQUE",),
                "arousal_signal": ("DEQUE",),
                "calc_hr": ("BOOLEAN", {"default": True}),
                "calc_rr": ("BOOLEAN", {"default": True}),
                "calc_scr": ("BOOLEAN", {"default": True}),
                "calc_arousal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("hr_weighted", "rr_weighted", "scr_weighted", "arousal_weighted")
    FUNCTION = "normalize"
    CATEGORY = "Pedro_PIC/🧭 Arousal"

    def normalize(
        self,
        hr_signal,
        rr_signal,
        scr_signal,
        arousal_signal,
        calc_hr=True,
        calc_rr=True,
        calc_scr=True,
        calc_arousal=False,
    ):
        hr_w = Arousal.weighted_hr(hr_signal) if calc_hr else float("nan")
        rr_w = Arousal.weighted_rr(rr_signal) if calc_rr else float("nan")
        scr_w = Arousal.weighted_scr(scr_signal) if calc_scr else float("nan")
        arousal_w = Arousal.compute_arousal(hr_signal, rr_signal, scr_signal) if calc_arousal else float("nan")
        return (hr_w, rr_w, scr_w, arousal_w)

# Node registration
NODE_CLASS_MAPPINGS = {
    "PhysioNormalizeNode": PhysioNormalizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhysioNormalizeNode": "📏 Physio Normalize"
}
