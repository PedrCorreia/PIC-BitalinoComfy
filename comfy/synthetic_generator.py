from ..src.synthetic_data import SyntheticDataGenerator

# Node implementation for ComfyUI
class SynthNode:
    """ComfyUI node for SyntheticDataGenerator"""
    
    def __init__(self):
        self.generator = SyntheticDataGenerator()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_type": (["EDA", "ECG", "RR"], {"default": "EDA"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "sampling_rate": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "buffer_size": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),  # Buffer size in seconds
                "plot": ("BOOLEAN", {"default": True}),
                "fps": ("INT", {"default": 60, "min": 10, "max": 120}),
                "auto_restart": ("BOOLEAN", {"default": True}),
                "keep_window": ("BOOLEAN", {"default": True}),  # Keep window after completion
            }
        }

    @classmethod 
    def IS_CHANGED(cls, signal_type, duration, sampling_rate, buffer_size, plot, fps, auto_restart, keep_window):
        # Return NaN to trigger node execution on each workflow run
        return float("NaN")

    RETURN_TYPES = ("LIST", "LIST", "BOOLEAN", "TUPLE")
    RETURN_NAMES = ("fx", "y", "plot_trigger", "data")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "Synthetic Data"

    def generate(self, signal_type, duration, sampling_rate, buffer_size, plot, fps, auto_restart, keep_window):
        return self.generator.generate(signal_type, duration, sampling_rate, buffer_size, plot, fps, auto_restart, keep_window)

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SynthNode": SynthNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SynthNode": "Synthetic Data Generator (Real-time)"
}
