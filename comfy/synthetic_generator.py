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
                # Replace single signal_type dropdown with individual toggles
                "show_eda": ("BOOLEAN", {"default": True}),
                "show_ecg": ("BOOLEAN", {"default": False}),
                "show_rr": ("BOOLEAN", {"default": False}),
                
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "sampling_rate": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "buffer_size": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),  # Buffer size in seconds
                "plot": ("BOOLEAN", {"default": True}),
                "fps": ("INT", {"default": 60, "min": 10, "max": 120}),
                "auto_restart": ("BOOLEAN", {"default": True}),
                "keep_window": ("BOOLEAN", {"default": True}),  # Keep window after completion
                "performance_mode": ("BOOLEAN", {"default": False}),  # Use faster rendering
                "window_width": ("INT", {"default": 640, "min": 320, "max": 1280}),  # Window width
                "window_height": ("INT", {"default": 480, "min": 240, "max": 960}),  # Window height
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 3}),  # Line thickness
                "enable_downsampling": ("BOOLEAN", {"default": False}),  # Control downsampling
            }
        }
        
    @classmethod 
    def IS_CHANGED(cls, show_eda, show_ecg, show_rr, duration, sampling_rate, buffer_size, 
                  plot, fps, auto_restart, keep_window, performance_mode, 
                  window_width, window_height, line_thickness, enable_downsampling):
        # Return NaN to trigger node execution on each workflow run
        return float("NaN")
        
    RETURN_TYPES = ("LIST", "LIST", "BOOLEAN", "TUPLE")
    RETURN_NAMES = ("fx", "y", "plot_trigger", "data")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def generate(self, show_eda, show_ecg, show_rr, duration, sampling_rate, buffer_size, 
                plot, fps, auto_restart, keep_window, performance_mode, 
                window_width, window_height, line_thickness, enable_downsampling):
                
        # Make sure at least one signal is enabled
        if not (show_eda or show_ecg or show_rr):
            show_eda = True  # Default to EDA if none selected
        
        return self.generator.generate_multi(
            show_eda, show_ecg, show_rr,
            duration, sampling_rate, buffer_size, plot, fps, auto_restart, 
            keep_window, performance_mode, window_width, window_height, line_thickness,
            enable_downsampling
        )

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SynthNode": SynthNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SynthNode": "Synthetic Data Generator (Multi-Signal)"
}
