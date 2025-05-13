# filepath: c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\comfy\synthetic_generator.py
from ...src.utils.synthetic_data import SyntheticDataGenerator

# Node implementation for ComfyUI
class SynthNode:
    '''ComfyUI node for SyntheticDataGenerator'''
    def __init__(self):
        self.generator = SyntheticDataGenerator()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                # Signal selection toggles
                'show_eda': ('BOOLEAN', {'default': True}),
                'show_ecg': ('BOOLEAN', {'default': False}),
                'show_rr': ('BOOLEAN', {'default': False}),
                
                # Core parameters
                'duration': ('INT', {'default': 10, 'min': 1, 'max': 60}),
                'sampling_rate': ('INT', {'default': 100, 'min': 1, 'max': 1000}),
                'buffer_size': ('INT', {'default': 10, 'min': 1, 'max': 60, 'step': 1}),  # Buffer size in seconds
                'plot': ('BOOLEAN', {'default': True}),
                'fps': ('INT', {'default': 60, 'min': 10, 'max': 120}),
                
                # Display quality settings
                'performance_mode': ('BOOLEAN', {'default': False}),  # Use faster rendering
                'line_thickness': ('INT', {'default': 1, 'min': 1, 'max': 3}),  # Line thickness
                'enable_downsampling': ('BOOLEAN', {'default': False}),  # Control downsampling
            }
        }
        
    @classmethod 
    def IS_CHANGED(cls, show_eda, show_ecg, show_rr, duration, sampling_rate, buffer_size, 
                  plot, fps, performance_mode, line_thickness, enable_downsampling):
        # Return NaN to trigger node execution on each workflow run
        return float('NaN')
        
    RETURN_TYPES = ('ARRAY', 'ARRAY', 'TUPLE', 'STRING')
    RETURN_NAMES = ('x', 'y', 'data', 'signal_ids')
    OUTPUT_NODE = True
    FUNCTION = 'generate'
    CATEGORY = 'Pedro_PIC/🧰 Tools'

    def generate(self, show_eda, show_ecg, show_rr, duration, sampling_rate, buffer_size, 
                plot, fps, performance_mode, line_thickness, enable_downsampling):
                
        # Make sure at least one signal is enabled
        if not (show_eda or show_ecg or show_rr):
            show_eda = True  # Default to EDA if none selected
        
        # Always use auto_restart=True and keep_window=True, predefined window size
        auto_restart = True
        keep_window = True
        
        # Get results from the generator
        result = self.generator.generate_multi(
            show_eda, show_ecg, show_rr,
            duration, sampling_rate, buffer_size, plot, fps, auto_restart, 
            keep_window, performance_mode, line_thickness, enable_downsampling
        )
        
        # Handle the result based on what was returned
        if len(result) == 5:
            # New format that already includes signal IDs
            x, y, plot_result, data, signal_ids = result
            return x, y, plot_result, data, ','.join(signal_ids)
        elif len(result) == 4:
            # Older format that doesn't include signal IDs
            x, y, plot_result, data = result
            
            # Extract active signal IDs
            active_signals = []
            if show_eda:
                active_signals.append('EDA')
            if show_ecg:
                active_signals.append('ECG')
            if show_rr:
                active_signals.append('RR')
            
            return x, y, plot_result, data, ','.join(active_signals)
        
        # Fallback case - unlikely to happen
        return [], [], False, {}, 'EDA'

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    'SynthNode': SynthNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SynthNode': 'Synthetic Data Generator (Multi-Signal)'
}
