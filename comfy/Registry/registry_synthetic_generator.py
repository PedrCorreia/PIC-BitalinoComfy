import sys
import os
import uuid
import numpy as np
import torch
import random
import time
from typing import Dict, List, Tuple, Union
# Import from registry
from ...src.plot.plot_registry import PlotRegistry
from ...src.utils.synthetic_data import SyntheticDataGenerator

class RegistrySyntheticGenerator:
    """
    Registry-compatible synthetic data generator that sends signals to the PlotRegistry.
    This node creates synthetic physiological signals (EDA, ECG, RR) and registers them
    with the centralized PlotRegistry.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Signal selection toggles
                "show_eda": ("BOOLEAN", {"default": True}),
                "show_ecg": ("BOOLEAN", {"default": False}),
                "show_rr": ("BOOLEAN", {"default": False}),
                
                # Core parameters
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "sampling_rate": ("INT", {"default": 100, "min": 1, "max": 1000}),
                
                # Signal characteristics
                "noise_level": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "signal_amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "custom_eda_id": ("STRING", {"default": "EDA"}),
                "custom_ecg_id": ("STRING", {"default": "ECG"}),
                "custom_rr_id": ("STRING", {"default": "RR"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_ids",)
    FUNCTION = "generate_signals"
    CATEGORY = "signal/generators"
    OUTPUT_NODE = False
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"registry_synthetic_{str(uuid.uuid4())[:8]}"
        
        # Get registry singleton
        self.registry = PlotRegistry.get_instance()
        
        # Create synthetic data generator
        self.generator = SyntheticDataGenerator()
        self.generator.plot = False  # Disable direct plotting
        
        print(f"[Registry Synthetic] Node {self.node_id} initialized")
    
    def generate_signals(self, 
                         show_eda, show_ecg, show_rr, 
                         duration, sampling_rate, 
                         noise_level, signal_amplitude,
                         custom_eda_id="EDA", custom_ecg_id="ECG", 
                         custom_rr_id="RR", seed=-1):
        """
        Generate synthetic signals and register them with the PlotRegistry
        
        Args:
            show_eda: Whether to generate EDA signal
            show_ecg: Whether to generate ECG signal
            show_rr: Whether to generate RR signal
            duration: Duration in seconds
            sampling_rate: Sample rate (samples per second)
            noise_level: Amount of noise (0.0-1.0)
            signal_amplitude: Signal amplitude multiplier
            custom_eda_id: Custom ID for the EDA signal
            custom_ecg_id: Custom ID for the ECG signal
            custom_rr_id: Custom ID for the RR signal
            seed: Random seed (-1 for random)
            
        Returns:
            signal_ids: Comma-separated list of generated signal IDs
        """
        print(f"[Registry Synthetic] Generating signals: EDA={show_eda}, ECG={show_ecg}, RR={show_rr}")
          # Use specified random seed if provided
        if seed != -1:
            # Ensure seed is within numpy's valid range (0 to 2^32-1)
            valid_seed = abs(seed) % (2**32)
            random.seed(valid_seed)
            np.random.seed(valid_seed)
        
        # Make sure at least one signal is enabled
        if not (show_eda or show_ecg or show_rr):
            show_eda = True  # Default to EDA if none selected
        
        # Create mapping of signal types to IDs
        signal_ids = {}
        if show_eda:
            signal_ids["EDA"] = custom_eda_id
        if show_ecg:
            signal_ids["ECG"] = custom_ecg_id
        if show_rr:
            signal_ids["RR"] = custom_rr_id
            
        # Configure the generator
        self.generator.buffer_size = duration  # Use duration as buffer size
        
        # Generate the data without plotting
        result = self.generator.generate_multi(
            show_eda=show_eda, 
            show_ecg=show_ecg, 
            show_rr=show_rr,
            duration=duration, 
            sampling_rate=sampling_rate, 
            buffer_size=duration,  # Use same value for buffer
            plot=False,  # No direct plotting
            fps=60,  # Doesn't matter since plot=False
            auto_restart=False, 
            keep_window=False,
            performance_mode=False,
            line_thickness=1,
            enable_downsampling=False
        )
        
        # Extract data depending on the return format
        if len(result) == 5:
            _, _, _, data, _ = result
        elif len(result) == 4:
            _, _, _, data = result
        else:
            print("[Registry Synthetic] Error: Unexpected result format from generator")
            return (",".join(signal_ids.values()),)
            
        # Register each signal with the registry
        active_signals = []
        for signal_type, reg_id in signal_ids.items():
            if signal_type in data and len(data[signal_type]) > 0:
                # Extract x and y values
                x_values = np.array([point[0] for point in data[signal_type]], dtype=np.float32)
                y_values = np.array([point[1] for point in data[signal_type]], dtype=np.float32)
                
                # Apply amplitude scaling and noise
                y_values = y_values * signal_amplitude
                
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * np.std(y_values), len(y_values))
                    y_values = y_values + noise
                
                # Determine color based on signal type
                if signal_type == "EDA":
                    color = (0, 255, 0)  # Green
                elif signal_type == "ECG":
                    color = (255, 0, 0)  # Red
                elif signal_type == "RR":
                    color = (255, 165, 0)  # Orange
                else:
                    color = (0, 0, 255)  # Blue default
                
                # Register with the registry
                self.registry.register_signal(
                    signal_id=reg_id,
                    signal_data=y_values,
                    metadata={
                        'color': color,
                        'source_node': self.node_id,
                        'signal_type': signal_type,
                        'x_values': x_values,
                        'sampling_rate': sampling_rate
                    }
                )
                
                # Connect this node to the signal
                self.registry.connect_node_to_signal(self.node_id, reg_id)
                
                active_signals.append(reg_id)
                print(f"[Registry Synthetic] Registered {signal_type} signal as '{reg_id}' with {len(y_values)} samples")
        
        return (",".join(active_signals),)

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RegistrySyntheticGenerator": RegistrySyntheticGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegistrySyntheticGenerator": "Registry Synthetic Generator"
}
