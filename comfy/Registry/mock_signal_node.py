import torch
import numpy as np
import random
import traceback
import time
from ...src.plot.signal_registry import SignalRegistry  # Import the SignalRegistry class

# Create a global signal registry to store signals by ID

class MockSignalGenerator:
    """
    Node that generates mock signals for testing visualization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_type": (["sine", "square", "sawtooth", "random", "ecg", "noise", "circle"], ),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sample_count": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "noise_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "signal_id": ("STRING", {"default": "signal1"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)  # Only returning the signal ID, not the actual tensor
    RETURN_NAMES = ("signal_id",)
    FUNCTION = "generate_signal"
    CATEGORY = "Pedro_PIC/🧰 Tools"
    OUTPUT_NODE = True
    
    def generate_signal(self, signal_type, frequency, amplitude, sample_count, noise_level, signal_id):
        try:
            print("\n\n[DEBUG-CALL] MockSignalGenerator.generate_signal called!")
            print(f"[DEBUG-CALL] Function params: {signal_type}, {frequency}, {amplitude}, {sample_count}, {noise_level}, {signal_id}")
            
            # Create registry instance early
            registry = SignalRegistry.get_instance()
            print(f"[DEBUG-CALL] Registry age: {time.time() - registry.creation_time:.2f} seconds")
            print(f"[DEBUG-CALL] Current registry contents: {list(registry.signals.keys())}")
            
            # Create time values
            t = np.linspace(0, 2*np.pi, sample_count)
            
            # Generate base signal based on type
            if signal_type == "sine":
                signal = amplitude * np.sin(frequency * t)
                print("[DEBUG-GENERATOR] Created sine wave signal")
            elif signal_type == "square":
                signal = amplitude * np.sign(np.sin(frequency * t))
                print("[DEBUG-GENERATOR] Created square wave signal")
            elif signal_type == "sawtooth":
                signal = amplitude * (2 * (t * frequency / (2 * np.pi) - np.floor(0.5 + t * frequency / (2 * np.pi))))
                print("[DEBUG-GENERATOR] Created sawtooth wave signal")
            elif signal_type == "random":
                signal = amplitude * np.random.rand(sample_count)
                print("[DEBUG-GENERATOR] Created random signal")
            elif signal_type == "circle":
                # Create a circle visualization - two signals 90 degrees out of phase
                signal_x = amplitude * np.cos(frequency * t)
                signal_y = amplitude * np.sin(frequency * t)
                # Combine into a single signal for visualization
                signal = np.concatenate((signal_x.reshape(1, -1), signal_y.reshape(1, -1)), axis=0)
                print("[DEBUG-GENERATOR] Created circle signal (x and y components)")
                # Return as 2D tensor directly
                signal_tensor = torch.from_numpy(signal.astype(np.float32))
                print(f"[DEBUG-GENERATOR] Final circle signal shape: {signal_tensor.shape}")
                
                # Store in registry
                registry.register_signal(signal_id, signal_tensor)
                print(f"[DEBUG-GENERATOR] Registered signal '{signal_id}' in registry")
                
                print("[DEBUG-CALL] MockSignalGenerator.generate_signal completed successfully")
                print(f"[DEBUG-CALL] Updated registry contents: {list(registry.signals.keys())}")
                return (signal_id,)
            elif signal_type == "ecg":
                # Simplified ECG-like pattern
                signal = np.zeros(sample_count)
                for i in range(int(sample_count / (10 * frequency))):
                    peak_pos = int(i * 10 * frequency)
                    if peak_pos + 5 < sample_count:
                        signal[peak_pos] = amplitude * 0.2
                        signal[peak_pos + 1] = amplitude * 0.5
                        signal[peak_pos + 2] = amplitude
                        signal[peak_pos + 3] = -amplitude * 0.5
                        signal[peak_pos + 4] = amplitude * 0.3
                        signal[peak_pos + 5] = amplitude * 0.2
                print("[DEBUG-GENERATOR] Created ECG-like signal")
            elif signal_type == "noise":
                signal = amplitude * np.random.normal(0, 1, sample_count)
                print("[DEBUG-GENERATOR] Created noise signal")
            
            # Add noise if specified (only for 1D signals)
            if noise_level > 0 and signal_type != "circle":
                noise = amplitude * noise_level * np.random.normal(0, 1, sample_count)
                signal += noise
                print(f"[DEBUG-GENERATOR] Added noise with level {noise_level}")
            
            # Convert to tensor (for non-circle signals)
            if signal_type != "circle":
                signal_tensor = torch.from_numpy(signal.astype(np.float32)).view(1, -1)
                print(f"[DEBUG-GENERATOR] Created tensor with shape: {signal_tensor.shape}")
                
                # Store in registry
                registry.register_signal(signal_id, signal_tensor)
                print(f"[DEBUG-GENERATOR] Registered signal '{signal_id}' in registry")
            
            # Important workflow note
            print(f"[DEBUG-GENERATOR] Signal '{signal_id}' processing complete.")
            print(f"[DEBUG-GENERATOR] TO VISUALIZE: Connect signal_id -> SignalInputNode -> PlotUnitNode")
            
            print("[DEBUG-CALL] MockSignalGenerator.generate_signal completed successfully")
            print(f"[DEBUG-CALL] Updated registry contents: {list(registry.signals.keys())}")
            return (signal_id,)
            
        except Exception as e:
            print(f"[ERROR-GENERATOR] Exception in generate_signal: {str(e)}")
            print("[ERROR-GENERATOR] Traceback:")
            traceback.print_exc()
            # Return a default signal ID to prevent workflow from breaking
            return ("error_signal",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "MockSignalGenerator": MockSignalGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MockSignalGenerator": "Mock Signal Generator"
}

