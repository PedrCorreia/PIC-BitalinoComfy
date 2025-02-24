import numpy as np
import torch
"""
This module defines custom nodes for generating and manipulating periodic signals using PyTorch tensors.
It includes classes for generating sine and cosine waves, as well as summing multiple periodic signals.
Classes:
    SineNode: Generates a sine wave tensor based on specified frequency, amplitude, sample rate, and duration.
    CosineNode: Generates a cosine wave tensor based on specified frequency, amplitude, sample rate, and duration.
    PerSumNode2: Sums two periodic signal tensors, ensuring they have the same time values.
    PerSumNode3: Sums three periodic signal tensors, ensuring they have the same time values.
    PerSumNode4: Sums four periodic signal tensors, ensuring they have the same time values.
Usage:
    These nodes can be used in a signal processing pipeline to generate and combine periodic signals for various applications.
"""

class SineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2 * np.pi, "step": 0.01}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 80000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_sine_wave"
    CATEGORY = "PIC/Obsolete/Modulated Signals"

    def generate_sine_wave(self, frequency, amplitude, phase, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = amplitude * torch.sin(2 * np.pi* frequency * t + phase)
        return (torch.stack([t, signal]),)

class CosineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2 * np.pi, "step": 0.01}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 80000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_cosine_wave"
    CATEGORY = "PIC/Obsolete/Modulated Signals"

    def generate_cosine_wave(self, frequency, amplitude, phase, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = amplitude * torch.cos(2 * np.pi * frequency * t + phase)
        return (torch.stack([t, signal]),)
class NoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 80000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_noise"
    CATEGORY = "PIC/Active/Modulated Signals"

    def generate_noise(self, amplitude, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        noise = amplitude * torch.randn(t.size())
        return (torch.stack([t, noise]),)
class PeriodicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wave_type": (["sine", "cosine"],),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2 * np.pi, "step": 0.01}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 80000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_periodic_wave"
    CATEGORY = "PIC/Active/Modulated Signals"

    def generate_periodic_wave(self, wave_type, frequency, amplitude, phase, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        if wave_type == "sine":
            signal = amplitude * torch.sin(2 * np.pi * frequency * t + phase)
        elif wave_type == "cosine":
            signal = amplitude * torch.cos(2 * np.pi * frequency * t + phase)
        else:
            raise ValueError(f"Unknown wave type: {wave_type}")
        return (torch.stack([t, signal]),)







