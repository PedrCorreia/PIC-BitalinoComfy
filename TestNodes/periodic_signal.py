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
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2000.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2 * np.pi, "step": 0.1}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_sine_wave"
    CATEGORY = "Periodic Signals"

    def generate_sine_wave(self, frequency, amplitude, phase, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = amplitude * torch.sin(2 * np.pi* frequency * t + phase)
        return (torch.stack([t, signal]),)

class CosineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2000.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2 * np.pi, "step": 0.1}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sample_rate": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
            }
        }

    RETURN_TYPES = ("tensor",)
    FUNCTION = "generate_cosine_wave"
    CATEGORY = "Periodic Signals"

    def generate_cosine_wave(self, frequency, amplitude, phase, duration, sample_rate):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = amplitude * torch.cos(2 * np.pi * frequency * t + phase)
        return (torch.stack([t, signal]),)

class PerSumNode2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("tensor",),
                "tensor2": ("tensor",),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "sum_periodic"
    CATEGORY = "Custom Nodes"

    def sum_periodic(self, tensor1, tensor2=None, tensor3=None, tensor4=None):
        tensors = [tensor for tensor in [tensor1, tensor2, tensor3, tensor4] if tensor is not None]

        if not tensors:
            raise ValueError("At least one tensor must be provided.")

        for i, tensor in enumerate(tensors):
            if not torch.equal(tensors[0][0], tensor[0]):
                raise ValueError("All tensors must have the same time values.")
            if tensor.device != torch.device('cpu'):
                tensors[i] = tensor.cpu()

        summed_amplitude = sum(tensor[1] for tensor in tensors)
        combined_tensor = torch.stack([tensors[0][0], summed_amplitude])

        return (combined_tensor,)

class PerSumNode3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("tensor",),
                "tensor2": ("tensor",),
                "tensor3": ("tensor",),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "sum_periodic"
    CATEGORY = "Custom Nodes"

    def sum_periodic(self, tensor1, tensor2=None, tensor3=None, tensor4=None):
        tensors = [tensor for tensor in [tensor1, tensor2, tensor3, tensor4] if tensor is not None]

        if not tensors:
            raise ValueError("At least one tensor must be provided.")

        for i, tensor in enumerate(tensors):
            if not torch.equal(tensors[0][0], tensor[0]):
                raise ValueError("All tensors must have the same time values.")
            if tensor.device != torch.device('cpu'):
                tensors[i] = tensor.cpu()

        summed_amplitude = sum(tensor[1] for tensor in tensors)
        combined_tensor = torch.stack([tensors[0][0], summed_amplitude])

        return (combined_tensor,)

class PerSumNode4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor1": ("tensor",),
                "tensor2": ("tensor",),
                "tensor3": ("tensor",),
                "tensor4": ("tensor",),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "sum_periodic"
    CATEGORY = "Custom Nodes"

    def sum_periodic(self, tensor1, tensor2=None, tensor3=None, tensor4=None):
        tensors = [tensor for tensor in [tensor1, tensor2, tensor3, tensor4] if tensor is not None]

        if not tensors:
            raise ValueError("At least one tensor must be provided.")

        for i, tensor in enumerate(tensors):
            if not torch.equal(tensors[0][0], tensor[0]):
                raise ValueError("All tensors must have the same time values.")
            if tensor.device != torch.device('cpu'):
                tensors[i] = tensor.cpu()

        summed_amplitude = sum(tensor[1] for tensor in tensors)
        combined_tensor = torch.stack([tensors[0][0], summed_amplitude])

        return (combined_tensor,)








