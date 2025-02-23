import numpy as np
import torch
from scipy.signal import butter, filtfilt

class FFTNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "max_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_fft"
    CATEGORY = "Custom Nodes"

    def apply_fft(self, tensor, max_frequency):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        time_values = tensor[0]
        amplitude_values = tensor[1]

        # Apply FFT
        fft_result = torch.fft.fft(amplitude_values)
        fft_freqs = torch.fft.fftfreq(len(amplitude_values), d=(time_values[1] - time_values[0]).item())

        # Filter positive frequencies and apply max frequency limit
        positive_freqs = fft_freqs > 0
        freq_mask = (fft_freqs <= max_frequency) & positive_freqs

        filtered_freqs = fft_freqs[freq_mask]
        filtered_fft_result = fft_result[freq_mask]

        # Combine frequency and FFT result into a tensor
        combined_tensor = torch.stack([torch.tensor(filtered_freqs, dtype=torch.float32), torch.abs(filtered_fft_result)])

        return (combined_tensor,)

class LowPassFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "cutoff_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_low_pass_filter"
    CATEGORY = "Custom Nodes"

    def apply_low_pass_filter(self, tensor, cutoff_frequency):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        time_values = tensor[0]
        amplitude_values = tensor[1]

        # Infer sample rate from time values
        sample_rate = 1.0 / (time_values[1] - time_values[0]).item()

        # Design low-pass filter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)

        # Apply low-pass filter
        filtered_signal = filtfilt(b, a, amplitude_values.numpy())

        # Combine time and filtered signal into a tensor
        combined_tensor = torch.stack([time_values, torch.tensor(filtered_signal, dtype=torch.float32)])

        return (combined_tensor,)

class HighPassFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "cutoff_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_high_pass_filter"
    CATEGORY = "Custom Nodes"

    def apply_high_pass_filter(self, tensor, cutoff_frequency):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        time_values = tensor[0]
        amplitude_values = tensor[1]

        # Infer sample rate from time values
        sample_rate = 1.0 / (time_values[1] - time_values[0]).item()

        # Design high-pass filter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)

        # Apply high-pass filter
        filtered_signal = filtfilt(b, a, amplitude_values.numpy())

        # Combine time and filtered signal into a tensor
        combined_tensor = torch.stack([time_values, torch.tensor(filtered_signal, dtype=torch.float32)])

        return (combined_tensor,)