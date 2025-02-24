import numpy as np
import torch
from scipy.signal import butter, filtfilt, firwin

"""
This module provides custom processing nodes for signal processing using PyTorch and SciPy.
The nodes include FFT, low-pass filter, high-pass filter, and band-pass filter functionalities.
Classes:
    FFTNode: Applies Fast Fourier Transform (FFT) to a given tensor and filters frequencies up to a specified maximum frequency.
    LowPassFilterNode: Applies a low-pass filter to a given tensor using the windowed sinc method.
    HighPassFilterNode: Applies a high-pass filter to a given tensor using the windowed sinc method.
    BandPassFilterNode: Applies a band-pass filter to a given tensor using the high-pass and low-pass filters in series.
Methods:
    apply_fft(tensor, max_frequency):
        Applies FFT to the amplitude values of the input tensor and filters frequencies up to max_frequency.
        Args:
            tensor (torch.Tensor): Input tensor containing time and amplitude values.
            max_frequency (float): Maximum frequency to retain in the FFT result.
        Returns:
            tuple: A tuple containing the filtered FFT result as a tensor.
    apply_low_pass_filter(tensor, cutoff_frequency, num_taps):
        Applies a low-pass filter to the amplitude values of the input tensor.
        Args:
            tensor (torch.Tensor): Input tensor containing time and amplitude values.
            cutoff_frequency (float): Cutoff frequency for the low-pass filter.
            num_taps (int): Number of filter taps.
        Returns:
            tuple: A tuple containing the filtered signal as a tensor.
    apply_high_pass_filter(tensor, cutoff_frequency, num_taps):
        Applies a high-pass filter to the amplitude values of the input tensor.
        Args:
            tensor (torch.Tensor): Input tensor containing time and amplitude values.
            cutoff_frequency (float): Cutoff frequency for the high-pass filter.
            num_taps (int): Number of filter taps.
        Returns:
            tuple: A tuple containing the filtered signal as a tensor.
    apply_band_pass_filter(tensor, low_cutoff_frequency, high_cutoff_frequency, num_taps):
        Applies a band-pass filter to the amplitude values of the input tensor using the high-pass and low-pass filters in series.
        Args:
            tensor (torch.Tensor): Input tensor containing time and amplitude values.
            low_cutoff_frequency (float): Low cutoff frequency for the band-pass filter.
            high_cutoff_frequency (float): High cutoff frequency for the band-pass filter.
            num_taps (int): Number of filter taps.
        Returns:
            tuple: A tuple containing the filtered signal as a tensor.
"""

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
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 1001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_low_pass_filter"
    CATEGORY = "Custom Nodes"

    def apply_low_pass_filter(self, tensor, cutoff_frequency, num_taps):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        time_values = tensor[0]
        amplitude_values = tensor[1]

        # Ensure the tensor is contiguous
        if not amplitude_values.is_contiguous():
            amplitude_values = amplitude_values.contiguous()

        # Infer sample rate from time values
        sample_rate = 1.0 / (time_values[1] - time_values[0]).item()
        print(f"Calculated sample rate: {sample_rate}")

        # Design low-pass filter using windowed sinc method
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        taps = firwin(num_taps, normal_cutoff)

        # Convert amplitude values to numpy array and check strides
        amplitude_values_np = amplitude_values.numpy()
        print("Original strides:", amplitude_values_np.strides)

        # Ensure the numpy array has positive strides by making a contiguous copy
        amplitude_values_np = np.ascontiguousarray(amplitude_values_np)
        print("Contiguous strides:", amplitude_values_np.strides)

        # Apply low-pass filter
        try:
            filtered_signal = filtfilt(taps, [1.0], amplitude_values_np)
            # Ensure the filtered signal has positive strides by making a contiguous copy
            filtered_signal = np.ascontiguousarray(filtered_signal)
            print("Filtered signal strides:", filtered_signal.strides)
        except ValueError as e:
            print(f"Error applying filter: {e}")
            return (tensor,)

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
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 1001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_high_pass_filter"
    CATEGORY = "Custom Nodes"

    def apply_high_pass_filter(self, tensor, cutoff_frequency, num_taps):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        time_values = tensor[0]
        amplitude_values = tensor[1]

        # Ensure the tensor is contiguous
        if not amplitude_values.is_contiguous():
            amplitude_values = amplitude_values.contiguous()

        # Infer sample rate from time values
        sample_rate = 1.0 / (time_values[1] - time_values[0]).item()
        print(f"Calculated sample rate: {sample_rate}")

        # Design high-pass filter using windowed sinc method
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        taps = firwin(num_taps, normal_cutoff, pass_zero=False)

        # Convert amplitude values to numpy array and check strides
        amplitude_values_np = amplitude_values.numpy()
        print("Original strides:", amplitude_values_np.strides)

        # Ensure the numpy array has positive strides by making a contiguous copy
        amplitude_values_np = np.ascontiguousarray(amplitude_values_np)
        print("Contiguous strides:", amplitude_values_np.strides)

        # Apply high-pass filter
        try:
            filtered_signal = filtfilt(taps, [1.0], amplitude_values_np)
            # Ensure the filtered signal has positive strides by making a contiguous copy
            filtered_signal = np.ascontiguousarray(filtered_signal)
            print("Filtered signal strides:", filtered_signal.strides)
        except ValueError as e:
            print(f"Error applying filter: {e}")
            return (tensor,)

        # Combine time and filtered signal into a tensor
        combined_tensor = torch.stack([time_values, torch.tensor(filtered_signal, dtype=torch.float32)])

        return (combined_tensor,)

class BandPassFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "low_cutoff_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "high_cutoff_frequency": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 1001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_band_pass_filter"
    CATEGORY = "Custom Nodes"

    def apply_band_pass_filter(self, tensor, low_cutoff_frequency, high_cutoff_frequency, num_taps):
        # Apply high-pass filter first
        high_pass_filter_node = HighPassFilterNode()
        high_passed_tensor = high_pass_filter_node.apply_high_pass_filter(tensor, low_cutoff_frequency, num_taps)[0]

        # Apply low-pass filter to the result of the high-pass filter
        low_pass_filter_node = LowPassFilterNode()
        band_passed_tensor = low_pass_filter_node.apply_low_pass_filter(high_passed_tensor, high_cutoff_frequency, num_taps)[0]

        return (band_passed_tensor,)