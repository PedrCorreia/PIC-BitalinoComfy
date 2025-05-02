import numpy as np
import torch
from scipy.signal import filtfilt, firwin
from .tools import SavePlot,  SavePlotCustom
import random
import folder_paths

"""
This module provides custom processing nodes for signal processing using PyTorch and SciPy.
The nodes include FFT, low-pass filter, high-pass filter, and band-pass filter functionalities.
Classes:
    FFTNode: Applies Fast Fourier Transform (FFT) to a given tensor and filters frequencies up to a specified maximum frequency.
    LowPassFilterNode: Applies a low-pass filter to a given tensor using the windowed sinc method.
    HighPassFilterNode: Applies a high-pass filter to a given tensor using the windowed sinc method.
    BandPassFilterNode: Applies a band-pass filter to a given tensor using the high-pass and low-pass filters in series.
    FilterNode: Allows selection of low-pass, high-pass, or band-pass filter to apply to a given tensor.
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
                "max_frequency": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_fft"
    CATEGORY = "PIC/Active/Basic Signal Processing"

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
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 5001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_low_pass_filter"
    CATEGORY = "PIC/Obsolete/Basic Signal Processing"

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
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 5001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_high_pass_filter"
    CATEGORY = "PIC/Obsolete/Basic Signal Processing"

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
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 5001, "step": 1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_band_pass_filter"
    CATEGORY = "PIC/Obsolete/Basic Signal Processing"

    def apply_band_pass_filter(self, tensor, low_cutoff_frequency, high_cutoff_frequency, num_taps):
        # Apply high-pass filter first
        high_pass_filter_node = HighPassFilterNode()
        high_passed_tensor = high_pass_filter_node.apply_high_pass_filter(tensor, low_cutoff_frequency, num_taps)[0]

        # Apply low-pass filter to the result of the high-pass filter
        low_pass_filter_node = LowPassFilterNode()
        band_passed_tensor = low_pass_filter_node.apply_low_pass_filter(high_passed_tensor, high_cutoff_frequency, num_taps)[0]

        return (band_passed_tensor,)

class FilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "filter_type": (["low_pass", "high_pass", "band_pass"],),
                "cutoff_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "num_taps": ("INT", {"default": 101, "min": 1, "max": 5001, "step": 1}),
            },
            "optional": {
                "low_cutoff_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "high_cutoff_frequency": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "apply_filter"
    CATEGORY = "PIC/Active/Basic Signal Processing"

    def apply_filter(self, tensor, filter_type, cutoff_frequency, num_taps, low_cutoff_frequency=None, high_cutoff_frequency=None):
        if filter_type == "low_pass":
            low_pass_filter_node = LowPassFilterNode()
            return low_pass_filter_node.apply_low_pass_filter(tensor, cutoff_frequency, num_taps)
        elif filter_type == "high_pass":
            high_pass_filter_node = HighPassFilterNode()
            return high_pass_filter_node.apply_high_pass_filter(tensor, cutoff_frequency, num_taps)
        elif filter_type == "band_pass":
            if low_cutoff_frequency is None or high_cutoff_frequency is None:
                raise ValueError("Both low_cutoff_frequency and high_cutoff_frequency must be provided for band_pass filter")
            band_pass_filter_node = BandPassFilterNode()
            return band_pass_filter_node.apply_band_pass_filter(tensor, low_cutoff_frequency, high_cutoff_frequency, num_taps)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

class SaveFFT:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor", {"tooltip": "The tensor to apply the fft."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "maxfreq" : ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_FFT"
    RETURN_TYPES = ()

    OUTPUT_NODE = True

    CATEGORY = "PIC/Active/Basic Signal Processing"
    DESCRIPTION = "Plots the FFT tensor and saves the resulting image to your ComfyUI output directory."
    def save_FFT(self, tensor,maxfreq, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        print("Running FFT node...")
        fft_node = FFTNode()
        FFT = fft_node.apply_fft(tensor, maxfreq)  # Assuming max_frequency is 200
        save_plot_instance = SavePlot()
        results = save_plot_instance.save_plot(FFT[0], filename_prefix, prompt, extra_pnginfo)
        return results
class PreviewFFTNode(SaveFFT):
    def __init__(self):
        print("Initializing PreviewImageBETA node...")
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),
                "maxfreq" : ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
            },
        }

class DiscreteTransferFunctionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_signal": ("tensor",),
                "output_signal": ("tensor",),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "calculate_transfer_function"
    CATEGORY = "PIC/Obsolete/Basic Signal Processing"

    def calculate_transfer_function(self, input_signal, output_signal):
        if input_signal.device != torch.device('cpu'):
            input_signal = input_signal.cpu()
        if output_signal.device != torch.device('cpu'):
            output_signal = output_signal.cpu()

        input_time_values = input_signal[0]
        input_amplitude_values = input_signal[1]
        output_amplitude_values = output_signal[1]

        # Apply FFT to input and output signals
        input_fft = torch.fft.fft(input_amplitude_values)
        output_fft = torch.fft.fft(output_amplitude_values)
        fft_freqs = torch.fft.fftfreq(len(input_amplitude_values), d=(input_time_values[1] - input_time_values[0]).item())

        # Calculate transfer function (H(f) = Y(f) / X(f))
        transfer_function = output_fft / input_fft

        # Filter positive frequencies
        positive_freqs = fft_freqs > 0
        fft_freqs = fft_freqs[positive_freqs]
        transfer_function = transfer_function[positive_freqs]

        # Calculate magnitude and phase
        magnitude = torch.abs(transfer_function)
        # Combine frequency and magnitude/phase into tensors
        magnitude_tensor = torch.stack([torch.tensor(fft_freqs, dtype=torch.float32), magnitude])

        return (magnitude_tensor,)


class FrequencySamplingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frequencies": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "max_frequency": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("tensor",)  # Must be a tuple (comma needed)
    FUNCTION = "sample_frequencies"
    CATEGORY = "PIC/Active/Basic Signal Processing"

    def sample_frequencies(self, num_frequencies, max_frequency):
        # Generate linearly spaced frequencies
        frequencies = np.linspace(0, max_frequency, num_frequencies)
        
        # Create a signal with these frequencies
        time = np.linspace(0, 1, 1000)  # 1 second duration, 1000 samples
        signal = np.zeros_like(time)
        for freq in frequencies:
            signal += np.sin(2 * np.pi * freq * time)
        
        # Convert to tensor
        time_tensor = torch.tensor(time, dtype=torch.float32)
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        combined_tensor = torch.stack([time_tensor, signal_tensor])

        return (combined_tensor,)
class SavePlot_H:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_signal": ("tensor", {"tooltip": "The tensor containing the input signal."}),
                "output_signal": ("tensor", {"tooltip": "The tensor containing the output signal."}),
                "filename_prefix": ("STRING", {"default": "TransferFunction", "tooltip": "The prefix for the file to save."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_plot_H"
    RETURN_TYPES = ()

    OUTPUT_NODE = True

    CATEGORY = "PIC/Active/Basic Signal Processing"
    DESCRIPTION = "Plots the transfer function and saves the resulting image to your ComfyUI output directory."

    def save_plot_H(self, input_signal, output_signal, filename_prefix="TransferFunction", prompt=None, extra_pnginfo=None):
        # Calculate the transfer function using DiscreteTransferFunctionNode
        transfer_function_node = DiscreteTransferFunctionNode()
        transfer_function_tensor = transfer_function_node.calculate_transfer_function(input_signal, output_signal)[0]

        # Use SavePlot instance to save the plot
        save_plot_instance = SavePlot()
        results = save_plot_instance.save_plot(transfer_function_tensor, filename_prefix, prompt, extra_pnginfo)
        return results

class PreviewPlot_H(SavePlot_H):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_signal": ("tensor", {"tooltip": "The tensor containing the input signal."}),
                "output_signal": ("tensor", {"tooltip": "The tensor containing the output signal."}),
            },
        }
class SaveFFTCustom:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "tensor": ("tensor", {"tooltip": "The tensor to apply the fft."}),
                    "FFT_max_freq": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000000.0, "step": 0.01}),
                    "fig_width": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                    "fig_height": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                    "axis_y_upper": ("FLOAT", {"default": 5.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                    "axis_y_lower": ("FLOAT", {"default": 0.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                    "axis_x_upper": ("FLOAT", {"default": 5.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                    "axis_x_lower": ("FLOAT", {"default": 0.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                    "axis_y_name": ("STRING", {"default": "Amplitude", "tooltip": "Label for the Y-axis."}),
                    "axis_x_name": ("STRING", {"default": "Frequency", "tooltip": "Label for the X-axis."}),
                    "title": ("STRING", {"default": "FFT Plot", "tooltip": "Title of the plot."}),
                    "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                },
            }

        FUNCTION = "save_FFTCustom"
        RETURN_TYPES = ()

        OUTPUT_NODE = True

        CATEGORY = "PIC/Active/Basic Signal Processing"
        DESCRIPTION = "Plots the FFT tensor with custom settings and saves the resulting image to your ComfyUI output directory."

        def save_FFTCustom(self, tensor, fig_width, fig_height, axis_y_upper, axis_y_lower, axis_x_upper, axis_x_lower, axis_y_name, axis_x_name, title, FFT_max_freq, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
            fft_node = FFTNode()
            FFT = fft_node.apply_fft(tensor, FFT_max_freq)
            save_plot_instance = SavePlotCustom()
            results = save_plot_instance.save_plotCustom(FFT[0],fig_width, fig_height, axis_y_upper, axis_y_lower, axis_x_upper, axis_x_lower, axis_y_name, axis_x_name, title, filename_prefix, prompt, extra_pnginfo)
            return results

class PreviewFFTCustom(SaveFFTCustom):
                def __init__(self):
                    self.output_dir = folder_paths.get_temp_directory()
                    self.type = "temp"
                    self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
                    self.compress_level = 1

                @classmethod
                def INPUT_TYPES(cls):
                    return {
                        "required": {
                            "tensor": ("tensor", {"tooltip": "The tensor to apply the fft."}),
                            "FFT_max_freq": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                            "fig_width": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                            "fig_height": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                            "axis_y_upper": ("FLOAT", {"default": 5.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                            "axis_y_lower": ("FLOAT", {"default": 0.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                            "axis_x_upper": ("FLOAT", {"default": 5.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                            "axis_x_lower": ("FLOAT", {"default": 0.0, "min": -2000000000000.0, "max": 2000000000000.0, "step": 0.01}),
                            "axis_y_name": ("STRING", {"default": "Amplitude", "tooltip": "Label for the Y-axis."}),
                            "axis_x_name": ("STRING", {"default": "Frequency", "tooltip": "Label for the X-axis."}),
                            "title": ("STRING", {"default": "FFT Plot", "tooltip": "Title of the plot."}),
                        },
                    }
