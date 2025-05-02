import os
import json
import random
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import matplotlib.pyplot as plt
import io
import torch
import torchvision.transforms as transforms
import cv2
from collections import deque
from scipy.signal import butter, filtfilt


"""
This module defines custom nodes for plotting signals using PyTorch tensors.
It includes classes for plotting signals and returning the plot as an image tensor, 
as well as classes for summing periodic signals and saving plots.
Classes:
    PlotNode: Plots a given tensor as a signal plot and returns the plot as an image tensor.
    PerSumNode2: Sums two  tensors and returns the result.
    PerSumNode3: Sums three  tensors and returns the result.
    PerSumNode4: Sums four  tensors and returns the result.
    SavePlot: Plots a given tensor and saves the plot as an image file.
    PreviewPlot: Plots a given tensor and saves the plot as a temporary image file.
    SavePlotCustom: Plots a given tensor with custom plot settings and saves the plot as an image file.
    PreviewPlotCustom: Plots a given tensor with custom plot settings and saves the plot as a temporary image file.
"""

class PlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("FLOAT",),  
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Returns an image tensor
    FUNCTION = "plot"
    CATEGORY = "PIC/Obsolete/Tools"

    def plot(self, signal):
        """
        Dynamically display plots in an OpenCV window.

        Parameters:
        - signal: The raw signal tensor (time and amplitude).
        - bandpass_signal: The bandpass-filtered signal tensor (time and amplitude).
        """

        # Initialize plot
        print("Initializing plt")
        fig, axs = plt.subplots(nrows=1, figsize=(10, 3), dpi=100)
        if not isinstance(axs, np.ndarray):  # Ensure axs is always a list
            print("axs is not an array")
            axs = [axs]
   
        # Configure subplots
        print("Configuring subplots")
        axs[0].set_title("Signal")
        axs[0].set_xlim(0, 100)  # Example x-axis limit
        axs[0].set_ylim(-1, 1)  # Adjust based on signal range
        axs[0].grid(True)

        # Initialize plot lines
        signal_line, = axs[0].plot([], [], c="b")

        # Draw the background once
        fig.canvas.draw()
        bg_signal = fig.canvas.copy_from_bbox(axs[0].bbox)

        # Dynamic plotting loop
        signal_deque = deque(maxlen=100)  # Initialize a deque to store signal values
        for i in range(len(signal_deque)):
            signal_deque.append(signal[i])  # Add the current signal value to the deque
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()  # Convert the signal tensor to a NumPy array
        else:
            signal_np = np.array(signal)  # Convert the signal to a NumPy array if it's not a tensor
        for i in range(len(signal_deque)):
            signal_deque.append(signal_np[i])  # Add the current signal value to the deque
            # Update plot lines
            signal_line.set_data(range(i + 1), signal_np[:i + 1])
            fig.canvas.restore_region(bg_signal)
            axs[0].draw_artist(signal_line)
            fig.canvas.blit(axs[0].bbox)

            # Convert the plot to an OpenCV-compatible image
            fig.canvas.draw()
        return fig.canvas.buffer_rgba()

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
    CATEGORY = "PIC/Active/Tools"

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
    CATEGORY = "PIC/Active/Tools"

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
    CATEGORY = "PIC/Active/Tools"

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

class SavePlot:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor", {"tooltip": "The tensor to plot and save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_plot"
    RETURN_TYPES = ()

    OUTPUT_NODE = True

    CATEGORY = "PIC/Active/Tools"
    DESCRIPTION = "Plots the input tensor and saves the resulting image to your ComfyUI output directory."

    def plot_tensor(self, tensor):

        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")

        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()

        print(f"Converted tensor shape: {tensor_np.shape}")
        print(f"Converted tensor dtype: {tensor_np.dtype}")
        x = tensor_np[0]
        y = tensor_np[1]

        plt.switch_backend('Agg') 
        plt.figure(figsize=(5, 3))  # Set the figure size
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Signal Plot')

        # Save the plot as an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # Ensure image is in RGB mode
        plt.close()

        return image

    def save_plot(self, tensor, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        print("Running SavePlot node...")
 
        image = self.plot_tensor(tensor)

        transform = transforms.ToTensor()
        image_tensor = transform(image)

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image_tensor[0].shape[1], image_tensor[0].shape[0])
        results = list()

        for (batch_number, image) in enumerate(image_tensor):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class PreviewPlot(SavePlot):
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
            },
        }

class SavePlotCustom:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor", {"tooltip": "The tensor to plot and save."}),
                "fig_width": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "fig_height": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "axis_y_upper": ("FLOAT", {"default": 5.0, "min": -200000, "max": 2000.0, "step": 0.01}),
                "axis_y_lower": ("FLOAT", {"default": 0.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_x_upper": ("FLOAT", {"default": 5.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_x_lower": ("FLOAT", {"default": 0.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_y_name": ("STRING", {"default": "Amplitude", "tooltip": "Label for the Y-axis."}),
                "axis_x_name": ("STRING", {"default": "Time", "tooltip": "Label for the X-axis."}),
                "title": ("STRING", {"default": "Signal Plot", "tooltip": "Title of the plot."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_plotCustom"
    RETURN_TYPES = ()

    OUTPUT_NODE = True

    CATEGORY = "PIC/Active/Tools"
    DESCRIPTION = "Plots the input tensor and saves the resulting image to your ComfyUI output directory."

    def plot_tensorCustom(self, tensor, fig_width, fig_height, axis_y_upper, axis_y_lower, axis_x_upper, axis_x_lower, axis_y_name, axis_x_name, title):
        """
        Plots the given tensor as a signal plot and returns the image.

        Parameters:
        tensor (torch.Tensor): The input tensor containing time and amplitude.
        fig_width (float): Width of the figure.
        fig_height (float): Height of the figure.
        axis_y_upper (float): Upper limit for the Y-axis.
        axis_y_lower (float): Lower limit for the Y-axis.
        axis_x_upper (float): Upper limit for the X-axis.
        axis_x_lower (float): Lower limit for the X-axis.
        axis_y_name (str): Label for the Y-axis.
        axis_x_name (str): Label for the X-axis.
        title (str): Title of the plot.

        Returns:
        PIL.Image: The plotted image.
        """
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")

        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()

        print(f"Converted tensor shape: {tensor_np.shape}")
        print(f"Converted tensor dtype: {tensor_np.dtype}")
        x = tensor_np[0]
        y = tensor_np[1]

        plt.switch_backend('Agg')
        plt.figure(figsize=(fig_width, fig_height))  # Set the figure size
        plt.plot(x, y)
        plt.xlabel(axis_x_name)
        plt.ylabel(axis_y_name)
        plt.title(title)
        plt.ylim(axis_y_lower, axis_y_upper)
        plt.xlim(axis_x_lower, axis_x_upper)

        # Save the plot as an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # Ensure image is in RGB mode
        plt.close()

        return image

    def save_plotCustom(self, tensor, fig_width, fig_height, axis_y_upper, axis_y_lower, axis_x_upper, axis_x_lower, axis_y_name, axis_x_name, title, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        print("Running SavePlotC node...")

        image = self.plot_tensorCustom(tensor, fig_width, fig_height, axis_y_upper, axis_y_lower, axis_x_upper, axis_x_lower, axis_y_name, axis_x_name, title)
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image_tensor[0].shape[1], image_tensor[0].shape[0])
        results = list()

        for (batch_number, image) in enumerate(image_tensor):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class PreviewPlotCustom(SavePlotCustom):
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
                "tensor": ("tensor", {"tooltip": "The tensor to plot and save."}),
                "fig_width": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "fig_height": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 2000.0, "step": 0.01}),
                "axis_y_upper": ("FLOAT", {"default": 5.0, "min": -200000, "max": 2000.0, "step": 0.01}),
                "axis_y_lower": ("FLOAT", {"default": 0.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_x_upper": ("FLOAT", {"default": 5.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_x_lower": ("FLOAT", {"default": 0.0, "min": -200000, "max": 200000.0, "step": 0.01}),
                "axis_y_name": ("STRING", {"default": "Amplitude", "tooltip": "Label for the Y-axis."}),
                "axis_x_name": ("STRING", {"default": "Time", "tooltip": "Label for the X-axis."}),
                "title": ("STRING", {"default": "Signal Plot", "tooltip": "Title of the plot."}),

            },
        }
