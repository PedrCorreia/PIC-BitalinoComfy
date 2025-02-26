import os
import json
import random
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from comfy.cli_args import args
import matplotlib.pyplot as plt
import io
import torch
import torchvision.transforms as transforms


"""
This module defines a custom node for plotting signals using PyTorch tensors.
It includes a class for plotting signals and returning the plot as an image tensor.
Classes:
    PlotNode: Plots a given tensor as a signal plot and returns the plot as an image tensor.
Usage:
    This node can be used in a signal processing pipeline to plot signals for various applications.
"""

class PlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("tensor",),  
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Returns an image tensor
    FUNCTION = "plot"
    CATEGORY = "PIC/Obsolete/Tools"

    def plot(self, tensor):
        """
        Plots the given tensor as a signal plot.

        Parameters:
        tensor (torch.Tensor): The input tensor containing time and amplitude.

        Returns:
        tuple: A tuple containing the image tensor.
        """
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()

        plt.figure(figsize=(5, 3))  # Set the figure size
        plt.plot(tensor_np[0], tensor_np[1])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Signal Plot')

        # Save the plot as an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()

        # Convert image to tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        return image_tensor,
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
        """
        Plots the given tensor as a signal plot and returns the image.

        Parameters:
        tensor (torch.Tensor): The input tensor containing time and amplitude.

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
        """
        Plots the given tensor as a signal plot and saves the image.

        Parameters:
        tensor (torch.Tensor): The input tensor containing time and amplitude.

        Returns:
        dict: A dictionary containing the saved image information.
        """
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



       