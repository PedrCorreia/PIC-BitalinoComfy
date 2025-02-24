import torch
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
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
    CATEGORY = "PIC/Active/Tools"

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
        image = Image.open(buf).convert('RGB')  # Ensure image is in RGB mode
        plt.close()

        # Convert image to tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        return (image_tensor,)
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