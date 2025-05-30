import os
import torch
import requests
from PIL import Image

CONTROLNET_MODELS = {
    "depth": {
        "url": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.safetensors",
        "filename": "controlnet-depth-sdxl-1.0-small.safetensors"
    },
    "canny": {
        "url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.safetensors",
        "filename": "controlnet-canny-sdxl-1.0-small.safetensors"
    }
}



class ControlNetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_type": ("STRING", {"default": "depth", "choices": ["depth", "canny"]}),
                "control_image": ("IMAGE", {}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONTROLNET_PARAMS", )
    RETURN_NAMES = ("controlnet_params", )
    FUNCTION = "run"
    CATEGORY = "LunarRing/visual"

    def run(self, controlnet_type, control_image, scale):
        model_path = download_model(controlnet_type)
        # Optionally, load the model here if needed, or just pass the path
        params = {
            "model": model_path,
            "control_image": control_image,
            "scale": scale
        }
        return (params,)
