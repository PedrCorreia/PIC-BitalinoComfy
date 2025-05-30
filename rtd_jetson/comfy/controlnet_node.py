import os
import torch
import requests
from PIL import Image
from huggingface_hub import snapshot_download

CONTROLNET_MODELS = {
    "depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0-small",
        "local_dir_name": "controlnet-depth-sdxl-1.0-small"
    },
    "canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0-small", 
        "local_dir_name": "controlnet-canny-sdxl-1.0-small"
    }
}

def download_model(controlnet_type, enable_download=True):
    """
    Download ControlNet model directory from Hugging Face.
    
    Args:
        controlnet_type (str): Type of ControlNet model ("depth" or "canny")
        enable_download (bool): Whether to allow downloading if model doesn't exist locally
        
    Returns:
        str: Path to the local model directory
    """
    info = CONTROLNET_MODELS.get(controlnet_type)
    if info is None:
        raise ValueError(f"Unknown controlnet_type: {controlnet_type}")
    
    model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "controlnet_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    model_dir = os.path.join(model_cache_dir, info["local_dir_name"])
    
    # Check if model directory already exists and has required files
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    
    if os.path.exists(config_path) and os.path.exists(model_path):
        print(f"[ControlNetNode] Model already exists: {model_dir}")
        return model_dir
    
    if not enable_download:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found and download is disabled")
        else:
            raise FileNotFoundError(f"Model directory {model_dir} exists but is incomplete and download is disabled")
    
    print(f"[ControlNetNode] Downloading {controlnet_type} model from {info['repo_id']} to {model_dir}")
    try:
        # Download the complete model directory
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=model_dir,
            local_files_only=False,
            ignore_patterns=["*.md", "*.txt", "*.gitignore"]  # Skip unnecessary files
        )
        print(f"[ControlNetNode] Download complete: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"[ControlNetNode] Error downloading model: {e}")
        raise e


class ControlNetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_type": ("STRING", {"default": "depth", "choices": ["depth", "canny"]}),
                "control_image": ("IMAGE", {}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1}),
                "enable_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONTROLNET_PARAMS", )
    RETURN_NAMES = ("controlnet_params", )
    FUNCTION = "run"
    CATEGORY = "LunarRing/visual"

    def run(self, controlnet_type, control_image, scale, enable_download):
        # Download the model if needed
        model_path = download_model(controlnet_type, enable_download)
        
        # Preprocess control image for better ControlNet influence
        processed_image = self.preprocess_control_image(control_image, controlnet_type)
        
        # Return parameters with scale
        params = {
            "model": model_path,
            "control_image": processed_image,
            "scale": scale,
        }
        #print(f"[ControlNetNode] Created params with scale: {scale}")
        return (params,)
    
    def preprocess_control_image(self, control_image, controlnet_type):
        """Preprocess control image to ensure proper format and range for ControlNet"""
        import torch
        
        if isinstance(control_image, torch.Tensor):
            # Ensure proper range [0, 1] for ControlNet
            if control_image.max() > 1.0:
                control_image = control_image / 255.0
            
            # For depth maps, ensure proper contrast
            if controlnet_type == "depth":
                # Enhance contrast for better depth perception
                control_image = torch.clamp(control_image, 0, 1)
                # Normalize to full range to maximize depth effect
                min_val = control_image.min()
                max_val = control_image.max()
                if max_val > min_val:
                    control_image = (control_image - min_val) / (max_val - min_val)
                
                print(f"[ControlNetNode] Depth preprocessing - range: [{control_image.min():.3f}, {control_image.max():.3f}]")
            
            return control_image
        else:
            print(f"[ControlNetNode] Warning: Unexpected control_image type: {type(control_image)}")
            return control_image
