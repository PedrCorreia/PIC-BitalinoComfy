import cv2
import numpy as np
import torch 
import os
import time
from functools import lru_cache
import logging
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('depth_processor')

class ImgUtils:
    @staticmethod
    def resize_image(image, size, keep_aspect_ratio=True, resample=Image.Resampling.LANCZOS):
        """
        Resizes a PIL Image.
        Args:
            image (PIL.Image.Image): The image to resize.
            size (tuple): The target (width, height).
            keep_aspect_ratio (bool): If True, keeps aspect ratio and pads if necessary.
                                      If False, resizes to the exact size.
            resample (Image.Resampling): Resampling filter.
        Returns:
            PIL.Image.Image: The resized image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input image must be a PIL.Image.Image instance.")
        
        if keep_aspect_ratio:
            background_color = (0, 0, 0, 0) if image.mode == 'RGBA' else (0, 0, 0)
            resized_image = Image.new(image.mode, size, background_color)
            img_copy = image.copy()
            img_copy.thumbnail(size, resample)
            paste_x = (size[0] - img_copy.width) // 2
            paste_y = (size[1] - img_copy.height) // 2
            resized_image.paste(img_copy, (paste_x, paste_y))
        else:
            resized_image = image.resize(size, resample=resample)
        return resized_image

    @staticmethod
    def pil_to_numpy(pil_image):
        """
        Converts a PIL Image to a NumPy array.
        Returns a (H, W, C) NumPy array (RGB or RGBA).
        """
        if not isinstance(pil_image, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image instance.")
        return np.array(pil_image)

    @staticmethod
    def numpy_to_pil(numpy_array, mode=None):
        """
        Converts a NumPy array to a PIL Image.
        Assumes input is (H, W, C) for RGB/RGBA or (H, W) for L.
        Handles (H, W, 1) by converting to (H, W) for mode 'L'.
        Input array should be of dtype uint8 or be ready for direct conversion via .astype(np.uint8).
        """
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        
        # Store original shape for error messages if any
        original_shape = numpy_array.shape

        if mode is None:
            if numpy_array.ndim == 3:
                if numpy_array.shape[2] == 3:
                    mode = 'RGB'
                elif numpy_array.shape[2] == 4:
                    mode = 'RGBA'
                elif numpy_array.shape[2] == 1:
                    mode = 'L'
                    # Squeeze the last dimension to make it (H, W) for 'L' mode.
                    # This is generally safer for PIL's fromarray.
                    logger.debug(f"Squeezing numpy array from {original_shape} to {numpy_array.squeeze(axis=2).shape} for PIL 'L' mode.")
                    numpy_array = numpy_array.squeeze(axis=2)
                else:
                    raise ValueError(f"Cannot infer PIL mode from 3D array with shape {original_shape}. Number of channels not 1, 3, or 4.")
            elif numpy_array.ndim == 2:
                mode = 'L' # Grayscale
            else:
                raise ValueError(f"Cannot infer PIL mode from array shape {original_shape}. Expected 2D or 3D array.")
        
        # Ensure the array is C-contiguous, especially after a squeeze, can prevent some PIL issues.
        # Note: astype(np.uint8) usually returns a C-contiguous array if the original is.
        # If issues persist, uncommenting this might help, but usually not needed here.
        # if not numpy_array.flags['C_CONTIGUOUS']:
        #     logger.debug(f"Making numpy array C-contiguous for PIL conversion. Original flags: {numpy_array.flags}")
        #     numpy_array = np.ascontiguousarray(numpy_array)
            
        # PIL's fromarray expects uint8. If the input is not uint8, it must be convertible.
        # Common cases: float [0,1] -> [0,255] uint8, or float [0,255] -> uint8.
        # The original function relied on a simple .astype(np.uint8).
        # For robustness, one might add scaling logic here if inputs are float and not in [0,255] range.
        # However, to keep changes minimal and aligned with original intent:
        if numpy_array.dtype != np.uint8:
            if np.issubdtype(numpy_array.dtype, np.floating):
                # Assuming float arrays are in [0,1] or [0,255] range.
                # If in [0,1], scale it. If already [0,255] float, astype is fine.
                # This simple check might not cover all cases but is a common one.
                if numpy_array.max() <= 1.0 and numpy_array.min() >= 0.0: # Likely [0,1] float
                    logger.debug(f"Scaling float numpy array from [0,1] to [0,255] for PIL conversion.")
                    numpy_array = (numpy_array * 255)
                # else: assume it's already in [0,255] float range or needs clipping by astype
            # Clip and convert to uint8. This handles integers outside 0-255 and floats.
            numpy_array = np.clip(numpy_array, 0, 255)

        return Image.fromarray(numpy_array.astype(np.uint8), mode=mode)

    @staticmethod
    def numpy_to_tensor(numpy_array: np.ndarray) -> torch.Tensor:
        """Converts a NumPy array (H, W, C) [0, 255] to a PyTorch tensor (1, H, W, C) [0, 1]."""
        if numpy_array.ndim == 2: # Handle grayscale (H, W)
            numpy_array = np.expand_dims(numpy_array, axis=2) # (H, W, 1)
        if numpy_array.shape[2] == 1: # Grayscale to RGB if needed by some nodes
             numpy_array = np.concatenate([numpy_array]*3, axis=2)

        if numpy_array.dtype == np.uint8:
            tensor = torch.from_numpy(numpy_array).float() / 255.0
        elif numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
            tensor = torch.from_numpy(numpy_array.astype(np.float32)) # Ensure it's float32
        else:
            raise TypeError(f"Unsupported numpy_array dtype: {numpy_array.dtype}")
        return tensor.unsqueeze(0) # Add batch dimension

    @staticmethod
    def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a PyTorch tensor (1, H, W, C) [0, 1]."""
        numpy_array = ImgUtils.pil_to_numpy(pil_image)
        return ImgUtils.numpy_to_tensor(numpy_array)

    @staticmethod
    def tensor_to_numpy(tensor_image: torch.Tensor) -> np.ndarray:
        """Converts a PyTorch tensor (B, H, W, C) or (H, W, C) [0, 1] to a NumPy array (H, W, C) [0, 255]."""
        #print(f"[ImgUtils.tensor_to_numpy] Input tensor_image shape: {tensor_image.shape}, dtype: {tensor_image.dtype}, min: {tensor_image.min():.4f}, max: {tensor_image.max():.4f}")
        if tensor_image.ndim == 4: # (B, H, W, C)
            tensor_image = tensor_image.squeeze(0) # Remove batch dimension if present
        
        numpy_array = tensor_image.cpu().numpy()
        # Determine if scaling is needed based on range
        if numpy_array.min() >= 0 and numpy_array.max() <= 1.0 and (numpy_array.max() - numpy_array.min()) > 1e-5 : # Check if it's likely in [0,1] range and not flat
             #print(f"[ImgUtils.tensor_to_numpy] Scaling from [0,1] to [0,255]")
             numpy_array = (numpy_array * 255)
        else:
             #print(f"[ImgUtils.tensor_to_numpy] Assuming already in [0,255] or not a standard [0,1] float image, just casting dtype.")
             pass
        
        numpy_array = numpy_array.astype(np.uint8)
        
        #print(f"[ImgUtils.tensor_to_numpy] Output numpy_array shape: {numpy_array.shape}, dtype: {numpy_array.dtype}, min: {numpy_array.min()}, max: {numpy_array.max()}")
        return numpy_array

    @staticmethod
    def tensor_to_pil(tensor_image: torch.Tensor, mode='RGB') -> Image.Image:
        """Converts a PyTorch tensor to a PIL Image."""
        numpy_array = ImgUtils.tensor_to_numpy(tensor_image)
        return ImgUtils.numpy_to_pil(numpy_array, mode=mode)

class Canny:
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, img):
        """Apply Canny edge detection to a grayscale or color image (numpy array)."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, self.low_threshold, self.high_threshold)
        return edges

class Midas:
    def __init__(self, model_type='MiDaS_small', device=None, force_perspective=False, min_z=-15, max_blur=8, use_half_precision=None, optimize_memory=False):
        self.model_type = model_type
        
        # Auto-select best available device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Auto-detect if half precision should be used based on device capabilities
        if use_half_precision is None and 'cuda' in self.device:
            self.use_half_precision = torch.cuda.is_available() and \
                                     torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 6
        else:
            self.use_half_precision = bool(use_half_precision)
            
        self.optimize_memory = optimize_memory
        self.model = None
        self.transform = None
        self._loaded = False
        self.force_perspective = force_perspective
        self.min_z = min_z
        self.max_blur = max_blur

    def load_model(self):
        """Load the MiDaS model with caching for better performance"""
        if self._loaded:
            return
        self.model = torch.hub.load('intel-isl/MiDaS', self.model_type)  # type: ignore
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')  # type: ignore
        if self.model_type == 'MiDaS_small':
            self.transform = getattr(transforms, 'small_transform', None)
        else:
            self.transform = getattr(transforms, 'default_transform', None)
        self._loaded = True

    def apply_perspective_blur(self, img, z):
        """Apply logarithmic Gaussian blur based on z position."""
        if z < 0:
            if z <= self.min_z:
                blur_sigma = self.max_blur
            else:
                blur_sigma = self.max_blur * (np.log1p(abs(z)) / np.log1p(abs(self.min_z)))
        else:
            blur_sigma = 0
        if blur_sigma > 0.2:
            ksize = int(2 * round(blur_sigma) + 1)
            img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
        return img

    def predict(self, img, z=None, optimize_size=True):
        """Predict depth map from an input image (numpy array, BGR or RGB). Optionally apply perspective blur if enabled."""
        self.load_model()
        
        # Simple image caching - avoid reprocessing the exact same image
        # --- CACHE BYPASSED FOR DEBUGGING ---
        # img_hash = hash(img.tobytes())
        # if img_hash in self._depth_cache:
        #     logger.info("Using cached depth result (DEBUG: CACHE DISABLED, THIS SHOULD NOT BE HIT)")
        #     return self._depth_cache[img_hash]
        # --- END CACHE BYPASS ---
        
        # Convert image to RGB if needed
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Optimize image size if requested (resize large images for speed)
        original_size = img_rgb.shape[:2]
        resized = False
        if optimize_size and (original_size[0] > 512 or original_size[1] > 512):
            # Keep aspect ratio
            max_dim = max(original_size)
            scale_factor = 512 / max_dim
            new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
            img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
            resized = True
            # logger.info(f"Resized image from {original_size} to {img_rgb.shape[:2]} for faster processing")
        
        # Check if transform is properly loaded
        if self.transform is None:
            raise RuntimeError('MiDaS transform not loaded')
        input_tensor = self.transform(img_rgb).to(self.device)  # type: ignore
        if self.model is None:
            raise RuntimeError('MiDaS model not loaded')
        with torch.no_grad():
            prediction = self.model(input_tensor)  # type: ignore
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            output = prediction.cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        if self.force_perspective and z is not None:
            output = self.apply_perspective_blur(output, z)
        return output
