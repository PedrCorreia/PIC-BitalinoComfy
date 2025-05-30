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
        print(f"[ImgUtils.tensor_to_numpy] Input tensor_image shape: {tensor_image.shape}, dtype: {tensor_image.dtype}, min: {tensor_image.min():.4f}, max: {tensor_image.max():.4f}")
        if tensor_image.ndim == 4: # (B, H, W, C)
            tensor_image = tensor_image.squeeze(0) # Remove batch dimension if present
        
        numpy_array = tensor_image.cpu().numpy()
        # Determine if scaling is needed based on range
        if numpy_array.min() >= 0 and numpy_array.max() <= 1.0 and (numpy_array.max() - numpy_array.min()) > 1e-5 : # Check if it's likely in [0,1] range and not flat
             print(f"[ImgUtils.tensor_to_numpy] Scaling from [0,1] to [0,255]")
             numpy_array = (numpy_array * 255)
        else:
             print(f"[ImgUtils.tensor_to_numpy] Assuming already in [0,255] or not a standard [0,1] float image, just casting dtype.")
        
        numpy_array = numpy_array.astype(np.uint8)
        print(f"[ImgUtils.tensor_to_numpy] Output numpy_array shape: {numpy_array.shape}, dtype: {numpy_array.dtype}, min: {numpy_array.min()}, max: {numpy_array.max()}")
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
    # Cache of model instances to avoid reloading
    _model_cache = {}
    
    def __init__(self, model_type='MiDaS_small', device=None, optimize_memory=True, use_half_precision=None):
        """
        Initialize the MiDaS depth prediction model
        
        Args:
            model_type: Model type to use ('MiDaS_small', 'DPT_Large', etc.)
            device: Device to use ('cuda', 'cpu', or specific like 'cuda:0')
            optimize_memory: Whether to optimize memory usage by clearing cache
            use_half_precision: Use half precision (fp16) for faster inference on supported hardware
        """
        self.model_type = model_type
        
        # Auto-select best available device if not specified
        if device is None:
            self.device = self._get_optimal_device()
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
        self._cache_key = f"{model_type}_{self.device}_{self.use_half_precision}"
        
        # Cache for processed images - MRU cache of size 3
        self._depth_cache = {}
        self._cache_keys = []
        self._max_cache_size = 3
        
        # Performance tracking
        self._inference_times = []
        self._max_times_tracked = 5
        
        logger.info(f"Initialized MiDaS ({model_type}) on {self.device} " +
                   f"{'with' if self.use_half_precision else 'without'} half precision")
    
    def _get_optimal_device(self):
        """Determine the best available device for inference"""
        if torch.cuda.is_available():
            # Check GPU memory to determine the best device
            device_count = torch.cuda.device_count()
            if device_count > 1:
                # Select GPU with most free memory
                max_free_mem = 0
                best_device = 0
                for i in range(device_count):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_mem = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                    if free_mem > max_free_mem:
                        max_free_mem = free_mem
                        best_device = i
                return f"cuda:{best_device}"
            return "cuda"
        
        # Check if MPS is available (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
            
        return "cpu"
    
    def load_model(self):
        """Load the MiDaS model with caching for better performance"""
        if self._loaded:
            return
            
        # Check if this model configuration is already cached
        if self._cache_key in self._model_cache:
            logger.info(f"Using cached model for {self._cache_key}")
            self.model, self.transform = self._model_cache[self._cache_key]
            self._loaded = True
            return
            
        start_time = time.time()
        
        try:
            # Before loading model, optimize memory
            if self.optimize_memory and 'cuda' in self.device:
                torch.cuda.empty_cache()
                
            # Check if custom model path is available in environment
            custom_model_path = os.environ.get('MIDAS_MODEL_PATH', '')
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"Loading custom MiDaS model from {custom_model_path}")
                try:
                    self.model = torch.load(custom_model_path, map_location=self.device)
                    # Try to get transform from common locations
                    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', pretrained=False)
                    if self.model_type == 'MiDaS_small':
                        self.transform = getattr(transforms, 'small_transform', None)
                    else:
                        self.transform = getattr(transforms, 'default_transform', None)
                    
                except Exception as e:
                    logger.warning(f"Failed to load custom model: {e}")
                    self.model = None  # Will be loaded from torch hub below
            
            # If custom model failed or wasn't specified, use torch hub
            if self.model is None:
                logger.info(f"Loading MiDaS model {self.model_type} from torch hub")
                self.model = torch.hub.load('intel-isl/MiDaS', self.model_type)
                transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                if self.model_type == 'MiDaS_small':
                    self.transform = getattr(transforms, 'small_transform', None)
                else:
                    self.transform = getattr(transforms, 'default_transform', None)
                    
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Use half precision if requested and supported
            if self.use_half_precision:
                logger.info("Converting model to half precision (fp16)")
                self.model = self.model.half()
                
            # Set model to evaluation mode
            self.model.eval()
            
            # Cache the model for future use
            self._model_cache[self._cache_key] = (self.model, self.transform)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Error loading MiDaS model: {e}")
            raise RuntimeError(f"Failed to load MiDaS model: {e}")
    
    @lru_cache(maxsize=8)
    def _get_cached_hash(self, image_bytes):
        """Helper for image caching using hashing - used with the lru_cache decorator"""
        return hash(image_bytes)
    
    def predict(self, img, optimize_size=True):
        """
        Predict depth map from an input image (numpy array, BGR or RGB)
        
        Args:
            img: Input image as numpy array
            optimize_size: Whether to optimize image size for faster processing
            
        Returns:
            Depth map as numpy array
        """
        start_time = time.time()
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
            logger.info(f"Resized image from {original_size} to {img_rgb.shape[:2]} for faster processing")
        
        # Check if transform is properly loaded
        if self.transform is None:
            raise RuntimeError('MiDaS transform not loaded')
            
        # Process the image with the transform and run inference
        try:
            logger.debug("Running MiDaS inference")
            
            # Apply transform
            input_batch = self.transform(img_rgb)
            
            # Convert to half precision if requested
            if self.use_half_precision:
                input_batch = input_batch.half()
                
            # Move input to device
            input_batch = input_batch.to(self.device)
            
            # Ensure model is loaded
            if self.model is None:
                raise RuntimeError('MiDaS model not loaded')
                
            # Run inference without gradient calculation
            with torch.no_grad():
                # Get model prediction
                prediction = self.model(input_batch)
                
                # If model returns a list/tuple (some MiDaS models), take the first element
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]
                
                # prediction is likely (B, H_model, W_model). Unsqueeze to (B, 1, H_model, W_model) for interpolate
                # Interpolate output will be (B, 1, H_out, W_out)
                interpolated_prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2], # Target H_out, W_out from (potentially resized) img_rgb
                    mode='bicubic',
                    align_corners=False
                )
                # Squeeze batch and channel dimensions to get (H_out, W_out)
                # Assumes B=1 (batch size is 1)
                output_tensor_2d = interpolated_prediction.squeeze(0).squeeze(0) 
                
                # Convert to numpy and normalize
                output = output_tensor_2d.cpu().numpy()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                
                # Ensure output is a proper float32 format before resizing
                output = output.astype(np.float32)
                
                # Resize back to original size if needed
                if resized:
                    # Make sure output is properly normalized and in the right format for resizing
                    if output.min() < 0 or output.max() > 1:
                        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                        
                    # Use cv2.resize with proper interpolation
                    output = cv2.resize(
                        output, 
                        (original_size[1], original_size[0]),  # Width, height
                        interpolation=cv2.INTER_CUBIC
                    )
                
                # Update inference time statistics
                inference_time = time.time() - start_time
                self._update_timing(inference_time)
                
                # Add result to cache
                # --- CACHE BYPASSED FOR DEBUGGING (Don't add to cache) ---
                # self._add_to_cache(img_hash, output)
                # --- END CACHE BYPASS ---
                
                # Clean up GPU memory if optimizing for memory
                if self.optimize_memory and 'cuda' in self.device:
                    torch.cuda.empty_cache()
            
            logger.debug(f"[Midas.predict] Returning output with shape: {output.shape}, dtype: {output.dtype}")
            return output
                
        except Exception as e:
            logger.error(f"Error during depth prediction: {e}")
            raise
    
    def _update_timing(self, inference_time):
        """Update inference timing statistics"""
        self._inference_times.append(inference_time)
        if len(self._inference_times) > self._max_times_tracked:
            self._inference_times.pop(0)
        avg_time = sum(self._inference_times) / len(self._inference_times)
        logger.info(f"Depth prediction completed in {inference_time:.3f}s (avg: {avg_time:.3f}s)")
    
    def _add_to_cache(self, img_hash, result):
        """Add result to the MRU cache"""
        if len(self._cache_keys) >= self._max_cache_size:
            # Remove least recently used item
            oldest_key = self._cache_keys.pop(0)
            if oldest_key in self._depth_cache:
                del self._depth_cache[oldest_key]
                
        # Add new result
        self._depth_cache[img_hash] = result
        self._cache_keys.append(img_hash)
        
    def get_colored_depth(self, depth_map, colormap=cv2.COLORMAP_INFERNO):
        """Convert depth map to a colored visualization using the specified colormap"""
        # Make sure depth_map is properly normalized for colormap
        if depth_map.min() < 0 or depth_map.max() > 1:
            depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        else:
            depth_map_norm = depth_map
            
        # Convert to uint8 for colormap
        depth_map_uint8 = (depth_map_norm * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, colormap)
        return colored_depth
        
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        # Clean up GPU memory
        if hasattr(self, 'optimize_memory') and self.optimize_memory and \
           hasattr(self, 'device') and 'cuda' in self.device:
            try:
                torch.cuda.empty_cache()
            except:
                pass
