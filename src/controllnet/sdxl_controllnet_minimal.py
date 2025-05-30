"""
Minimal implementation of SDXL ControlNet generation for benchmarking.

This module provides simple utility functions to initialize a ControlNet pipeline once
and then reuse it for multiple generations with different parameters.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2  # Add OpenCV for canny edge detection
import time
import signal

# Define cache directory path
CACHE_DIR = "/media/lugo/data/sd_card_cache"

# IMPORTANT: Set cache paths BEFORE importing huggingface_hub
os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["XDG_CACHE_HOME"] = f"{CACHE_DIR}/cache"
os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

# Create directories if they don't exist
for dir_path in [f"{CACHE_DIR}/huggingface",
                 f"{CACHE_DIR}/cache",
                 f"{CACHE_DIR}/torch",
                 f"{CACHE_DIR}/diffusers"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Add necessary compatibility functions for huggingface_hub
import huggingface_hub

print(f"Using huggingface_hub version: {huggingface_hub.__version__}")
print(f"HF_HOME set to: {os.environ.get('HF_HOME')}")

# Add cached_download for compatibility
if not hasattr(huggingface_hub, 'cached_download'):
    print("Adding cached_download compatibility function")
    def cached_download(*args, **kwargs):
        """Compatibility function for older diffusers versions"""
        print("Redirecting cached_download to hf_hub_download")
        return huggingface_hub.hf_hub_download(*args, **kwargs)
    
    # Add the compatibility function directly to the module
    huggingface_hub.cached_download = cached_download

# Now try to import the required packages
try:
    # Try monkeypatching before importing diffusers
    import importlib
    if 'diffusers.utils' in sys.modules:
        importlib.reload(sys.modules['diffusers.utils'])
    if 'diffusers' in sys.modules:
        importlib.reload(sys.modules['diffusers'])
    
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers.utils import load_image
    print("Successfully imported diffusers modules")
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Make sure you have the required packages installed.")
    sys.exit(1)

# Define a timeout handler for the compile operation
class CompileTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise CompileTimeoutError("Model compilation timed out")

# Try to import optional optimizations
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
    print("CuPy available - using GPU accelerated image processing")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU image processing")

# Global variables to avoid unnecessary reloading
DEFAULT_MODEL_ID = "stabilityai/sdxl-turbo"
DEFAULT_CONTROLNET_ID = "diffusers/controlnet-canny-sdxl-1.0-small"
DEPTH_CONTROLNET_ID = "diffusers/controlnet-depth-sdxl-1.0-small"

def fast_gpu_canny(image_np, low_threshold=50, high_threshold=100):
    """
    GPU-accelerated Canny edge detection using CuPy if available.
    Much faster than OpenCV's CPU implementation for larger images.
    
    Args:
        image_np: Input image as numpy array
        low_threshold: Lower threshold for edge detection
        high_threshold: Higher threshold for edge detection
        
    Returns:
        Edge map as numpy array
    """
    # Set a max timeout to avoid hanging
    start_time = time.time()
    max_processing_time = 30  # seconds
    
    if not CUPY_AVAILABLE:
        # Fall back to CPU OpenCV if CuPy not available
        if len(image_np.shape) == 3:
            # Convert to grayscale for Canny
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    try:
        # Upload image to GPU
        gpu_image = cp.asarray(image_np)
        
        # Convert to grayscale if needed
        if len(gpu_image.shape) == 3:
            gpu_gray = cp.mean(gpu_image, axis=2).astype(cp.uint8)
        else:
            gpu_gray = gpu_image
        
        # Apply Gaussian blur to reduce noise
        gpu_blur = cp_ndimage.gaussian_filter(gpu_gray, sigma=1.0)
        
        # Compute gradients using Sobel operator
        sobelx = cp_ndimage.sobel(gpu_blur, axis=0)
        sobely = cp_ndimage.sobel(gpu_blur, axis=1)
        
        # Compute magnitude and direction
        magnitude = cp.hypot(sobelx, sobely)
        magnitude = magnitude / magnitude.max() * 255
        
        # Apply thresholding
        edges = cp.zeros_like(gpu_gray)
        edges[(magnitude > low_threshold) & (magnitude < high_threshold)] = 255
        
        # Download result from GPU
        result = cp.asnumpy(edges).astype(np.uint8)
        
        print(f"GPU Canny processing time: {time.time() - start_time:.3f}s")
        return result
        
    except Exception as e:
        print(f"Error in GPU Canny processing: {e}, falling back to CPU")
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        return cv2.Canny(gray, low_threshold, high_threshold)

def init_controlnet_pipeline(
    base_model_id=DEFAULT_MODEL_ID,
    controlnet_type="canny",  # "canny", "depth", or "none"
    use_tiny_vae=True,  # Changed default to True for speed
    enable_xformers=True,
    torch_dtype=torch.float16,
    use_compile=False,  # Changed default to False to avoid hanging
    use_slicing=True,  # Added memory optimization
    use_trt=False,  # Added TensorRT option
    compile_timeout=60  # Added timeout for compilation in seconds
):
    """
    Initialize the SDXL ControlNet pipeline for different controlnet types.
    
    Args:
        base_model_id (str): HuggingFace model ID for the base model
        controlnet_type (str): "canny", "depth", or "none"
        use_tiny_vae (bool): Whether to use tiny VAE for faster encoding/decoding
        enable_xformers (bool): Whether to enable xformers for faster attention
        torch_dtype (torch.dtype): Torch dtype for model precision
        use_compile (bool): Whether to compile the UNet for faster inference
        use_slicing (bool): Whether to enable VAE and attention slicing for less memory
        use_trt (bool): Whether to use TensorRT optimization if available
        compile_timeout (int): Maximum time in seconds for model compilation
        
    Returns:
        StableDiffusionXLControlNetPipeline: The initialized pipeline
    """
    if controlnet_type == "none":
        controlnet_id = None
    elif controlnet_type == "depth":
        controlnet_id = DEPTH_CONTROLNET_ID
    else:
        controlnet_id = DEFAULT_CONTROLNET_ID

    print(f"Loading ControlNet type: {controlnet_type} (id: {controlnet_id})")
    start_time = time.time()

    controlnet = None
    if controlnet_id is not None:
        try:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,  # Faster loading
                variant="fp16",  # Use half precision
                local_files_only=False  # Allow downloading if not found locally
            )
            print(f"ControlNet loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error loading ControlNet: {e}")
            print("Trying to load the standard model as fallback...")
            controlnet = ControlNetModel.from_pretrained(
                DEFAULT_CONTROLNET_ID,
                torch_dtype=torch_dtype,
                variant="fp16"
            )

    print(f"Loading base model: {base_model_id}")
    start_time = time.time()
    try:
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch_dtype,
            variant="fp16"  # Use half precision
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Trying without variant specification...")
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch_dtype
        )

    # Use tiny VAE for faster encoding/decoding if requested
    if use_tiny_vae:
        try:
            from diffusers import AutoencoderTiny
            pipeline.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd-diffusers", 
                torch_dtype=torch_dtype
            ).cuda()
            print("Using TinyVAE for faster encoding/decoding")
        except Exception as e:
            print(f"Failed to load TinyVAE: {e}, using standard VAE")
    
    # Move pipeline to GPU
    pipeline = pipeline.to("cuda")
    
    # Memory optimizations
    if use_slicing:
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing(slice_size=1)
        print("Enabled VAE and attention slicing for reduced memory usage")
    
    # Turn off safety checker for speed
    pipeline.safety_checker = None
    
    # Enable xformers if requested
    if enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Xformers enabled for memory-efficient attention")
        except ImportError:
            print("Xformers not available, using standard attention")
    
    # Compile UNet if requested (requires PyTorch 2.0+)
    if use_compile:
        try:
            print("Compiling UNet with torch.compile()...")
            
            # Register the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(compile_timeout)  # Set alarm for compile_timeout seconds
            
            start_time = time.time()
            try:
                # Safer compilation settings that are less likely to hang
                pipeline.unet = torch.compile(
                    pipeline.unet, 
                    mode="reduce-overhead", 
                    fullgraph=False  # Changed to False to prevent hanging
                )
                print(f"UNet compiled in {time.time() - start_time:.2f}s")
            except CompileTimeoutError:
                print(f"UNet compilation timed out after {compile_timeout} seconds, using uncompiled model")
            finally:
                # Reset the alarm
                signal.alarm(0)
        except Exception as e:
            print(f"UNet compilation failed: {e}")
    
    # Use TensorRT if requested and available
    if use_trt:
        try:
            from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
            print("Optimizing with TensorRT...")
            start_time = time.time()
            
            # Register the timeout handler for TensorRT compilation too
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(compile_timeout)  # Set alarm for compile_timeout seconds
            
            try:
                config = CompilationConfig.Default()
                config.enable_xformers = True
                config.enable_cuda_graph = True
                config.enable_triton = True
                config.enable_tensorrt = True
                pipeline = compile(pipeline, config)
                print(f"TensorRT optimization complete in {time.time() - start_time:.2f}s")
            except CompileTimeoutError:
                print(f"TensorRT compilation timed out after {compile_timeout} seconds, using standard model")
            finally:
                # Reset the alarm
                signal.alarm(0)
        except ImportError:
            print("TensorRT optimization not available, using standard pipeline")
    
    # Turn off gradient calculation
    pipeline.set_progress_bar_config(disable=True)
    
    # Return the pipeline
    return pipeline

def generate_with_controlnet(
    pipeline,
    prompt,
    control_image,
    negative_prompt="low quality, blurry, distorted",
    num_steps=2,  # Reduced from 4 to 2 for speed
    guidance_scale=1.0,
    controlnet_conditioning_scale=1.0,
    seed=None,
    output_path=None,
    apply_canny=True,
    canny_low_threshold=50,  # Lower threshold for more edges
    canny_high_threshold=100  # Lower threshold for more edges
):
    """
    Generate an image with ControlNet guidance using a pre-initialized pipeline.
    
    Args:
        pipeline: Pre-initialized StableDiffusionXLControlNetPipeline
        prompt (str): Text prompt for generation
        control_image (PIL.Image): Control image for ControlNet guidance
        negative_prompt (str): Negative prompt for guidance
        num_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for text-to-image generation
        controlnet_conditioning_scale (float): Scale for ControlNet conditioning
        seed (int): Random seed for reproducibility
        output_path (str): Path to save the generated image
        apply_canny (bool): Whether to apply canny edge detection to the control image
        canny_low_threshold (int): Lower threshold for canny edge detection
        canny_high_threshold (int): Higher threshold for canny edge detection
        
    Returns:
        PIL.Image: Generated image
    """
    # Set maximum generation time to prevent hanging
    max_generation_time = 300  # 5 minutes
    
    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None
    
    # Process the control image
    if isinstance(control_image, str) and os.path.exists(control_image):
        control_image = load_image(control_image)
    
    # Ensure control image is right size but use a smaller size (512x512) for speed
    target_size = (512, 512)  # Reduced from 1024x1024 for speed
    if control_image.size != target_size:
        control_image = control_image.resize(target_size, Image.LANCZOS)
    
    # Apply canny edge detection if requested
    if apply_canny:
        print("Generating Canny edge map...")
        start_time = time.time()
        # Convert PIL image to numpy array for processing
        image_np = np.array(control_image)
        
        # Use the GPU-accelerated Canny edge detection if available
        edges = fast_gpu_canny(image_np, canny_low_threshold, canny_high_threshold)
        
        # Convert back to RGB (3 channels) expected by ControlNet
        edges_rgb = np.stack([edges, edges, edges], axis=2)
        
        # Convert back to PIL image
        control_image = Image.fromarray(edges_rgb)
        
        print(f"Canny edge detection completed in {time.time() - start_time:.3f}s")
        
        # Save the canny edge map if requested
        if output_path:
            edge_path = output_path.replace(".png", "_canny.png")
            control_image.save(edge_path)
            print(f"Canny edge map saved to: {edge_path}")
    
    # Generate image
    print(f"Generating image with {num_steps} steps...")
    start_time = time.time()
    
    try:
        # Register the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_generation_time)  # Set alarm for max_generation_time seconds
        
        try:
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            )
        except CompileTimeoutError:
            print(f"Image generation timed out after {max_generation_time} seconds")
            # Create a simple error image
            output = type('obj', (object,), {
                'images': [Image.new("RGB", target_size, (255, 0, 0))]  # Red error image
            })
        finally:
            # Reset the alarm
            signal.alarm(0)
            
        generation_time = time.time() - start_time
        print(f"Image generation completed in {generation_time:.2f}s")
        
        # Get the generated image
        image = output.images[0]
        
        # Save the image if requested
        if output_path:
            image.save(output_path)
            print(f"Image saved to: {output_path}")
        
        return image
        
    except Exception as e:
        print(f"Error during image generation: {e}")
        # Return a simple error image
        error_image = Image.new("RGB", target_size, (255, 0, 0))  # Red error image
        if output_path:
            error_image.save(output_path)
            print(f"Error image saved to: {output_path}")
        return error_image

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline (only once)
    pipeline = init_controlnet_pipeline()
    
    # Example generation
    # Check if a test image exists, otherwise use a blank canvas
    test_image_path = "example_image.jpg"
    if os.path.exists(test_image_path):
        image = load_image(test_image_path)
    else:
        # Create a blank white image
        image = Image.new("RGB", (512, 512), (255, 255, 255))  # Smaller size for speed
        print("Created blank image as example_image.jpg was not found")
    
    # Generate with different scales - faster with 2 steps
    for scale in [0.2, 0.5, 0.8, 1.2]:
        output = generate_with_controlnet(
            pipeline=pipeline,
            prompt="A beautiful landscape",
            control_image=image,
            controlnet_conditioning_scale=scale,
            num_steps=2,  # Use only 2 steps for SDXL Turbo
            output_path=f"controlnet_scale_{scale}.png"
        )
        print(f"Generated image with scale {scale}")