import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
import signal

# Import MiDaS properly from the test implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from sdxl_controllnet_minimal import generate_with_controlnet, init_controlnet_pipeline

# Define timeout handler for the entire benchmark
class BenchmarkTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise BenchmarkTimeoutError("Benchmark timed out")

# Use the MiDaS implementation from MiDaS test, not the webcam file
def generate_midas_depth_map(image_path, output_path=None):
    """
    Generate a depth map using MiDaS small model as used in MiDaS tests
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the depth map
        
    Returns:
        depth_map: Image object containing the depth map
    """
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Initialize MiDaS small model as in the MiDaS tests
    print("Initializing MiDaS depth estimator with small model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "midas", "weights", "midas_v21_small.pt")
    
    # Check if model exists, if not download it
    if not os.path.exists(model_path):
        print("Model file not found. Please ensure the MiDaS small model is available at:", model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Could add model download logic here if needed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MidasNet(model_path, non_negative=True)
    model.to(device)
    model.eval()
    
    # Setup preprocessing transforms as used in MiDaS tests
    transforms = torch.nn.Sequential(
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    )
    
    # Generate depth map
    print("Generating depth map using MiDaS small model...")
    start_time = time.time()
    
    # Convert BGR to RGB as MiDaS expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    
    # Apply preprocessing transforms
    input_tensor = transforms({"image": image_rgb})["image"]
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Compute depth with MiDaS small model
    with torch.no_grad():
        prediction = model.forward(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f"Depth map generation completed in {elapsed_time:.2f} seconds")
    
    # Normalize and convert to 8-bit for visualization
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create a 3-channel depth map as required by ControlNet
    depth_3channel = np.stack([depth_map_norm, depth_map_norm, depth_map_norm], axis=2)
    
    # Save the depth map if output path is provided
    if output_path:
        print(f"Saving depth map to {output_path}")
        cv2.imwrite(output_path, depth_3channel)
    
    # Create PIL image from numpy array
    depth_image = Image.fromarray(depth_3channel)
    
    return depth_image

def run_depth_controlnet_benchmark(input_image_path=None, prompt="A beautiful mountain landscape with a river", scales=[0.2, 0.5, 0.8, 1.2], max_benchmark_time=900):
    """
    Benchmark ControlNet with SDXL Turbo using depth maps with different conditioning scales
    
    Args:
        input_image_path: Path to input image
        prompt: Text prompt to use for all generations
        scales: List of ControlNet conditioning scales to test
        max_benchmark_time: Maximum time in seconds for the entire benchmark
    """
    # Set a timeout for the entire benchmark process
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_benchmark_time)  # e.g., 15 minutes max
    
    try:
        start_time = time.time()
        print(f"Running Depth ControlNet Benchmark with prompt: '{prompt}'")
        
        # Create or load input image
        if input_image_path and os.path.exists(input_image_path):
            print(f"Loading input image from: {input_image_path}")
            original_image = Image.open(input_image_path).convert("RGB")
        else:
            # Use a default image
            default_image_path = os.path.join(os.path.dirname(__file__), "image.png")
            if os.path.exists(default_image_path):
                print(f"Loading default image from: {default_image_path}")
                original_image = Image.open(default_image_path).convert("RGB")
            else:
                print("Creating a simple test image")
                # Create a simple test image with basic shapes
                size = 512
                original_image = Image.new('RGB', (size, size), color=(240, 240, 240))
                img_array = np.array(original_image)
                
                # Draw a simple shape
                cv2.circle(img_array, (size//2, size//2), size//4, (100, 150, 200), -1)
                cv2.rectangle(img_array, (100, 100), (200, 200), (200, 100, 100), -1)
                original_image = Image.fromarray(img_array)
        
        # Resize if needed - use smaller size (512x512) for speed
        if original_image.size != (512, 512):
            original_image = original_image.resize((512, 512), Image.LANCZOS)
            
        # Save the original image for reference
        original_image_path = os.path.join(os.path.dirname(__file__), "depth_benchmark_input.png")
        original_image.save(original_image_path)
            
        # Create depth map using your MiDaS implementation
        print("Generating depth map with MiDaS small model...")
        start_time_depth = time.time()
        
        # Use MiDaS small implementation
        depth_path = "controlnet_input_depth.png"
        depth_image = generate_midas_depth_map(original_image_path, depth_path)
        
        print(f"Depth map generation completed in {time.time() - start_time_depth:.3f}s")
        
        # Initialize the pipeline with the lighter depth ControlNet model
        print("Initializing ControlNet pipeline with lighter model...")
        try:
            # Set a 3 minute timeout for model initialization
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)
            
            # Always use the small depth model
            pipeline = init_controlnet_pipeline(
                # Use lighter depth ControlNet model
                controlnet_id="diffusers/controlnet-depth-sdxl-1.0-small",
                use_tiny_vae=True,
                use_compile=False,  # Disable compilation to avoid hanging
                use_slicing=True
            )
            
            # Reset the alarm for the overall benchmark
            signal.alarm(max_benchmark_time - int(time.time() - start_time))
        except BenchmarkTimeoutError:
            print("Pipeline initialization timed out, falling back to standard model")
            # Try again with another lighter model
            try:
                pipeline = init_controlnet_pipeline(
                    # Still use the small model as fallback
                    controlnet_id="diffusers/controlnet-depth-sdxl-1.0-small",
                    use_tiny_vae=False,
                    use_compile=False,
                    use_slicing=False
                )
            except Exception as e:
                print(f"Failed to initialize pipeline: {e}")
                # Exit gracefully
                print("Benchmark failed due to initialization errors")
                return None, None, []
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            print("Trying fallback initialization...")
            try:
                # Use another lightweight model as fallback
                pipeline = init_controlnet_pipeline(
                    controlnet_id="diffusers/controlnet-depth-sdxl-1.0-small",
                    use_tiny_vae=False,
                    use_compile=False,
                    use_slicing=False
                )
            except Exception as e2:
                print(f"Fallback initialization failed: {e2}")
                # Exit gracefully
                print("Benchmark failed due to initialization errors")
                return None, None, []
        
        # Generate images with different scales
        generated_images = []
        elapsed_times = []
        
        # Set up the matplotlib figure
        n_cols = len(scales) + 2
        plt.figure(figsize=(n_cols*4, 6))
        
        # Plot original image
        plt.subplot(1, n_cols, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot depth map
        plt.subplot(1, n_cols, 2)
        plt.imshow(depth_image)
        plt.title("MiDaS Small Depth Map")
        plt.axis('off')
        
        # Use only 2 steps for faster generation
        num_steps = 2
        print(f"Using {num_steps} inference steps for SDXL Turbo")
        
        # Generate images with different ControlNet scales
        for i, scale in enumerate(scales):
            # Start timer
            start_time_gen = time.time()
            
            # Generate the image with current scale
            print(f"Generating with ControlNet scale: {scale}")
            output_path = f"controlnet_depth_scale_{scale}.png"
            
            try:
                # Set a 2 minute timeout for each generation
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                
                # Generate image with the depth map
                generated_image = generate_with_controlnet(
                    pipeline=pipeline,
                    prompt=prompt,
                    control_image=depth_image,
                    num_steps=num_steps,
                    controlnet_conditioning_scale=scale,
                    output_path=output_path,
                    apply_canny=False  # No need to apply Canny since we're using depth
                )
                
                # Reset the alarm for the overall benchmark
                signal.alarm(max_benchmark_time - int(time.time() - start_time))
            except BenchmarkTimeoutError:
                print(f"Generation with scale {scale} timed out")
                # Create a red error image
                generated_image = Image.new("RGB", (512, 512), (255, 0, 0))
                if output_path:
                    generated_image.save(output_path)
            except Exception as e:
                print(f"Error generating image with scale {scale}: {e}")
                # Create a red error image
                generated_image = Image.new("RGB", (512, 512), (255, 0, 0))
                if output_path:
                    generated_image.save(output_path)
            
            # Calculate time
            elapsed_time = time.time() - start_time_gen
            elapsed_times.append(elapsed_time)
            print(f"Generation completed in {elapsed_time:.2f} seconds")
            
            # Plot the result
            plt.subplot(1, n_cols, i + 3)
            plt.imshow(np.array(generated_image))
            plt.title(f"Scale: {scale}\n{elapsed_time:.2f}s")
            plt.axis('off')
            
            # Save generated image
            generated_images.append(generated_image)
        
        # Set overall title and save the result
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            plt.suptitle(f"MiDaS Small Depth ControlNet SDXL Turbo Light with Different Scales\nPrompt: '{prompt}'\n{num_steps} steps, Avg. time: {avg_time:.2f}s", fontsize=14)
        else:
            plt.suptitle(f"MiDaS Small Depth ControlNet SDXL Turbo Light with Different Scales\nPrompt: '{prompt}'\nBenchmark failed", fontsize=14)
            
        plt.tight_layout()
        
        # Save the result
        result_path = "depth_controlnet_scale_benchmark.png"
        plt.savefig(result_path, dpi=150)
        print(f"Benchmark results saved to {result_path}")
        
        try:
            plt.savefig(result_path)
            print("Saved benchmark figure")
            # Don't use plt.show() to avoid hanging in headless environments
        except Exception as e:
            print(f"Could not save plot: {e}")
        
        # Reset the alarm
        signal.alarm(0)
        
        total_time = time.time() - start_time
        print(f"Total benchmark time: {total_time:.2f}s")
        
        return original_image, depth_image, generated_images
        
    except BenchmarkTimeoutError:
        print(f"Benchmark timed out after {max_benchmark_time} seconds")
        # Reset the alarm
        signal.alarm(0)
        return None, None, []
    except Exception as e:
        print(f"Error during benchmark: {e}")
        # Reset the alarm
        signal.alarm(0)
        return None, None, []

if __name__ == "__main__":
    # Get input path and prompt from command line if provided
    input_image_path = None
    prompt = "A 4k re-imagined restored greek beautiful temple with mountain landscape with a river with greek mythology creatures"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
        print(f"Using input image: {input_image_path}")
        
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    
    # Define scales to test - can be changed here
    scales = [0.2, 0.5, 0.8, 1.2]
    
    # For testing specific image.png
    if input_image_path is None:
        input_image_path = "/media/lugo/data/ComfyUI/custom_nodes/PIC-BitalinoComfy/src/controllnet/image.png"
        # If the path doesn't exist, don't specify it
        if not os.path.exists(input_image_path):
            input_image_path = None
    
    # Run the benchmark with a timeout of 15 minutes
    run_depth_controlnet_benchmark(
        input_image_path=input_image_path,
        prompt=prompt,
        scales=scales,
        max_benchmark_time=900  # 15 minutes max
    )
    
    print("MiDaS Depth ControlNet benchmark completed!")