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

# Import our minimal ControlNet implementation
from sdxl_controllnet_minimal import generate_with_controlnet, init_controlnet_pipeline, fast_gpu_canny

# Define timeout handler for the entire benchmark
class BenchmarkTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise BenchmarkTimeoutError("Benchmark timed out")

def run_controlnet_scale_benchmark(input_image_path=None, prompt="A beautiful mountain landscape with a river", scales=[0.2, 0.5, 0.8, 1.2], max_benchmark_time=900):
    """
    Benchmark ControlNet with SDXL Turbo using different conditioning scales for the same prompt
    
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
        print(f"Running ControlNet Scale Benchmark with prompt: '{prompt}'")
        
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
            
        # Create Canny edge map once (faster, simpler version)
        print("Generating Canny edge map...")
        start_time_canny = time.time()
        img_array = np.array(original_image)
        
        # Set a timeout just for the Canny edge detection
        try:
            # Set a 30 second timeout for just the edge detection
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            # Use our optimized GPU Canny if available
            canny_array = fast_gpu_canny(img_array, 50, 130)
            
            # Reset the alarm
            signal.alarm(0)
        except BenchmarkTimeoutError:
            print("Canny edge detection timed out, falling back to basic OpenCV")
            # Fall back to basic CPU implementation
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            canny_array = cv2.Canny(gray, 50, 100)
        except Exception as e:
            print(f"Error in Canny edge detection: {e}")
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            canny_array = cv2.Canny(gray, 50, 100)
        finally:
            # Reset the alarm for the overall benchmark
            signal.alarm(max_benchmark_time - int(time.time() - start_time))
            
        # Convert to 3-channel image expected by the ControlNet
        canny_3channel = np.stack([canny_array, canny_array, canny_array], axis=2)
        canny_image = Image.fromarray(canny_3channel)
        print(f"Edge map generation completed in {time.time() - start_time_canny:.3f}s")
        
        # Save the Canny edge map for reference
        canny_path = "controlnet_input_canny.png"
        canny_image.save(canny_path)
        print(f"Saved input Canny edge map to {canny_path}")
        
        # Initialize the pipeline once with the small, fast model
        print("Initializing ControlNet pipeline...")
        try:
            # Set a 3 minute timeout for model initialization
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)
            
            pipeline = init_controlnet_pipeline(
                # Use smaller, faster model
                controlnet_type="canny",
                use_tiny_vae=True,
                use_compile=False,  # Disable compilation to avoid hanging
                use_slicing=True
            )
            
            # Reset the alarm for the overall benchmark
            signal.alarm(max_benchmark_time - int(time.time() - start_time))
        except BenchmarkTimeoutError:
            print("Pipeline initialization timed out, falling back to standard model")
            # Try again without any optimizations that might hang
            try:
                pipeline = init_controlnet_pipeline(
                    controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
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
                pipeline = init_controlnet_pipeline(
                    controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
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
        
        # Plot Canny edge map
        plt.subplot(1, n_cols, 2)
        plt.imshow(canny_image)
        plt.title("Canny Edge Map")
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
            output_path = f"controlnet_scale_{scale}.png"
            
            try:
                # Set a 2 minute timeout for each generation
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                
                # Use the pre-computed canny image - directly pass the canny image and disable apply_canny
                generated_image = generate_with_controlnet(
                    pipeline=pipeline,
                    prompt=prompt,
                    control_image=canny_image,  # Use the pre-computed canny image
                    num_steps=num_steps,  # Using only 2 steps for speed
                    controlnet_conditioning_scale=scale,
                    output_path=output_path,
                    apply_canny=False  # Skip redundant canny processing
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
            plt.title(f"Scale: {scale} | {elapsed_time  :.2f}s")  # <-- scale and time on same line
            plt.axis('off')
            
            # Save generated image
            generated_images.append(generated_image)
        
        # Set overall title and save the result
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            plt.suptitle(f"ControlNet SDXL Turbo with Different Scales\nPrompt: '{prompt}'\n{num_steps} steps, Avg. time: {avg_time:.2f}s", fontsize=14)
        else:
            plt.suptitle(f"ControlNet SDXL Turbo with Different Scales\nPrompt: '{prompt}'\nBenchmark failed", fontsize=14)
            
        plt.tight_layout()
        
        # Save the result
        result_path = "controllnet_scale_benchmark.png"
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
        
        return original_image, canny_image, generated_images
        
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
    prompt = "A 4k re-imagined restored greek beautiful temple with mountain landscape with a river with greek mitology creatures with galactic night sky "
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
        print(f"Using input image: {input_image_path}")
        
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    
    # Define scales to test - can be changed here
    scales = [0.2, 0.5, 0.8, 1.0]
    
    # For testing specific image.png
    if input_image_path is None:
        input_image_path = "/media/lugo/data/ComfyUI/custom_nodes/PIC-BitalinoComfy/src/controllnet/image.png"
        # If the path doesn't exist, don't specify it
        if not os.path.exists(input_image_path):
            input_image_path = None
    
    # Run the benchmark with a timeout of 15 minutes
    run_controlnet_scale_benchmark(
        input_image_path=input_image_path,
        prompt=prompt,
        scales=scales,
        max_benchmark_time=900  # 15 minutes max
    )
    
    print("Benchmark completed!")