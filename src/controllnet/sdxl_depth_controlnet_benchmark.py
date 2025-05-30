"""
Benchmark for Depth ControlNet with SDXL Turbo

Tests different conditioning scales to find the optimal balance
between following depth guidance and creative freedom.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

# Import depth ControlNet functions
from sdxl_depth_controlnet_downloader import (
    init_depth_controlnet_pipeline, 
    generate_depth_map,
    generate_with_depth_controlnet,
    load_image,
    DEPTH_CONTROLNET_ID,
    SDXL_TURBO_MODEL_ID
)

def run_depth_controlnet_scale_benchmark(
    input_image_path=None, 
    prompt="A beautiful mountain temple with restored ruins and a cosmic galactic sky at night", 
    scales=[0.2, 0.5, 0.8, 1.2],
    num_steps=2
):
    """
    Benchmark Depth ControlNet with SDXL Turbo using different conditioning scales for the same prompt
    
    Args:
        input_image_path: Path to input image
        prompt: Text prompt to use for all generations
        scales: List of ControlNet conditioning scales to test
        num_steps: Number of inference steps (typically 2 for SDXL Turbo)
    """
    print(f"Running Depth ControlNet Scale Benchmark with prompt: '{prompt}'")
    
    # Initialize the pipeline and depth estimator
    print("Initializing Depth ControlNet pipeline...")
    pipeline, depth_estimator = init_depth_controlnet_pipeline()
    
    # Create or load input image
    if input_image_path and os.path.exists(input_image_path):
        print(f"Loading input image from: {input_image_path}")
        input_image = Image.open(input_image_path).convert("RGB")
    else:
        # Try to use melides.jpeg
        if os.path.exists("melides.jpeg"):
            print("Using melides.jpeg as input image")
            input_image = Image.open("melides.jpeg").convert("RGB")
        else:
            print("Creating a simple test image")
            # Create a simple test image with basic shapes
            size = 512
            input_image = Image.new('RGB', (size, size), color=(240, 240, 240))
            img_array = np.array(input_image)
            
            # Draw a simple landscape-like scene
            # Sky
            cv2.rectangle(img_array, (0, 0), (size, size//2), (135, 206, 235), -1)
            # Ground
            cv2.rectangle(img_array, (0, size//2), (size, size), (34, 139, 34), -1)
            # Mountains
            pts = np.array([[size//4, size//2], [size//2, size//5], [3*size//4, size//2]], np.int32)
            cv2.fillPoly(img_array, [pts], (120, 120, 120))
            # Sun
            cv2.circle(img_array, (size//4, size//4), size//10, (255, 255, 0), -1)
            
            input_image = Image.fromarray(img_array)
            input_image_path = "synthetic_landscape.png"
            input_image.save(input_image_path)
            print(f"Created and saved synthetic test image to {input_image_path}")
    
    # Resize if needed
    if input_image.size != (1024, 1024):
        input_image = input_image.resize((1024, 1024), Image.LANCZOS)
    
    # Generate depth map once (to avoid redundant processing)
    print("Generating depth map...")
    start_time = time.time()
    depth_map = generate_depth_map(input_image, depth_estimator)
    depth_time = time.time() - start_time
    print(f"Depth map generation completed in {depth_time:.2f} seconds")
    
    # Save the depth map for reference
    depth_path = "controlnet_input_depth.png"
    depth_map.save(depth_path)
    print(f"Saved input depth map to {depth_path}")
    
    # Generate images with different scales
    generated_images = []
    elapsed_times = []
    
    # Set up the matplotlib figure
    n_cols = len(scales) + 2
    plt.figure(figsize=(n_cols*4, 6))
    
    # Plot original image
    plt.subplot(1, n_cols, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot depth map
    plt.subplot(1, n_cols, 2)
    plt.imshow(depth_map)
    plt.title("Depth Map")
    plt.axis('off')
    
    print(f"Using {num_steps} inference steps for SDXL Turbo")
    
    # Generate images with different ControlNet scales
    for i, scale in enumerate(scales):
        # Start timer
        start_time = time.time()
        
        # Generate the image with current scale
        print(f"Generating with depth ControlNet scale: {scale}")
        output_path = f"controlnet_depth_scale_{scale}.png"
        
        # Skip the depth map generation as we've already done it
        output = pipeline(
            prompt=prompt,
            negative_prompt="low quality, blurry",
            image=depth_map,  # Use pre-generated depth map
            num_inference_steps=num_steps,
            guidance_scale=1.0,  # Default for SDXL Turbo
            controlnet_conditioning_scale=scale,
            generator=None
        )
        
        generated_image = output.images[0]
        generated_image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        # Calculate time
        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)
        print(f"Generation completed in {elapsed_time:.2f} seconds")
        
        # Plot the result
        plt.subplot(1, n_cols, i + 3)
        plt.imshow(generated_image)
        plt.title(f"Scale: {scale}\n{elapsed_time:.2f}s")
        plt.axis('off')
        
        # Save generated image
        generated_images.append(generated_image)
    
    # Set overall title and save the result
    avg_time = sum(elapsed_times) / len(elapsed_times)
    plt.suptitle(f"Depth ControlNet SDXL Turbo with Different Scales\nPrompt: '{prompt}'\n{num_steps} steps, Avg. time: {avg_time:.2f}s", fontsize=14)
    plt.tight_layout()
    
    # Save the result
    result_path = "depth_controlnet_scale_benchmark.png"
    plt.savefig(result_path, dpi=150)
    print(f"Benchmark results saved to {result_path}")
    
    try:
        plt.show()
    except:
        print("Could not display the plot interactively, but the image was saved")
    
    return input_image, depth_map, generated_images

if __name__ == "__main__":
    # Get input path and prompt from command line if provided
    input_image_path = "/media/lugo/data/ComfyUI/custom_nodes/PIC-BitalinoComfy/src/controllnet/image.png"
    
    prompt = "A photorealistic scene of ancient Greek ruins with columns, beautiful sky at sunset"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
        print(f"Using input image: {input_image_path}")
        
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    
    # Define scales to test
    scales = [0.2, 0.5, 0.8, 1.2]
    
    # For testing a specific image
    if input_image_path is None and os.path.exists("melides.jpeg"):
        input_image_path = "melides.jpeg"
    
    # Run the benchmark
    run_depth_controlnet_scale_benchmark(
        input_image_path=input_image_path,
        prompt=prompt,
        scales=scales,
        num_steps=2  # Use 2 steps for SDXL Turbo for speed
    )
    
    print("Benchmark completed!")