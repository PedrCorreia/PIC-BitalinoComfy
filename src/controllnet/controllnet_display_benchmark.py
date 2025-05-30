import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
import pickle

# Import our minimal ControlNet implementation
from sdxl_controllnet_minimal import generate_with_controlnet
# Import MiDaS depth estimator
sys.path.append(os.path.join(os.path.dirname(__file__), '../im_process'))
from process import Midas


def multi_prompt_benchmark(input_image_path=None, use_checkpoint=True, save_checkpoint=True, controlnet_scales=[0.2, 0.5, 0.8, 1.2]):
    """
    Run a benchmark test for ControlNet with SDXL Turbo using three different prompts and MiDaS depth maps.
    For each prompt, displays:
    - Original image
    - Depth map (from MiDaS small)
    - Generated images at different ControlNet scales
    
    Args:
        input_image_path: Path to input image (if None, uses the image.png from the controllnet folder)
        use_checkpoint: Whether to use checkpoint if available
        save_checkpoint: Whether to save checkpoint after processing
        controlnet_scales: List of scales for ControlNet conditioning
    """
    print("Running Multi-Prompt ControlNet Benchmark with SDXL Turbo and MiDaS depth...")
    
    # Define the three prompts
    prompts = [
        "A magnificent new 4k well preserved ancient Greek temple with marble columns",
        "A modern firehouse with fire trucks and equipment",
        "A surreal templefloating island with upside-down buildings and rainbow waterfalls"
    ]
    
    # Define checkpoint path
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "benchmark_checkpoint.pkl")
    
    # Try to load checkpoint if requested
    if use_checkpoint and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                original_image = checkpoint.get('original_image')
                depth_map_img = checkpoint.get('depth_map_img')
                generated_images = checkpoint.get('generated_images', [])
                elapsed_times = checkpoint.get('elapsed_times', [])
                
                # If we have all the needed data, we can continue from where we left off
                if original_image and depth_map_img and len(generated_images) > 0:
                    print(f"Checkpoint loaded with {len(generated_images)} generated images")
                    
                    # If we don't have all generated images, we need to generate the remaining ones
                    if len(generated_images) < len(prompts) * len(controlnet_scales):
                        print(f"Continuing generation from prompt {len(generated_images) // len(controlnet_scales) + 1}")
                        for idx in range(len(generated_images), len(prompts) * len(controlnet_scales)):
                            prompt_idx = idx // len(controlnet_scales)
                            scale_idx = idx % len(controlnet_scales)
                            
                            # Start timer
                            start_time = time.time()
                            
                            # Generate the image
                            print(f"Generating with prompt {prompt_idx+1}/{len(prompts)}: '{prompts[prompt_idx]}', scale={controlnet_scales[scale_idx]}")
                            generated_image = generate_with_controlnet(
                                prompt=prompts[prompt_idx],
                                control_image=original_image,
                                num_steps=2,  # Fast generation with SDXL Turbo
                                controlnet_conditioning_scale=controlnet_scales[scale_idx],
                                output_path=f"controlnet_output_{prompt_idx}_scale_{controlnet_scales[scale_idx]}.png"
                            )
                            
                            # Calculate time
                            elapsed_time = time.time() - start_time
                            elapsed_times.append(elapsed_time)
                            print(f"Generation completed in {elapsed_time:.2f} seconds")
                            
                            # Save generated image
                            generated_images.append(generated_image)
                    
                    # Save updated checkpoint
                    if save_checkpoint:
                        checkpoint = {
                            'original_image': original_image,
                            'depth_map_img': depth_map_img,
                            'generated_images': generated_images,
                            'elapsed_times': elapsed_times
                        }
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump(checkpoint, f)
                    
                    # Skip to plotting
                    create_plot(original_image, depth_map_img, generated_images, elapsed_times, controlnet_scales)
                    return original_image, depth_map_img, generated_images
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Processing from scratch...")
    
    # Create or load input image if checkpoint wasn't used
    if input_image_path and os.path.exists(input_image_path):
        print(f"Loading input image from: {input_image_path}")
        original_image = Image.open(input_image_path).convert("RGB")
    else:
        # Use the default image.png from the controllnet folder
        default_image_path = os.path.join(os.path.dirname(__file__), "image.png")
        if os.path.exists(default_image_path):
            print(f"Loading default image from: {default_image_path}")
            original_image = Image.open(default_image_path).convert("RGB")
        else:
            print("Default image not found. Creating a test grid image")
            # Create a simple grid test image as fallback
            grid_size = 512
            grid_spacing = 64
            original_image = Image.new('RGB', (grid_size, grid_size), color=(255, 255, 255))
            img_array = np.array(original_image)
            
            # Draw grid lines
            for i in range(0, grid_size, grid_spacing):
                img_array[i:i+2, :] = [0, 0, 0]
                img_array[:, i:i+2] = [0, 0, 0]
                
            # Draw a simple shape
            center = grid_size // 2
            radius = grid_size // 4
            cv2.circle(img_array, (center, center), radius, (100, 150, 200), -1)
            cv2.rectangle(img_array, (100, 100), (200, 200), (200, 100, 100), -1)
            
            original_image = Image.fromarray(img_array)
    # Resize if needed
    width, height = original_image.size
    if width != 512 or height != 512:
        original_image = original_image.resize((512, 512), Image.LANCZOS)

    # Generate MiDaS depth map
    print("Generating MiDaS depth map...")
    midas = Midas()
    img_array = np.array(original_image)
    depth_map = midas.predict(img_array)
    # Normalize and convert to 8-bit for visualization and ControlNet
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map_uint8 = (depth_map_norm * 255).astype(np.uint8)
    depth_map_img = Image.fromarray(depth_map_uint8)

    # Save depth map for reference
    depth_path = "controlnet_input_depth.png"
    depth_map_img.save(depth_path)
    print(f"Saved input depth map to {depth_path}")

    # Process each prompt and each scale
    generated_images = []
    elapsed_times = []
    for idx, prompt in enumerate(prompts):
        for scale in controlnet_scales:
            start_time = time.time()
            print(f"Generating with prompt {idx+1}/{len(prompts)}: '{prompt}', scale={scale}")
            generated_image = generate_with_controlnet(
                prompt=prompt,
                control_image=depth_map_img,
                num_steps=2,  # Fast generation with SDXL Turbo
                controlnet_conditioning_scale=scale,
                output_path=f"controlnet_output_{idx}_scale_{scale}.png"
            )
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)
            print(f"Generation completed in {elapsed_time:.2f} seconds")
            generated_images.append((prompt, scale, generated_image))
            # Save checkpoint after each generation
            if save_checkpoint:
                checkpoint = {
                    'original_image': original_image,
                    'depth_map_img': depth_map_img,
                    'generated_images': generated_images,
                    'elapsed_times': elapsed_times
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
    # Create the plot
    create_plot(original_image, depth_map_img, generated_images, elapsed_times, controlnet_scales)
    return original_image, depth_map_img, generated_images

def create_plot(original_image, depth_map_img, generated_images, elapsed_times, controlnet_scales):
    """Create and save the benchmark visualization plot for all prompts and scales"""
    n_prompts = len(set([g[0] for g in generated_images]))
    n_scales = len(controlnet_scales)
    plt.figure(figsize=(4 + n_scales * 4, n_prompts * 4))
    for idx, (prompt, scale, generated_image) in enumerate(generated_images):
        prompt_idx = list(set([g[0] for g in generated_images])).index(prompt)
        scale_idx = controlnet_scales.index(scale)
        # Original image (first column)
        if scale_idx == 0:
            plt.subplot(n_prompts, n_scales + 2, prompt_idx * (n_scales + 2) + 1)
            plt.imshow(np.array(original_image))
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(n_prompts, n_scales + 2, prompt_idx * (n_scales + 2) + 2)
            plt.imshow(np.array(depth_map_img), cmap='gray')
            plt.title("MiDaS Depth Map")
            plt.axis('off')
        # Generated image (subsequent columns)
        plt.subplot(n_prompts, n_scales + 2, prompt_idx * (n_scales + 2) + 3 + scale_idx)
        plt.imshow(np.array(generated_image))
        plt.title(f"Scale: {scale}")
        plt.axis('off')
    avg_time = sum(elapsed_times) / len(elapsed_times)
    plt.suptitle(f"ControlNet SDXL Turbo with MiDaS Depth\nAvg. generation time: {avg_time:.2f}s", fontsize=16)
    plt.tight_layout()
    result_path = "controllnet_benchmark_multi_prompt.png"
    plt.savefig(result_path)
    print(f"Benchmark results saved to {result_path}")
    try:
        plt.show()
    except:
        print("Could not display the plot interactively, but the image was saved")

if __name__ == "__main__":
    # Check if an image path was provided as command-line argument
    input_image_path = None
    use_checkpoint = True
    
    for i, arg in enumerate(sys.argv):
        if i == 1:
            input_image_path = arg
            print(f"Using input image: {input_image_path}")
        elif arg == "--no-checkpoint":
            use_checkpoint = False
            print("Checkpoint usage disabled")
    
    # Run the multi-prompt benchmark
    original, depth, generated = multi_prompt_benchmark(
        input_image_path=input_image_path,
        use_checkpoint=use_checkpoint
    )
    
    print("Multi-prompt benchmark completed!")