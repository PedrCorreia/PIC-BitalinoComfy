#!/usr/bin/env python3
"""
MiDaS Performance Testing Script

This script tests the optimized MiDaS depth estimation implementation on the same images
used in the benchmark files, with various performance configurations.
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import the optimized MiDaS implementation
from process import Midas, ImgUtils # Updated import
from PIL import Image # Added PIL import

def get_benchmark_images():
    """Find the benchmark images used in the controllnet benchmarks"""
    images = []
    
    # Image from controllnet directory
    controllnet_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "controllnet")
    default_image = os.path.join(controllnet_dir, "image.png")
    if os.path.exists(default_image):
        images.append(("Default benchmark image", default_image))
    
    # Melides image
    melides_image = os.path.join(controllnet_dir, "melides.jpeg")
    if os.path.exists(melides_image):
        images.append(("Melides image", melides_image))
    else:
        # Check in current directory
        melides_current = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "melides.jpeg")
        if os.path.exists(melides_current):
            images.append(("Melides image", melides_current))
    
    # Check for any other PNG or JPEG images in the controllnet directory
    if os.path.exists(controllnet_dir):
        for file in os.listdir(controllnet_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file != "image.png" and "depth" not in file.lower() and "canny" not in file.lower() and "output" not in file.lower():
                img_path = os.path.join(controllnet_dir, file)
                images.append((f"Additional image: {file}", img_path))
    
    # If no images were found, add a fallback
    if not images:
        # Add current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images.append(("Fallback test pattern (generated)", None))
        
    return images

def generate_test_pattern(size=(512, 512)):
    """Generate a test pattern with various shapes for depth testing"""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add a horizon line
    cv2.line(img, (0, size[0]//3), (size[1], size[0]//3), (180, 180, 180), 2)
    
    # Add a gradient from top to bottom
    for y in range(size[0]//3):
        brightness = 240 - int(y * 0.6)
        cv2.line(img, (0, y), (size[1], y), (brightness, brightness, brightness), 1)
    
    # Add some shapes at different "depths"
    # Far mountain
    triangle_pts = np.array([
        [size[1]//4, size[0]//3],
        [size[1]//2, size[0]//6],
        [3*size[1]//4, size[0]//3]
    ], np.int32)
    cv2.fillPoly(img, [triangle_pts], (150, 160, 170))
    
    # Medium distance cube
    cv2.rectangle(img, (size[1]//4, size[0]//2), (size[1]//2, 3*size[0]//4), (100, 120, 140), -1)
    
    # Foreground sphere
    cv2.circle(img, (3*size[1]//4, 2*size[0]//3), size[0]//8, (70, 90, 120), -1)
    
    return img

def test_midas_performance(image_path=None, model_types=None, use_half_precision=None, device=None,
                          optimize_size=True, save_results=True):
    """
    Test MiDaS performance with different configurations
    
    Args:
        image_path: Path to test image (or None for generated pattern)
        model_types: List of model types to test
        use_half_precision: Whether to use half precision (or None for auto)
        device: Device to use (or None for auto)
        optimize_size: Whether to optimize image size
        save_results: Whether to save results
    """
    if model_types is None:
        model_types = ['MiDaS_small']
    
    image_rgb_for_display = None
    image_bgr_for_midas = None

    if image_path is None:
        print("Generating test pattern...")
        image_bgr_for_midas = generate_test_pattern() # generate_test_pattern returns BGR
        image_rgb_for_display = cv2.cvtColor(image_bgr_for_midas, cv2.COLOR_BGR2RGB)
        image_name = "test_pattern"
    else:
        print(f"Loading image: {image_path} using PIL and ImgUtils")
        try:
            pil_img = Image.open(image_path)
            # Ensure image is in RGB format for consistency before ImgUtils conversion
            image_rgb_for_display = ImgUtils.pil_to_numpy(pil_img.convert('RGB')) 
            # Midas.predict currently expects BGR, so convert RGB to BGR for it
            image_bgr_for_midas = cv2.cvtColor(image_rgb_for_display, cv2.COLOR_RGB2BGR)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
        except FileNotFoundError:
            print(f"Error: Could not load image {image_path} - File not found.")
            return
        except Exception as e:
            print(f"Error loading image {image_path} with PIL/ImgUtils: {e}")
            return

    if image_rgb_for_display is None or image_bgr_for_midas is None:
        print("Error: Image preparation failed.")
        return

    print(f"Image shape for display (RGB): {image_rgb_for_display.shape}")
    
    # Prepare figure for results
    num_models = len(model_types)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(4 * (num_models + 1), 6))
    
    # Display input image
    if num_models > 0:  # Multi-plot case
        axes[0].imshow(image_rgb_for_display) # Use RGB image for display
        axes[0].set_title("Input Image")
        axes[0].axis('off')
    else:  # Single plot case
        axes.imshow(image_rgb_for_display) # Use RGB image for display
        axes.set_title("Input Image")
        axes.axis('off')
    
    # Run MiDaS with different configurations
    results = []
    for i, model_type in enumerate(model_types):
        print(f"\nTesting MiDaS with model: {model_type}")
        
        # Initialize with specific configuration
        start_time = time.time()
        midas = Midas(
            model_type=model_type,
            device=device,
            use_half_precision=use_half_precision,
            optimize_memory=True
        )
        
        init_time = time.time() - start_time
        print(f"Initialization time: {init_time:.3f}s")
        
        # Run depth prediction
        start_time = time.time()
        depth_map = midas.predict(image_bgr_for_midas, optimize_size=optimize_size) # Use BGR image for Midas
        inference_time = time.time() - start_time
        
        print(f"Depth prediction time: {inference_time:.3f}s")
        print(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        # Get colored visualization
        depth_colored = midas.get_colored_depth(depth_map)
        
        # Calculate memory usage
        if device and 'cuda' in device:
            try:
                import torch
                mem_allocated = torch.cuda.memory_allocated() / (1024**2)
                mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                print(f"GPU Memory - Allocated: {mem_allocated:.1f}MB, Reserved: {mem_reserved:.1f}MB")
                mem_info = f", Mem: {mem_allocated:.1f}MB"
            except:
                mem_info = ""
        else:
            mem_info = ""
        
        # Store result
        results.append({
            'model_type': model_type,
            'depth_map': depth_map,
            'depth_colored': depth_colored,
            'init_time': init_time,
            'inference_time': inference_time
        })
        
        # Plot result
        if num_models > 0:  # Multi-plot case
            axes[i+1].imshow(depth_colored)
            axes[i+1].set_title(f"{model_type}\n{inference_time:.3f}s{mem_info}")
            axes[i+1].axis('off')
        else:  # Single plot case
            # This case shouldn't happen as we always have at least one model
            pass
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save visualization
    if save_results:
        half_precision_str = "fp16" if use_half_precision else "fp32"
        device_str = device.replace(":", "_") if device else "auto"
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midas_test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined visualization
        output_path = os.path.join(output_dir, f"{image_name}_midas_compare_{device_str}_{half_precision_str}.png")
        plt.savefig(output_path, dpi=150)
        print(f"Saved comparison visualization to {output_path}")
        
        # Save individual depth maps
        for result in results:
            depth_path = os.path.join(output_dir, f"{image_name}_{result['model_type']}_depth.png")
            cv2.imwrite(depth_path, result['depth_colored'])
            print(f"Saved depth map to {depth_path}")
    
    # Show plot
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test MiDaS depth estimation performance")
    parser.add_argument("--image", type=str, help="Path to input image (optional)")
    parser.add_argument("--models", nargs='+', default=["MiDaS_small", "DPT_Hybrid"],
                        help="MiDaS model types to test (default: MiDaS_small and DPT_Hybrid)")
    parser.add_argument("--device", type=str, help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--full", action="store_true", help="Use full precision (FP32)")
    parser.add_argument("--no-resize", action="store_true", help="Disable image resizing optimization")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Find benchmark images
    benchmark_images = get_benchmark_images()
    
    if args.image:
        # Use specified image
        if os.path.exists(args.image):
            selected_images = [("User specified image", args.image)]
        else:
            print(f"Error: Image not found: {args.image}")
            selected_images = benchmark_images
    else:
        selected_images = benchmark_images
    
    # Set half precision flag
    if args.half:
        use_half_precision = True
    elif args.full:
        use_half_precision = False
    else:
        use_half_precision = None  # Auto-detect
    
    # Test each image
    for img_name, img_path in selected_images:
        print(f"\n\n===== Testing {img_name} =====")
        test_midas_performance(
            image_path=img_path,
            model_types=args.models,
            use_half_precision=use_half_precision,
            device=args.device,
            optimize_size=not args.no_resize,
        )
        
    print("\nMiDaS performance testing completed!")