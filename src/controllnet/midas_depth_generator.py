#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiDaS Depth Map Generator

This script loads an image file and generates a depth map using MiDaS depth estimation.
The depth map is saved as an image file and can be used with ControlNet.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import torch
import importlib.util

# Check for required dependencies
required_packages = ['timm', 'torch', 'cv2', 'numpy', 'matplotlib']
missing_packages = []

for package in required_packages:
    if package == 'cv2':
        # OpenCV is imported as cv2 but the package name is opencv-python
        package_name = 'opencv-python'
    else:
        package_name = package
        
    try:
        if package == 'cv2':
            # We already imported cv2 above, so just check if it's available
            if cv2 is None:
                missing_packages.append(package_name)
        else:
            # For other packages, try importing them
            importlib.import_module(package)
    except ImportError:
        missing_packages.append(package_name)

# If there are missing packages, prompt to install them
if missing_packages:
    print(f"Missing required packages: {', '.join(missing_packages)}")
    print("Please install them using:")
    print(f"pip install {' '.join(missing_packages)}")
    answer = input("Do you want to install them now? (y/n): ")
    if answer.lower() == 'y':
        import subprocess
        try:
            print(f"Installing {' '.join(missing_packages)}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Installation completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            print("Please install them manually and try again.")
            sys.exit(1)
    else:
        print("Please install the required packages manually and try again.")
        sys.exit(1)

# Add parent directory to path to import from src.im_process
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import MiDaS from process.py
try:
    from src.im_process.process import Midas
except ImportError as e:
    print(f"Error importing Midas: {e}")
    print("Make sure the path to src.im_process.process is correct.")
    sys.exit(1)

def generate_depth_map(image_path, output_path=None, model_type='MiDaS_small', display=True):
    """
    Generate a depth map from an input image using MiDaS.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the depth map (if None, will use image_path_depth.png)
        model_type: MiDaS model type to use ('MiDaS_small' is faster, 'DPT_Large' is more accurate)
        display: Whether to display the depth map
        
    Returns:
        depth_map: Numpy array containing the depth map
    """
    # Start timer to measure performance
    start_time = time.time()
    
    # Load image
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Initialize MiDaS
    print(f"Initializing MiDaS with model type: {model_type}")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        midas = Midas(model_type=model_type)
        
        # Generate depth map
        print("Generating depth map...")
        depth_map = midas.predict(image, optimize_size=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Depth map generation completed in {elapsed_time:.2f} seconds")
        
        # Normalize depth map to [0, 255] for visualization
        depth_map_norm = (depth_map * 255).astype(np.uint8)
        
        # Create color map for better visualization
        depth_colored = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_INFERNO)
        
        # Set output path if not provided
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_depth.png"
            color_output_path = f"{base_name}_depth_colored.png"
        else:
            base_name = os.path.splitext(output_path)[0]
            color_output_path = f"{base_name}_colored.png"
        
        # Save depth map
        print(f"Saving depth map to {output_path}")
        cv2.imwrite(output_path, depth_map_norm)
        
        # Save colored depth map
        print(f"Saving colored depth map to {color_output_path}")
        cv2.imwrite(color_output_path, depth_colored)
        
        # Display results if requested
        if display:
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Depth map (grayscale)
            plt.subplot(1, 3, 2)
            plt.imshow(depth_map, cmap='gray')
            plt.title('Depth Map (Grayscale)')
            plt.axis('off')
            
            # Depth map (colored)
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            plt.title('Depth Map (Colored)')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save the comparison figure
            comparison_path = f"{base_name}_comparison.png"
            print(f"Saving comparison image to {comparison_path}")
            plt.savefig(comparison_path, dpi=150)
            
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plots: {e}")
        
        return depth_map
        
    except RuntimeError as e:
        if "No module named 'timm'" in str(e):
            print("Error: Missing timm package which is required by MiDaS.")
            print("Please install it using: pip install timm")
            sys.exit(1)
        else:
            print(f"Error during depth map generation: {e}")
            raise
    except Exception as e:
        print(f"Error during depth map generation: {e}")
        raise

def prepare_controlnet_depth(image_path, output_path=None):
    """
    Prepare a depth map specifically for use with ControlNet.
    This function ensures the depth map is properly formatted for ControlNet.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the ControlNet-ready depth map
        
    Returns:
        controlnet_depth: Numpy array containing the ControlNet-ready depth map
    """
    try:
        # Generate the depth map
        depth_map = generate_depth_map(image_path, display=False)
        
        # Set output path if not provided
        if output_path is None:
            output_path = "controlnet_input_depth.png"
        
        # Create a 3-channel depth map as required by ControlNet
        # ControlNet expects RGB image with the same depth values in each channel
        depth_3channel = np.stack([depth_map, depth_map, depth_map], axis=2)
        depth_3channel = (depth_3channel * 255).astype(np.uint8)
        
        # Save the ControlNet-ready depth map
        print(f"Saving ControlNet-ready depth map to {output_path}")
        cv2.imwrite(output_path, depth_3channel)
        
        return depth_3channel
        
    except Exception as e:
        print(f"Error preparing ControlNet depth map: {e}")
        raise

def test_torch_hub_access():
    """Test if torch.hub can access online repositories."""
    print("Testing torch.hub connectivity...")
    try:
        # Test connectivity by trying to list available models
        torch.hub.list('pytorch/vision')
        print("torch.hub connectivity success!")
        return True
    except Exception as e:
        print(f"torch.hub connectivity error: {e}")
        print("MiDaS may not be able to download the required model files.")
        return False

if __name__ == "__main__":
    # Check torch.hub access
    test_torch_hub_access()
    
    # Get image path from command line if provided, otherwise use default
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
    else:
        # Default to image.png in the controllnet directory
        input_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
    
    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        sys.exit(1)
        
    print(f"Using input image: {input_image_path}")
    
    try:
        # Generate depth map for visualization
        depth_map = generate_depth_map(
            input_image_path,
            model_type='MiDaS_small'  # Use small model for speed
        )
        
        # Generate ControlNet-ready depth map
        controlnet_depth = prepare_controlnet_depth(input_image_path)
        
        print("Depth map generation complete!")
    except Exception as e:
        print(f"Failed to generate depth map: {e}")
        sys.exit(1)