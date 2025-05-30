#!/usr/bin/env python3
"""
Fix Torch Environment Script

This script safely uninstalls timm and repairs your PyTorch installation
without affecting torchvision or torchaudio packages.
"""

import os
import sys
import subprocess
import pkg_resources
import shutil
import tempfile
import time

def run_command(command, description=None):
    """Run a command and return its output"""
    if description:
        print(f"\n{description}...")
    
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Command returned non-zero exit code {result.returncode}")
        print(f"Error output: {result.stderr}")
    
    return result.stdout.strip()

def backup_file(filepath):
    """Create a backup of a file if it exists"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.bak-{int(time.time())}"
        print(f"Creating backup: {filepath} → {backup_path}")
        shutil.copy2(filepath, backup_path)
        return backup_path
    return None

def fix_environment():
    """Fix the conda/pip environment by uninstalling timm and repairing torch"""
    python_exe = sys.executable
    pip_exe = f"{python_exe} -m pip"
    
    print("="*80)
    print("TORCH ENVIRONMENT REPAIR TOOL")
    print("="*80)
    
    # 1. Check current environment
    print("\nChecking current environment...")
    
    # Get conda environment name
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    conda_env = os.path.basename(conda_prefix) if conda_prefix else 'Unknown'
    print(f"Active conda environment: {conda_env}")
    
    # Get Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # List key packages
    print("\nChecking for installed packages:")
    
    # Check torch specifically
    has_torch = False
    torch_path = ""
    torch_version = ""
    
    for pkg in pkg_resources.working_set:
        if pkg.key == 'torch':
            has_torch = True
            torch_path = pkg.location
            torch_version = pkg.version
            break
    
    if has_torch:
        print(f"Found torch {torch_version} at {torch_path}")
    else:
        print("PyTorch not found in Python packages")
    
    # Look for invalid torch distributions
    pip_list_output = run_command([python_exe, "-m", "pip", "list"], "Checking pip packages")
    
    if "-orch" in pip_list_output:
        print("\n⚠️ WARNING: Invalid torch distribution detected ('-orch')")
        print("This indicates a corrupted PyTorch installation.")
    
    # 2. Uninstall timm safely
    print("\n" + "="*80)
    print("STEP 1: Safely uninstalling timm package")
    print("="*80)
    
    run_command([python_exe, "-m", "pip", "uninstall", "-y", "timm"], "Uninstalling timm")
    
    # 3. Identify and fix corrupted torch files
    print("\n" + "="*80)
    print("STEP 2: Fixing corrupted torch files")
    print("="*80)
    
    site_packages = next((p for p in sys.path if p.endswith('site-packages')), None)
    if not site_packages:
        print("Could not find site-packages directory")
    else:
        print(f"Site-packages directory: {site_packages}")
        
        # Check for -orch directory
        invalid_torch_dir = os.path.join(site_packages, "-orch")
        if os.path.exists(invalid_torch_dir):
            print(f"Found invalid '-orch' directory at {invalid_torch_dir}")
            backup_file(invalid_torch_dir)
            
            print("Removing invalid directory...")
            try:
                shutil.rmtree(invalid_torch_dir)
                print("Successfully removed invalid directory!")
            except Exception as e:
                print(f"Error removing directory: {e}")
    
    # 4. Reinstall torch if wheel file is available
    print("\n" + "="*80)
    print("STEP 3: Checking for local torch wheel file")
    print("="*80)
    
    torch_wheel_path = "/media/lugo/data/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    
    if os.path.exists(torch_wheel_path):
        print(f"Found local torch wheel file: {torch_wheel_path}")
        run_command([python_exe, "-m", "pip", "install", torch_wheel_path], 
                   "Reinstalling torch from local wheel")
    else:
        print(f"Local torch wheel file not found at: {torch_wheel_path}")
        print("Please reinstall torch manually using your preferred wheel file.")
    
    # 5. Create a simple depth generator that doesn't need torch or timm
    print("\n" + "="*80)
    print("STEP 4: Creating a torch-free depth map generator")
    print("="*80)
    
    # Create simple depth generator script
    depth_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "simple_depth_generator.py")
    
    print(f"Creating simple depth generator at: {depth_generator_path}")
    
    with open(depth_generator_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Depth Map Generator

This script uses OpenCV to generate a depth map approximation from an image.
It doesn't require PyTorch, timm or other deep learning libraries.
"""

import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def generate_simple_depth_map(image_path, output_path=None, display=True):
    """
    Generate a simple depth map approximation using OpenCV operations.
    This doesn't require PyTorch or timm.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the depth map (if None, will use image_path_depth.png)
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
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply Laplacian for edge detection (helps with depth perception)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Compute gradient magnitude using Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Combine Laplacian and gradient for better depth estimation
    depth_raw = gradient_magnitude + np.abs(laplacian)
    
    # Invert the depth map (closer objects are brighter)
    depth_raw = 1.0 - depth_raw / depth_raw.max()
    
    # Enhance contrast
    depth_raw = cv2.normalize(depth_raw, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply bilateral filter for edge-preserving smoothing
    depth_raw = cv2.bilateralFilter(depth_raw.astype(np.float32), 9, 75, 75)
    
    # Convert to 8-bit for visualization
    depth_map_norm = (depth_raw * 255).astype(np.uint8)
    
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
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Depth map generation completed in {elapsed_time:.2f} seconds")
    
    # Display results if requested
    if display:
        try:
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Depth map (grayscale)
            plt.subplot(1, 3, 2)
            plt.imshow(depth_raw, cmap='gray')
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
        except Exception as e:
            print(f"Error displaying results: {e}")
    
    return depth_raw

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
        depth_map = generate_simple_depth_map(image_path, display=False)
        
        # Set output path if not provided
        if output_path is None:
            output_path = "controlnet_input_depth.png"
        
        # Create a 3-channel depth map as required by ControlNet
        depth_map_norm = (depth_map * 255).astype(np.uint8)
        depth_3channel = np.stack([depth_map_norm, depth_map_norm, depth_map_norm], axis=2)
        
        # Save the ControlNet-ready depth map
        print(f"Saving ControlNet-ready depth map to {output_path}")
        cv2.imwrite(output_path, depth_3channel)
        
        return depth_3channel
        
    except Exception as e:
        print(f"Error preparing ControlNet depth map: {e}")
        raise

if __name__ == "__main__":
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
        depth_map = generate_simple_depth_map(input_image_path)
        
        # Generate ControlNet-ready depth map
        controlnet_depth = prepare_controlnet_depth(input_image_path)
        
        print("Depth map generation complete!")
    except Exception as e:
        print(f"Failed to generate depth map: {e}")
        sys.exit(1)
''')

    os.chmod(depth_generator_path, 0o755)  # Make executable
    print(f"Created depth generator script at {depth_generator_path}")
    
    # 6. Final check
    print("\n" + "="*80)
    print("STEP 5: Verifying environment")
    print("="*80)
    
    # Check if timm is still installed
    try:
        import importlib
        importlib.import_module('timm')
        print("⚠️ timm is still installed!")
    except ImportError:
        print("✅ timm has been successfully uninstalled")
    
    # Try importing torch without loading it
    try:
        # Check if module exists without importing
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            print("✅ torch package is found")
        else:
            print("⚠️ torch package not found")
    except:
        print("⚠️ Error checking torch package")
    
    # Check for torchvision and torchaudio
    for pkg in ['torchvision', 'torchaudio']:
        try:
            spec = importlib.util.find_spec(pkg)
            if spec is not None:
                print(f"✅ {pkg} package is intact")
            else:
                print(f"⚠️ {pkg} package not found")
        except:
            print(f"⚠️ Error checking {pkg} package")
    
    print("\n" + "="*80)
    print("REPAIR PROCESS COMPLETE!")
    print("="*80)
    print("\nYou can now use the simple_depth_generator.py script to generate depth maps")
    print("without requiring torch or timm packages.")
    
    print("\nIf you want to completely reinstall your PyTorch environment, you can run:")
    print(f"  {pip_exe} install /media/lugo/data/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl")
    
    return True

if __name__ == "__main__":
    try:
        fix_environment()
    except Exception as e:
        print(f"\nError during environment fix: {e}")
        sys.exit(1)