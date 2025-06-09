#!/usr/bin/env python3
"""
Test script for the GeometryRenderNode with process isolation.
This verifies that our implementation properly renders 3D geometry with process isolation.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

# Add paths to make imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add PIC-BitalinoComfy directory
sys.path.append(os.path.dirname(script_dir))  # Add the parent directory (custom_nodes)

# Make sure src.geometry can be found
src_geometry_path = os.path.join(script_dir, "src", "geometry")
if os.path.exists(src_geometry_path):
    print(f"Found src/geometry at: {src_geometry_path}")
else:
    print(f"WARNING: src/geometry directory not found at {src_geometry_path}")

# Print sys.path for debugging
print("sys.path includes:", sys.path)

# Import GeometryRenderNode
from comfy.geom.geometry_node import GeometryRenderNode

def main():
    """Test the GeometryRenderNode with process isolation"""
    print("Testing GeometryRenderNode with process isolation...")
    
    # Create node
    node = GeometryRenderNode()
    
    # Parameters for rendering
    params = {
        "object_type": "cube",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size": 1.0,
        "rotation_deg_x": 30.0,
        "rotation_deg_y": 45.0,
        "rotation_deg_z": 0.0,
        "z_distance": 5.0,
        "img_size": 512,
        "color": "#FF5500"
    }
    
    # Render the geometry
    print("Rendering cube...")
    color_tensor, depth_tensor = node.render(**params)
    
    # Convert to numpy arrays
    print("Converting tensors to numpy arrays...")
    print(f"Color tensor shape: {color_tensor.shape}, dtype: {color_tensor.dtype}")
    print(f"Depth tensor shape: {depth_tensor.shape}, dtype: {depth_tensor.dtype}")
    
    color_img = color_tensor.numpy()[0]  # Remove batch dimension
    depth_img = depth_tensor.numpy()[0, :, :, 0]  # Remove batch and channel dimensions
    
    print(f"Color image shape: {color_img.shape}, dtype: {color_img.dtype}")
    print(f"Depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")
    print(f"Color image min: {color_img.min()}, max: {color_img.max()}")
    print(f"Depth image min: {depth_img.min()}, max: {depth_img.max()}")
    
    # Convert to uint8 for display
    color_img_uint8 = (color_img * 255).astype(np.uint8)
    depth_img_uint8 = (depth_img * 255).astype(np.uint8)
    
    # Save images - this is the most important part for verification
    print("Saving images to disk...")
    Image.fromarray(color_img_uint8).save("test_geometry_node_color.png")
    Image.fromarray(depth_img_uint8).save("test_geometry_node_depth.png")
    print(f"Images saved to {os.path.abspath('test_geometry_node_color.png')}")
    
    # Store images for combined display later
    cube_color_img = color_img_uint8
    cube_depth_img = depth_img_uint8
    
    # Try another shape
    params["object_type"] = "sphere"
    params["color"] = "#00AAFF"
    params["rotation_deg_y"] = 0.0
    
    print("Rendering sphere...")
    color_tensor, depth_tensor = node.render(**params)
    
    # Convert to numpy arrays
    print("Converting sphere tensors to numpy arrays...")
    print(f"Sphere color tensor shape: {color_tensor.shape}, dtype: {color_tensor.dtype}")
    print(f"Sphere depth tensor shape: {depth_tensor.shape}, dtype: {depth_tensor.dtype}")
    
    color_img = color_tensor.numpy()[0]
    depth_img = depth_tensor.numpy()[0, :, :, 0]
    
    print(f"Sphere color image shape: {color_img.shape}, dtype: {color_img.dtype}")
    print(f"Sphere depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")
    print(f"Sphere color image min: {color_img.min()}, max: {color_img.max()}")
    print(f"Sphere depth image min: {depth_img.min()}, max: {depth_img.max()}")
    
    # Convert to uint8 for display
    color_img_uint8 = (color_img * 255).astype(np.uint8)
    depth_img_uint8 = (depth_img * 255).astype(np.uint8)
    
    # Save images
    print("Saving sphere images to disk...")
    Image.fromarray(color_img_uint8).save("test_geometry_node_sphere_color.png")
    Image.fromarray(depth_img_uint8).save("test_geometry_node_sphere_depth.png")
    print(f"Sphere images saved to {os.path.abspath('test_geometry_node_sphere_color.png')}")
    
    # Store sphere images
    sphere_color_img = color_img_uint8
    sphere_depth_img = depth_img_uint8
    
    # Create combined display of all images
    try:
        print("Creating combined image display...")
        # Create a 2x2 grid layout
        h, w = cube_color_img.shape[:2]
        combined_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Add labels to the images
        cube_color_labeled = cube_color_img.copy()
        cv2.putText(cube_color_labeled, "Cube Color", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cube_depth_labeled = cv2.cvtColor(cube_depth_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(cube_depth_labeled, "Cube Depth", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        sphere_color_labeled = sphere_color_img.copy()
        cv2.putText(sphere_color_labeled, "Sphere Color", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                   
        sphere_depth_labeled = cv2.cvtColor(sphere_depth_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(sphere_depth_labeled, "Sphere Depth", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Place images in the grid
        combined_img[0:h, 0:w] = cv2.cvtColor(cube_color_labeled, cv2.COLOR_RGB2BGR)
        combined_img[0:h, w:w*2] = cube_depth_labeled
        combined_img[h:h*2, 0:w] = cv2.cvtColor(sphere_color_labeled, cv2.COLOR_RGB2BGR)
        combined_img[h:h*2, w:w*2] = sphere_depth_labeled
        
        # Save combined image
        cv2.imwrite("test_geometry_node_combined.png", combined_img)
        print(f"Combined image saved to {os.path.abspath('test_geometry_node_combined.png')}")
        
        # Display combined image
        cv2.imshow("Geometry Rendering Test Results", combined_img)
        print("Press any key to close the display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Warning: Failed to display combined images with OpenCV: {e}")
        print("This is normal if running in a headless environment.")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
