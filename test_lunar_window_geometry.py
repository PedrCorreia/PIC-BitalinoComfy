#!/usr/bin/env python3
"""
Test script for displaying GeometryRenderNode output in a lunar_tools display window.
This continuously renders different 3D geometries in a background thread
and displays them in the lunar_tools window.
"""

import os
import sys
import time
import numpy as np
import threading
import torch
import math
import random
from PIL import Image

# Add paths to make imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add PIC-BitalinoComfy directory
sys.path.append(os.path.dirname(script_dir))  # Add the parent directory (custom_nodes)

# Import GeometryRenderNode
from comfy.geom.geometry_node import GeometryRenderNode

# Import lunar_tools display window
lunar_tools_path = os.path.join(os.path.dirname(script_dir), "lunar_tools_comfy")
if os.path.exists(lunar_tools_path):
    sys.path.append(lunar_tools_path)
    from comfy.display_window import LRRenderer
    HAS_LUNAR_TOOLS = True
    print(f"Found lunar_tools at: {lunar_tools_path}")
else:
    HAS_LUNAR_TOOLS = False
    print(f"WARNING: Could not find lunar_tools at {lunar_tools_path}")

class GeometryRenderer:
    """Class to handle continuous 3D geometry rendering in a background thread"""
    
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.node = GeometryRenderNode()
        self.running = False
        self.render_thread = None
        self.current_image = None
        self.lock = threading.Lock()
        
        # Initialize lunar tools renderer if available
        self.lunar_renderer = LRRenderer() if HAS_LUNAR_TOOLS else None
        
        # Parameters that will be animated
        self.params = {
            "object_type": "cube",
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size": 1.0,
            "rotation_deg_x": 0.0,
            "rotation_deg_y": 0.0,
            "rotation_deg_z": 0.0,
            "z_distance": 5.0,
            "img_size": self.img_size,
            "color": "#FF5500"
        }
    
    def start(self):
        """Start the render thread"""
        if self.render_thread is not None and self.render_thread.is_alive():
            print("Render thread is already running")
            return
            
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.daemon = True
        self.render_thread.start()
    
    def stop(self):
        """Stop the render thread"""
        self.running = False
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
    
    def _get_animated_params(self, t):
        """Generate animated parameters based on time t"""
        # Switch between cube and sphere periodically
        if int(t / 3) % 2 == 0:
            self.params["object_type"] = "cube"
            self.params["color"] = "#FF5500"
        else:
            self.params["object_type"] = "sphere"
            self.params["color"] = "#00AAFF"
        
        # Animate rotation
        self.params["rotation_deg_x"] = (t * 20) % 360
        self.params["rotation_deg_y"] = (t * 30) % 360
        self.params["rotation_deg_z"] = (t * 10) % 360
        
        # Animate position in a small circle
        radius = 0.5
        self.params["center_x"] = radius * math.cos(t * 0.5)
        self.params["center_y"] = radius * math.sin(t * 0.5)
        self.params["center_z"] = 0.2 * math.sin(t * 1.5)
        
        # Animate size slightly
        self.params["size"] = 1.0 + 0.3 * math.sin(t * 0.8)
        
        return self.params
    
    def _render_loop(self):
        """Background thread for continuous rendering"""
        print("Starting render loop in background thread")
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.running:
                # Get time for animation
                t = time.time() - start_time
                
                # Get animated parameters
                params = self._get_animated_params(t)
                
                # Render the geometry
                color_tensor, depth_tensor = self.node.render(**params)
                
                # Create a combined tensor with both color and depth
                # Depth in bottom right corner (1/4 size)
                if color_tensor is not None and depth_tensor is not None:
                    with self.lock:
                        self.current_image = color_tensor
                    
                    # Update lunar renderer if available
                    if self.lunar_renderer is not None:
                        self.lunar_renderer.render(
                            image=color_tensor, 
                            height=self.img_size, 
                            width=self.img_size,
                            window_title=f"3D Geometry Render - {params['object_type']}"
                        )
                
                # Count frames and calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Rendered {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
                
                # Sleep a bit to avoid using 100% CPU
                time.sleep(0.05)
                
        except Exception as e:
            import traceback
            print(f"Error in render loop: {e}")
            traceback.print_exc()
            self.running = False
        
        print("Render loop stopped")
    
    def get_current_image(self):
        """Get the current rendered image"""
        with self.lock:
            if self.current_image is not None:
                return self.current_image.clone()
            return None

def main():
    """Main function to run the test"""
    print("Starting geometry renderer with lunar_tools display window...")
    
    if not HAS_LUNAR_TOOLS:
        print("ERROR: lunar_tools_comfy not found. Cannot continue.")
        return
    
    img_size = 512
    renderer = GeometryRenderer(img_size=img_size)
    
    try:
        # Start the renderer
        renderer.start()
        
        # Wait for the user to press Ctrl+C
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        # Clean up
        renderer.stop()
        print("Done.")

if __name__ == "__main__":
    main()
