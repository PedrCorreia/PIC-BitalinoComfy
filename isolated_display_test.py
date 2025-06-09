#!/usr/bin/env python3
"""Test isolated display window without PyVista interference"""

import multiprocessing
import numpy as np
import torch
import ctypes
import sys
sys.path.append('/home/lugo/ComfyUI/custom_nodes')
from lunar_tools_comfy.comfy.display_window import LRRenderer
import time
import sys
import os

def display_worker(shared_array_info, shape, running_flag):
    """Worker process for displaying images"""
    print(f"[display_worker] Started with PID: {os.getpid()}")
    
    # Import here to avoid sharing CUDA/OpenGL contexts
    import ctypes
    
    # Recreate shared array
    shared_array = np.ctypeslib.as_array(shared_array_info).reshape(shape)
    
    # Create renderer in this process
    renderer = LRRenderer()
    frame_count = 0
    
    while running_flag.value:
        try:
            # Copy data from shared memory
            image_data = shared_array.copy()
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_data).float()
            
            frame_count += 1
            print(f"[display_worker] Frame {frame_count}, image stats: min={image_data.min()}, max={image_data.max()}, mean={image_data.mean():.2f}")
            
            # Render
            renderer.render(image_tensor)
            
            time.sleep(0.1)  # 10 fps
            
        except Exception as e:
            print(f"[display_worker] Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("[display_worker] Exiting")

def geometry_worker(shared_array_info, shape, running_flag):
    """Worker process for geometry rendering"""
    print(f"[geometry_worker] Started with PID: {os.getpid()}")
    
    # Import PyVista here to isolate it
    import pyvista as pv
    import ctypes
    
    # Recreate shared array
    shared_array = np.ctypeslib.as_array(shared_array_info).reshape(shape)
    
    # Create PyVista scene
    pv.set_plot_theme('dark')
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    
    # Create a simple rotating cube
    mesh = pv.Cube()
    plotter.add_mesh(mesh, color='red')
    
    frame_count = 0
    
    while running_flag.value:
        try:
            frame_count += 1
            
            # Rotate the camera
            plotter.camera.azimuth += 5
            
            # Render to image
            image = plotter.screenshot(return_img=True)  # Returns PIL Image
            image_array = np.array(image)
            
            # Resize to match expected shape if needed
            if image_array.shape != shape:
                import cv2
                image_array = cv2.resize(image_array, (shape[1], shape[0]))
                if len(image_array.shape) == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.shape[2] == 4:  # RGBA -> RGB
                    image_array = image_array[:, :, :3]
            
            # Write to shared memory
            shared_array[:] = image_array
            
            print(f"[geometry_worker] Frame {frame_count}, rendered image stats: min={image_array.min()}, max={image_array.max()}, mean={image_array.mean():.2f}")
            
            time.sleep(0.1)  # 10 fps
            
        except Exception as e:
            print(f"[geometry_worker] Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("[geometry_worker] Exiting")
    plotter.close()

def main():
    print("Testing isolated display with separate processes...")
    
    # Image dimensions
    height, width, channels = 512, 512, 3
    shape = (height, width, channels)
    
    # Create shared memory for image data
    shared_array_base = multiprocessing.Array(ctypes.c_uint8, height * width * channels)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj()).reshape(shape)
    
    # Initialize with a test pattern
    test_image = np.zeros(shape, dtype=np.uint8)
    test_image[:, :, 0] = 255  # Red channel
    shared_array[:] = test_image
    
    # Running flag
    running_flag = multiprocessing.Value('i', 1)
    
    # Start processes
    display_process = multiprocessing.Process(
        target=display_worker, 
        args=(shared_array_base.get_obj(), shape, running_flag)
    )
    
    geometry_process = multiprocessing.Process(
        target=geometry_worker,
        args=(shared_array_base.get_obj(), shape, running_flag)
    )
    
    display_process.start()
    geometry_process.start()
    
    try:
        # Let it run for a while
        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Shutting down...")
        running_flag.value = 0
        
        # Wait for processes to finish
        display_process.join(timeout=2)
        geometry_process.join(timeout=2)
        
        # Force terminate if they don't exit cleanly
        if display_process.is_alive():
            display_process.terminate()
        if geometry_process.is_alive():
            geometry_process.terminate()
    
    print("Test completed")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Important for CUDA/OpenGL isolation
    main()
