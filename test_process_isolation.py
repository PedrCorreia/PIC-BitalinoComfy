#!/usr/bin/env python3
"""
Test script for the Render3D process isolation implementation.
This verifies that our implementation properly isolates OpenGL/CUDA contexts
between the PyVista 3D rendering and display window systems.
"""

import os
import sys
import time
import numpy as np
import threading
import random
import cv2
import math
import gc

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Render3D and geometry classes
from src.geometry.render3d import Render3D
from src.geometry.geom import Sphere, Cube

# Try to import lunar_tools for display window (optional)
try:
    from lunar_tools.display_window import Renderer
    HAS_DISPLAY_WINDOW = True
except ImportError:
    print("lunar_tools.display_window not available - will skip display window test")
    HAS_DISPLAY_WINDOW = False

class ProcessIsolationTest:
    """Test harness for render3d process isolation"""
    
    def __init__(self):
        self.img_size = 512
        self.background = 'white'
        self.safe_mode = False
        self.use_process_isolation = True
        self.display_enabled = HAS_DISPLAY_WINDOW
        
        self.display_window = None
        self.renderer = None
        self.running = True
        self.render_thread = None
        self.display_thread = None
    
    def create_sphere_scene(self):
        """Create a simple scene with multiple colored spheres"""
        renderer = Render3D(
            img_size=self.img_size,
            background=self.background,
            safe_mode=self.safe_mode,
            process_isolation=self.use_process_isolation
        )
        
        # Add spheres in different positions
        sphere_positions = [
            (0, 0, 0),
            (-1, 0.5, 0.5),
            (1, -0.5, 0.8),
            (0.3, 0.7, -0.5),
            (-0.7, -0.3, -0.2)
        ]
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        
        for i, (pos, color) in enumerate(zip(sphere_positions, colors)):
            sphere = Sphere(
                center=pos,
                radius=0.4,
                quality='medium'
            )
            renderer.add_geometry(sphere, color=color, edge_color='white', opacity=1.0)
        
        return renderer
    
    def create_cube_scene(self):
        """Create a simple scene with multiple colored cubes"""
        renderer = Render3D(
            img_size=self.img_size,
            background=self.background,
            safe_mode=self.safe_mode,
            process_isolation=self.use_process_isolation
        )
        
        # Add cubes in different positions with rotations
        cube_positions = [
            (0, 0, 0, 0, 0, 0),
            (-1, 0.5, 0.5, 30, 0, 0),
            (1, -0.5, 0.8, 0, 45, 0),
            (0.3, 0.7, -0.5, 0, 0, 60),
            (-0.7, -0.3, -0.2, 30, 45, 60)
        ]
        
        colors = ['cyan', 'magenta', 'teal', 'orange', 'pink']
        
        for i, (x, y, z, rx, ry, rz, color) in enumerate(zip([p[0] for p in cube_positions], 
                                                           [p[1] for p in cube_positions],
                                                           [p[2] for p in cube_positions],
                                                           [p[3] for p in cube_positions],
                                                           [p[4] for p in cube_positions],
                                                           [p[5] for p in cube_positions],
                                                           colors)):
            cube = Cube(
                center=(x, y, z),
                width=0.5,
                rotation=(rx, ry, rz)
            )
            renderer.add_geometry(cube, color=color, edge_color='black', opacity=1.0)
        
        return renderer
    
    def render_thread_func(self):
        """Thread for continuous 3D rendering"""
        print("Render thread starting")
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Alternate between sphere and cube scenes
                if frame_count % 2 == 0:
                    renderer = self.create_sphere_scene()
                else:
                    renderer = self.create_cube_scene()
                
                # Render the scene
                camera_pos = (4, 3, 2)  # Example camera position
                img = renderer.render(camera_position=camera_pos, show_edges=True)
                
                # Make a copy of the image before cleanup to avoid shared memory issues
                if img is not None:
                    try:
                        img = img.copy()
                    except Exception as e:
                        print(f"Error copying image: {e}")
                        img = None
                
                # Clean up renderer resources properly
                try:
                    if renderer:
                        renderer.cleanup()
                        # Force garbage collection to help release shared memory
                        gc.collect()
                except Exception as e:
                    print(f"Error during renderer cleanup: {e}")
                
                # Update display if available
                if self.display_enabled and img is not None:
                    self.last_rendered_img = img
                
                # Sleep briefly
                frame_count += 1
                time.sleep(0.1)
                
                # Print status every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Rendered {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
                    
            except Exception as e:
                print(f"Error in render thread: {e}")
                time.sleep(1.0)  # Wait a bit before retry
        
        print("Render thread exiting")
    
    def display_thread_func(self):
        """Thread for display window update using OpenCV for reliable rendering"""
        if not self.display_enabled:
            return
            
        print("Display thread starting with OpenCV window")
        
        try:
            # Create OpenCV window - more reliable than using lunar_tools.display_window
            cv2.namedWindow("Render3D Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Render3D Test", self.img_size, self.img_size)
            
            # Create a rotating box animation for when no render is available
            no_render_frame_count = 0
            
            # Update display
            while self.running:
                try:
                    if hasattr(self, 'last_rendered_img') and self.last_rendered_img is not None:
                        # Make a copy of the image to avoid any shared memory issues
                        img_to_display = self.last_rendered_img.copy()
                        # OpenCV uses BGR format
                        img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Render3D Test", img_to_display)
                        no_render_frame_count = 0
                    else:
                        # Create a simple animation when no render is available
                        no_render_frame_count += 1
                        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 40
                        
                        # Draw a rotating box
                        center = (self.img_size // 2, self.img_size // 2)
                        size = self.img_size // 4
                        angle = (no_render_frame_count * 2) % 360
                        
                        # Calculate rotated box corners
                        rad = math.radians(angle)
                        cos_val = math.cos(rad)
                        sin_val = math.sin(rad)
                        
                        def rotate_point(x, y):
                            x_rot = (x - center[0]) * cos_val - (y - center[1]) * sin_val + center[0]
                            y_rot = (x - center[0]) * sin_val + (y - center[1]) * cos_val + center[1]
                            return (int(x_rot), int(y_rot))
                        
                        # Create box points
                        pts = [
                            rotate_point(center[0] - size, center[1] - size),
                            rotate_point(center[0] + size, center[1] - size),
                            rotate_point(center[0] + size, center[1] + size),
                            rotate_point(center[0] - size, center[1] + size)
                        ]
                        
                        # Draw the box
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, (0, 120, 255), 2)
                        
                        # Add text
                        cv2.putText(img, "Waiting for rendered frame...", 
                                  (self.img_size // 10, self.img_size // 2 + size + 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        
                        cv2.imshow("Render3D Test", img)
                    
                    # Process any key presses (with short timeout)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        print("ESC pressed, exiting...")
                        self.running = False
                        break
                        
                    # Limit refresh rate
                    time.sleep(0.02)  # ~50 FPS update rate
                    
                except Exception as e:
                    print(f"Error during display update: {e}")
                    time.sleep(0.5)  # Slower retry on error
                
        except Exception as e:
            print(f"Error in display thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up OpenCV window
            try:
                cv2.destroyWindow("Render3D Test")
            except:
                pass
            
        print("Display thread exiting")
    
    def run_test(self, duration=10.0):
        """Run the test for the specified duration in seconds"""
        print(f"Starting process isolation test (duration: {duration}s)")
        print(f"Process isolation enabled: {self.use_process_isolation}")
        print(f"Display window enabled: {self.display_enabled}")
        
        # Start render thread
        self.render_thread = threading.Thread(target=self.render_thread_func)
        self.render_thread.daemon = True
        self.render_thread.start()
        
        # Start display thread if enabled
        if self.display_enabled:
            self.display_thread = threading.Thread(target=self.display_thread_func)
            self.display_thread.daemon = True
            self.display_thread.start()
        
        # Wait for specified duration
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("Test interrupted by user")
        
        # Cleanup
        self.running = False
        
        if self.render_thread is not None:
            self.render_thread.join(timeout=2.0)
            
        if self.display_thread is not None:
            self.display_thread.join(timeout=2.0)
            
        print("Test completed")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Render3D process isolation')
    parser.add_argument('--no-isolation', action='store_true', help='Disable process isolation')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--duration', type=float, default=10.0, help='Test duration in seconds')
    parser.add_argument('--safe-mode', action='store_true', help='Use CPU rendering (safe mode)')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = ProcessIsolationTest()
    test.use_process_isolation = not args.no_isolation
    test.display_enabled = HAS_DISPLAY_WINDOW and not args.no_display
    test.safe_mode = args.safe_mode
    
    test.run_test(duration=args.duration)
