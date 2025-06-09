import sys
import os
import inspect
import importlib.util
import atexit
import numpy as np
import torch

# More robustly determine project paths
current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
comfy_dir = os.path.dirname(os.path.dirname(current_file_path))  # comfy/geom -> comfy
pic_root = os.path.dirname(comfy_dir)  # PIC-BitalinoComfy
project_root = os.path.dirname(pic_root)  # custom_nodes

# Add paths to sys.path if not already there
for path in [project_root, pic_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Handle both direct and relative imports
geometry_module_path = os.path.join(pic_root, "src", "geometry")

try:
    # Try direct import first
    try:
        from src.geometry.render3d_comfy import render_geometry_for_comfy, cleanup_renderer
        from src.geometry.geom import Sphere, Cube
        print("Successfully imported render modules directly")
    except ImportError:
        # If direct import fails, try a more explicit approach with importlib
        render3d_comfy_path = os.path.join(geometry_module_path, "render3d_comfy.py")
        geom_path = os.path.join(geometry_module_path, "geom.py")
        
        if os.path.exists(render3d_comfy_path) and os.path.exists(geom_path):
            print(f"Found modules at: {render3d_comfy_path}")
            
            # Import render3d_comfy dynamically
            spec = importlib.util.spec_from_file_location("render3d_comfy", render3d_comfy_path)
            render3d_comfy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(render3d_comfy)
            render_geometry_for_comfy = render3d_comfy.render_geometry_for_comfy
            cleanup_renderer = render3d_comfy.cleanup_renderer
            
            # Import geometry classes dynamically
            spec = importlib.util.spec_from_file_location("geom", geom_path)
            geom = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(geom)
            Sphere = geom.Sphere
            Cube = geom.Cube
            
            print("Successfully imported render modules with importlib")
        else:
            raise ImportError(f"Could not find required modules at {geometry_module_path}")
    
    # Register cleanup function to ensure renderer is properly shut down on exit
    atexit.register(cleanup_renderer)
except ImportError as e:
    print("[ERROR] Could not import from src.geometry. sys.path is:", sys.path)
    print("[ERROR] Exception:", e)
    print("[ERROR] Module search paths:", geometry_module_path)
    print("[ERROR] Make sure you are running this script from a location where the project root is in sys.path and that src/geometry exists.")
    raise

class GeometryRenderNode:
    """
    ComfyUI node for rendering 3D geometry with process isolation.
    Uses the Render3D class with improved process isolation to prevent OpenGL context conflicts.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_type": (["sphere", "cube"],),
                "center_x": ("FLOAT",{"default":0.0,"min":-5, "max":5}),
                "center_y": ("FLOAT",{"default":0.0,"min":-5, "max":5}),
                "center_z": ("FLOAT",{"default":0.0,"min":-5, "max":5}),
                "size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "rotation_deg_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
                "rotation_deg_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
                "rotation_deg_z": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
                "z_distance": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0}),
                "img_size": ("INT", {"default": 512, "min": 512, "max": 1024}),
                "color": ("STRING", {"default": "#FFD700"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "geometry"
    
    def __init__(self):
        # Register cleanup on node destruction
        self._destroyed = False
    
    def __del__(self):
        if not self._destroyed:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources when node is deleted"""
        self._destroyed = True
        # Ensure renderer is cleaned up properly
        try:
            cleanup_renderer()
        except Exception as e:
            print(f"Error during GeometryRenderNode cleanup: {e}")
            pass

    def render(self, object_type, center_x, center_y, center_z, size, rotation_deg_x, rotation_deg_y, rotation_deg_z, z_distance, img_size, color):
        """Render the geometry with process isolation"""
        # Camera setup: looking at origin from +z
        camera_position = [(0, 0, z_distance), (0, 0, 0), (0, 1, 0)]
        
        print(f"[GeometryRenderNode] Rendering {object_type} at ({center_x}, {center_y}, {center_z}) with size {size}")
        print(f"[GeometryRenderNode] Rotation: ({rotation_deg_x}, {rotation_deg_y}, {rotation_deg_z}), color: {color}")
        print(f"[GeometryRenderNode] Output image size: {img_size}x{img_size}")
        
        # Track render attempts for better error handling
        max_attempts = 2
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Try rendering with process isolation
                result = self._render_with_renderer(
                    None, object_type, center_x, center_y, center_z, size,
                    rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color,
                    img_size=img_size, background='white', show_edges=True,
                    force_recreate=(attempt > 0)  # Force recreation on retry
                )
                
                # If we got a valid result, return it
                if result is not None:
                    if torch.is_tensor(result):
                        print(f"[GeometryRenderNode] Successfully rendered image: shape={tuple(result.shape)}, dtype={result.dtype}")
                    return result
                
                # If we get here, rendering failed but returned None - try again
                print(f"[GeometryRenderNode] Render attempt {attempt+1}/{max_attempts} failed with None result, retrying...")
                attempt += 1
                
            except Exception as e:
                print(f"[GeometryRenderNode] Process-isolated rendering failed on attempt {attempt+1}/{max_attempts}: {e}")
                import traceback
                traceback.print_exc()
                
                # Only retry once
                if attempt < max_attempts - 1:
                    print("Retrying with fresh renderer...")
                    attempt += 1
                    # Force cleanup before retry
                    try:
                        cleanup_renderer()
                    except:
                        pass
                else:
                    # We've tried enough, return fallback
                    break
        
        # If we got here, all attempts failed
        print(f"All {max_attempts} render attempts failed, returning fallback image")
        return self._create_fallback_tensors(img_size)
    
    def _render_with_renderer(self, renderer, object_type, center_x, center_y, center_z, size, 
                            rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color, 
                            img_size=None, background=None, show_edges=True, force_recreate=False):
        """
        Shared rendering logic for both GPU and CPU, responsive to all node inputs
        
        Args:
            renderer: Optional existing renderer to use (ignored in process-isolated mode)
            object_type: Type of geometry to render ("sphere" or "cube")
            center_x, center_y, center_z: Center position of the geometry
            size: Size of the geometry (diameter for sphere, width for cube)
            rotation_deg_x, rotation_deg_y, rotation_deg_z: Rotation angles in degrees
            camera_position: Camera position for rendering
            color: Color of the geometry (hex string)
            img_size: Size of the rendered image
            background: Background color
            show_edges: Whether to show edges of the geometry
            force_recreate: If True, force recreation of the renderer
        """
        print(f"[GeometryRenderNode] Starting render with object_type={object_type}, center=({center_x}, {center_y}, {center_z}), size={size}, rotation=({rotation_deg_x}, {rotation_deg_y}, {rotation_deg_z}), camera_position={camera_position}, color={color}")
        # Create geometry object
        if object_type == "sphere":
            geom = Sphere(center=(center_x, center_y, center_z), 
                        radius=size/2, 
                        quality='medium', 
                        rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))
        else:  # cube
            geom = Cube(center=(center_x, center_y, center_z), 
                      width=size, 
                      quality='medium', 
                      rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))
        
        # Use process-isolated rendering
        geom_args = [(geom, color, 'black', 1.0)]
        
        # Use render_geometry_for_comfy for process-isolated rendering with retry logic
        color_img = render_geometry_for_comfy(
            geom_args,
            img_size=img_size if img_size is not None else 512,
            background=background if background is not None else 'white',
            show_edges=show_edges,
            camera_position=camera_position,
            retry_on_failure=True  # Allow one retry if rendering fails
        )
        
        # Process and return the output
        return self._process_outputs(color_img)
    
    def _process_outputs(self, color_img):
        """Process rendered outputs into ComfyUI-compatible tensors with proper negative strides handling"""
        print("[GeometryRenderNode] Processing outputs")
        # Process color image
        if color_img is not None:
            # Convert to numpy array if needed
            if torch.is_tensor(color_img):
                color_img = color_img.detach().cpu().numpy()
            
            # Ensure array is contiguous and make a copy to fix negative strides
            color_img = np.ascontiguousarray(color_img.copy())
            
            # Handle RGBA to RGB conversion
            if len(color_img.shape) == 3 and color_img.shape[2] == 4:  # RGBA
                color_img = color_img[:, :, :3]  # Remove alpha channel
            
            # Normalize to [0, 1] if needed
            if color_img.dtype == np.uint8:
                color_img = color_img.astype(np.float32) / 255.0
            elif color_img.max() > 1.0:
                color_img = color_img.astype(np.float32) / color_img.max()
            else:
                color_img = color_img.astype(np.float32)
            
            # Ensure RGB format (H, W, 3)
            if len(color_img.shape) == 2:  # Grayscale
                color_img = np.stack([color_img] * 3, axis=-1)
            elif len(color_img.shape) == 3 and color_img.shape[2] == 1:  # Single channel
                color_img = np.repeat(color_img, 3, axis=2)
            
            # Convert to tensor and add batch dimension [1, H, W, 3]
            color_tensor = torch.from_numpy(color_img).float()
            if len(color_tensor.shape) == 3:
                color_tensor = color_tensor.unsqueeze(0)
        else:
            # Fallback: create black image
            color_tensor = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
            
        return color_tensor
    
    def _create_fallback_tensors(self, img_size):
        """Create fallback tensors when rendering fails"""
        color_tensor = torch.zeros(1, img_size, img_size, 3, dtype=torch.float32)
        return color_tensor


if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    import cv2 # Keep cv2 import here if used later for display
    from PIL import Image # Keep PIL import here if used later for saving

    node = GeometryRenderNode()
    params = {
        "object_type": "cube",
        "center_x": 1,
        "center_y": 0,
        "center_z": -1,
        "size": 1.0,
        "rotation_deg_x": 45.0,
        "rotation_deg_y": 65.0,
        "rotation_deg_z": 45.0,
        "z_distance": 10.0,
        "img_size": 512,
        "color": "#FFD700"
    }

    # Create a cProfile object
    profiler = cProfile.Profile()

    # Run the render method under profiler
    print("Starting profiling...")
    profiler.enable()
    
    # Run the function multiple times to get a better average if needed,
    # but for a single call profile:
    color_img_tensor = node.render(**params)
    
    profiler.disable()
    print("Profiling finished.")

    # Create a stream for the stats
    s = io.StringIO()
    # Sort stats by cumulative time
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()

    # Print the stats to console
    print(s.getvalue())

    # Optionally, save stats to a file
    profile_output_path = "geometry_node_profile.prof"
    profiler.dump_stats(profile_output_path)
    print(f"Profile data saved to {profile_output_path}")
    print(f"You can view it using: python -m pstats {profile_output_path}")

    # --- Original image processing and display/save logic ---
    if color_img_tensor is not None:
        print("Processing image for display/saving...")
        # Prepare color image - convert tensor to numpy first
        color_img_np = color_img_tensor.cpu().numpy() if hasattr(color_img_tensor, 'cpu') else color_img_tensor
        if color_img_np.dtype != np.uint8:
            # Ensure it's float before clipping and scaling
            if not np.issubdtype(color_img_np.dtype, np.floating):
                 color_img_np = color_img_np.astype(np.float32)
            
            # Handle potential normalization issues if max is 0 or very small
            if color_img_np.max() > 1e-6 : # Check if there's actual image data
                if color_img_np.max() > 1.0: # If not in [0,1] range, scale by max
                    color_img_np = color_img_np / color_img_np.max()

            color_img_disp = (np.clip(color_img_np, 0, 1) * 255).astype(np.uint8)
        else:
            color_img_disp = color_img_np
        
        # Remove batch dimension if present
        if color_img_disp.ndim == 4 and color_img_disp.shape[0] == 1:
            color_img_disp = color_img_disp[0]
        
        # Ensure 3 channels for display with OpenCV
        if color_img_disp.ndim == 2: # Grayscale
            color_img_disp = cv2.cvtColor(color_img_disp, cv2.COLOR_GRAY2BGR)
        elif color_img_disp.shape[2] == 1: # Single channel (e.g. depth map)
             color_img_disp = cv2.cvtColor(color_img_disp, cv2.COLOR_GRAY2BGR)


        # Display the rendered image (optional, can be slow)
        # cv2.imshow('3D Color Render', color_img_disp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Save the image
        try:
            Image.fromarray(color_img_disp).save("test_profiled_render_color.png")
            print("Saved profiled color render to test_profiled_render_color.png")
        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        print("Rendering returned None, cannot process or save image.")

    print("Done.")
