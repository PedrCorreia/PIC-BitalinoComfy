import sys
import os
import inspect
import importlib.util
import atexit
import numpy as np
import torch
import cProfile
import pstats
import io
import time # Ensure time is imported at the top

# More robustly determine project paths
current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
comfy_dir = os.path.dirname(os.path.dirname(current_file_path))  # comfy/geom -> comfy
pic_root = os.path.dirname(comfy_dir)  # PIC-BitalinoComfy
project_root = os.path.dirname(pic_root)  # custom_nodes

# Add paths to sys.path if not already there
for path in [project_root, pic_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Global list for active node instances
_active_geometry_nodes = []

# Global atexit handler for saving node profiles
def _save_all_node_profiles_on_exit():
    print(f"[ATEIXT] Attempting to save profiles for {len(_active_geometry_nodes)} active node(s).")
    for node_instance in list(_active_geometry_nodes): # Iterate over a copy
        try:
            print(f"[ATEIXT] Calling dump_profile for node: {id(node_instance)}")
            node_instance.dump_profile()
        except Exception as e:
            print(f"[ATEIXT] Error dumping profile for node {id(node_instance)}: {e}")

atexit.register(_save_all_node_profiles_on_exit)

# Handle both direct and relative imports
geometry_module_path = os.path.join(pic_root, "src", "geometry")
surface_eff_module_path = os.path.join(pic_root, "src", "geometry", "surface_eff.py") # Added for Surface_Noise

try:
    # Try direct import first
    try:
        from src.geometry.render3d_comfy import render_geometry_for_comfy, cleanup_renderer
        from src.geometry.geom import Sphere, Cube
        from src.geometry.surface_eff import Surface_Noise # Added import
        print("Successfully imported render and surface_eff modules directly")
    except ImportError:
        # If direct import fails, try a more explicit approach with importlib
        render3d_comfy_path = os.path.join(geometry_module_path, "render3d_comfy.py")
        geom_path = os.path.join(geometry_module_path, "geom.py")
        
        if os.path.exists(render3d_comfy_path) and os.path.exists(geom_path) and os.path.exists(surface_eff_module_path): # Check for surface_eff
            print(f"Found modules at: {render3d_comfy_path}, {geom_path}, {surface_eff_module_path}")
            
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

            # Import Surface_Noise dynamically
            spec_surface_eff = importlib.util.spec_from_file_location("surface_eff", surface_eff_module_path)
            surface_eff = importlib.util.module_from_spec(spec_surface_eff)
            spec_surface_eff.loader.exec_module(surface_eff)
            Surface_Noise = surface_eff.Surface_Noise
            
            print("Successfully imported render and surface_eff modules with importlib")
        else:
            raise ImportError(f"Could not find required modules at {geometry_module_path} or {surface_eff_module_path}")
    
    # Register cleanup function to ensure renderer is properly shut down on exit
    # This is for the worker process renderer, separate from node profile saving.
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
                "quality": (["low", "medium", "high", "ultra"], {"default": "medium"}), # Added quality
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
                "deformation_type": (["none", "perlin", "simplex"], {"default": "none"}),
                "deformation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deformation_noise_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "deformation_noise_octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "deformation_noise_persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deformation_noise_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "deformation_noise_seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "geometry"
    
    def __init__(self):
        self._destroyed = False
        self._profiler = cProfile.Profile() # Initialize profiler instance
        self._profiler_active = False
        _active_geometry_nodes.append(self)
        print(f"[GeometryRenderNode {id(self)}] Initialized, profiler created, and added to active list.")
    
    def dump_profile(self):
        # This method contains the logic to save the profile
        if hasattr(self, '_profiler') and self._profiler and self._profiler_active:
            print(f"[GeometryRenderNode {id(self)}] Dumping profile...")
            self._profiler.disable()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # current_file_path is defined at module level
            profile_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))) # PIC-BitalinoComfy
            profile_filename = f"geometry_node_profile_instance_{id(self)}_{timestamp}.prof"
            profile_path = os.path.join(profile_dir, profile_filename)

            try:
                self._profiler.dump_stats(profile_path)
                print(f"[GeometryRenderNode {id(self)}] Profiling data saved to {profile_path}")
            except Exception as e:
                print(f"[GeometryRenderNode {id(self)}] Error saving profile to {profile_path}: {e}")
            finally:
                self._profiler_active = False # Mark as not active after saving
        else:
            status_profiler = "exists" if hasattr(self, '_profiler') and self._profiler else "None"
            status_active = self._profiler_active if hasattr(self, '_profiler_active') else "N/A"
            print(f"[GeometryRenderNode {id(self)}] Profile dump skipped. Profiler: {status_profiler}, Active: {status_active}")

    def __del__(self):
        print(f"[GeometryRenderNode {id(self)}] __del__ called.")
        if not self._destroyed:
            self._cleanup()
    
    def _cleanup(self):
        print(f"[GeometryRenderNode {id(self)}] _cleanup called.")
        if not self._destroyed: # Ensure cleanup runs once per instance
            self._destroyed = True
            
            self.dump_profile() # Save profile data for this instance

            # Global renderer cleanup (for the worker process)
            # This is registered with atexit separately but can also be called here if appropriate.
            # However, render3d_comfy.cleanup_renderer() is already registered with atexit.
            # Calling it here might be redundant or lead to issues if called multiple times.
            # Let's rely on its own atexit registration.
            # print(f"[GeometryRenderNode {id(self)}] Skipping redundant global renderer cleanup call here.")
            
            # Remove from global list
            if self in _active_geometry_nodes:
                _active_geometry_nodes.remove(self)
                print(f"[GeometryRenderNode {id(self)}] Removed from active list.")
            else:
                print(f"[GeometryRenderNode {id(self)}] Instance not found in active list during cleanup.")
        else:
            print(f"[GeometryRenderNode {id(self)}] _cleanup skipped, already destroyed.")


    def render(self, object_type, center_x, center_y, center_z, size, rotation_deg_x, rotation_deg_y, rotation_deg_z, z_distance, img_size, color, quality="medium", deformation_type="none", deformation_strength=0.0, deformation_noise_scale=1.0, deformation_noise_octaves=4, deformation_noise_persistence=0.5, deformation_noise_lacunarity=2.0, deformation_noise_seed=0): # Added quality
        """Render the geometry with process isolation"""
        
        if not self._profiler_active:
            # self._profiler is already initialized in __init__
            if self._profiler: # Check if profiler object exists
                self._profiler.enable()
                self._profiler_active = True
                print(f"[GeometryRenderNode {id(self)}] Profiler enabled. Profile will be saved on exit/cleanup.")
            else:
                print(f"[GeometryRenderNode {id(self)}] Profiler object does not exist, cannot enable.")

        # Camera setup: looking at origin from +z
        camera_position = [(0, 0, z_distance), (0, 0, 0), (0, 1, 0)]
        
        #print(f"[GeometryRenderNode] Rendering {object_type} at ({center_x}, {center_y}, {center_z}) with size {size}")
        #print(f"[GeometryRenderNode] Rotation: ({rotation_deg_x}, {rotation_deg_y}, {rotation_deg_z}), color: {color}")
        #print(f"[GeometryRenderNode] Output image size: {img_size}x{img_size}")
        
        # Track render attempts for better error handling
        max_attempts = 2
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Try rendering with process isolation
                result = self._render_with_renderer(
                    None, object_type, center_x, center_y, center_z, size,
                    rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color,
                    quality=quality, # Pass quality
                    img_size=img_size, background='white', show_edges=True,
                    force_recreate=(attempt > 0),  # Force recreation on retry
                    deformation_type=deformation_type, # Pass through
                    deformation_strength=deformation_strength, # Pass through
                    deformation_noise_scale=deformation_noise_scale,
                    deformation_noise_octaves=deformation_noise_octaves,
                    deformation_noise_persistence=deformation_noise_persistence,
                    deformation_noise_lacunarity=deformation_noise_lacunarity,
                    deformation_noise_seed=deformation_noise_seed
                )
                
                # If we got a valid result, return it
                if result is not None:
                    if torch.is_tensor(result):
                        pass
                        #print(f"[GeometryRenderNode] Successfully rendered image: shape={tuple(result.shape)}, dtype={result.dtype}")
                    return result
                
                # If we get here, rendering failed but returned None - try again
                print(f"[GeometryRenderNode {id(self)}] Render attempt {attempt+1}/{max_attempts} failed with None result, retrying...")
                attempt += 1
                
            except Exception as e:
                print(f"[GeometryRenderNode {id(self)}] Process-isolated rendering failed on attempt {attempt+1}/{max_attempts}: {e}")
                import traceback
                traceback.print_exc()
                
                # Only retry once
                if attempt < max_attempts - 1:
                    print("Retrying with fresh renderer...")
                    attempt += 1
                    # Force cleanup before retry
                    try:
                        cleanup_renderer() # This is the global worker cleanup
                    except:
                        pass
                else:
                    # We've tried enough, return fallback
                    break
        
        # If we got here, all attempts failed
        #print(f"All {max_attempts} render attempts failed, returning fallback image")
        
        return self._create_fallback_tensors(img_size)
    
    def _render_with_renderer(self, renderer, object_type, center_x, center_y, center_z, size, 
                            rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color, 
                            quality="medium", # Added quality
                            img_size=None, background=None, show_edges=True, force_recreate=False,
                            deformation_type="none", deformation_strength=0.0,
                            deformation_noise_scale=1.0, deformation_noise_octaves=4, 
                            deformation_noise_persistence=0.5, deformation_noise_lacunarity=2.0, 
                            deformation_noise_seed=0): # Added new params
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
        #print(f"[GeometryRenderNode] Starting render with object_type={object_type}, center=({center_x}, {center_y}, {center_z}), size={size}, rotation=({rotation_deg_x}, {rotation_deg_y}, rotation_deg_z}), camera_position={camera_position}, color={color}")
        # Create geometry object
        if object_type == "sphere":
            geom = Sphere(center=(center_x, center_y, center_z), 
                        radius=size/2, 
                        quality=quality, # Use quality
                        rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))
        else:  # cube
            geom = Cube(center=(center_x, center_y, center_z), 
                      width=size, 
                      quality=quality, # Use quality
                      rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))

        deformed_this_frame = False
        # Apply surface deformation if specified
        if deformation_type != "none" and deformation_strength > 0.0:
            noise_generator = Surface_Noise(seed=deformation_noise_seed) # Use seed here
            # Parameters for noise can be exposed as inputs later if needed
            geom = noise_generator.apply_deformation(
                geometry_object=geom,
                deformation_type=deformation_type,
                strength=deformation_strength,
                scale=deformation_noise_scale, 
                octaves=deformation_noise_octaves, 
                persistence=deformation_noise_persistence,
                lacunarity=deformation_noise_lacunarity
            )
            # If deformation occurred, ensure the mesh is treated as a new object
            # by the renderer by creating a deep copy of it.
            if hasattr(geom, 'mesh') and geom.mesh is not None:
                geom.mesh = geom.mesh.copy(deep=True)
            deformed_this_frame = True
        
        # If deformation happened, also create a new wrapper geometry object (Sphere/Cube)
        # to ensure the renderer sees a "new" high-level object if it caches by id(geom_wrapper).
        if deformed_this_frame:
            current_mesh = geom.mesh
            current_vertices = geom.vertices
            
            if object_type == "sphere":
                new_geom_wrapper = Sphere(center=(center_x, center_y, center_z), 
                                     radius=size/2, 
                                     quality=quality, # Use quality
                                     rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))
            else:  # cube
                new_geom_wrapper = Cube(center=(center_x, center_y, center_z), 
                                   width=size, 
                                   quality=quality, # Use quality
                                   rotation=(rotation_deg_x, rotation_deg_y, rotation_deg_z))
            
            new_geom_wrapper.mesh = current_mesh # Assign the (already deep-copied) deformed mesh
            new_geom_wrapper.vertices = current_vertices # Assign the deformed vertices
            
            geom = new_geom_wrapper # Replace original geom with this new wrapper for rendering

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
        #print("[GeometryRenderNode] Processing outputs")
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
            elif color_img.max() > 1.0: # Check if max() can be called, e.g. not empty
                if np.any(color_img): # Ensure not all zeros before dividing by max
                     color_img = color_img.astype(np.float32) / color_img.max()
                else:
                    color_img = color_img.astype(np.float32)
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
    # import time # Already imported at the top
    import math
    import cv2
    # from PIL import Image # Not used

    node = GeometryRenderNode()

    # CHOOSE TEST MODE: "deformation" or "movement"
    test_mode = "d"  # Change to "movement" to test object transformations

    params = {
        "object_type": "cube", # Changed to sphere for better deformation visibility
        "quality": "medium", # Added quality
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size": 1, # Slightly larger
        "rotation_deg_x": 0.0,
        "rotation_deg_y": 0.0,
        "rotation_deg_z": 0.0,
        "z_distance": 7.0, # Adjusted for size
        "img_size": 512,
        "color": "#4682B4", # Steel Blue
        "deformation_type": "perlin", 
        "deformation_strength": 0.0, # Start with 0 strength
        "deformation_noise_scale": 1.0,
        "deformation_noise_octaves": 4,
        "deformation_noise_persistence": 0.5,
        "deformation_noise_lacunarity": 2.0,
        "deformation_noise_seed": 42 # Fixed seed for consistent testing
    }

    t_anim = 0.0
    print(f"Starting live render loop in '{test_mode}' mode. Press Ctrl+C to exit.")
    try:
        while True:
            if test_mode == "deformation":
                # Keep object static, animate deformation
                params["rotation_deg_x"] = 15.0 # Slight static rotation for better view
                params["rotation_deg_y"] = 15.0
                params["center_x"] = 0.0
                params["center_y"] = 0.0
                params["center_z"] = 0.0
                
                if int(t_anim) % 40 == 0:
                    new_type = "simplex" if int(t_anim / 20) % 2 == 0 else "perlin"
                    if params["deformation_type"] != new_type:
                        print(f"Switching deformation_type to {new_type} at t={t_anim:.1f}s")
                    params["deformation_type"] = new_type  # Switch noise type every 20s
                params["deformation_strength"] = (math.sin(t_anim * 0.7) + 1) / 2 * 0.8 # Cycle strength 0.0 to 0.4
                params["deformation_noise_scale"] = 0.5 + (math.cos(t_anim * 0.4) + 1) / 2 * 1.5 # Cycle scale 0.5 to 2.0
                params["deformation_noise_octaves"] = int(3 + (math.sin(t_anim * 0.2) +1) / 2 * 3) # Cycle octaves 3 to 6
                params["deformation_noise_seed"] = int(t_anim / 10) # Change seed every 10s

            elif test_mode == "movement":
                # Animate movement, static or no deformation
                params["rotation_deg_x"] = (t_anim * 20) % 360
                params["rotation_deg_y"] = (math.cos(t_anim * 0.5) * 30) % 360
                params["rotation_deg_z"] = (math.sin(t_anim * 0.5) * 30) % 360
                params["center_x"] = math.sin(t_anim * 0.3) * 1.5
                params["center_y"] = math.cos(t_anim * 0.3) * 1.5
                params["center_z"] = math.sin(t_anim * 0.2) * 1.0
                
                # Keep deformation static or off
                params["deformation_type"] = "perlin"
                params["deformation_strength"] = 0.1 # Small static deformation
                params["deformation_noise_scale"] = 0.8
                params["deformation_noise_octaves"] = 3
                # params["deformation_type"] = "none" # Or turn off deformation completely

            else: # Default to combined if mode is unknown (original behavior)
                params["rotation_deg_x"] = (math.sin(t_anim) * 45) % 360
                params["rotation_deg_y"] = (math.cos(t_anim) * 45) % 360
                params["rotation_deg_z"] = (t_anim * 30) % 360
                params["center_x"] = math.sin(t_anim) * 2.0
                params["center_y"] = math.cos(t_anim) * 2.0
                params["center_z"] = math.sin(t_anim * 0.5) * 2.0
                params["deformation_strength"] = (math.sin(t_anim * 0.7) + 1) / 2 * 0.3
                params["deformation_noise_scale"] = 0.5 + (math.cos(t_anim * 0.4) + 1) / 2 * 1.5

            color_img_tensor = node.render(**params)

            # Prepare for display
            color_img_np = color_img_tensor.cpu().numpy() if hasattr(color_img_tensor, 'cpu') else color_img_tensor
            if color_img_np.dtype != np.uint8:
                if not np.issubdtype(color_img_np.dtype, np.floating):
                    color_img_np = color_img_np.astype(np.float32)
                if np.any(color_img_np) and color_img_np.max() > 1e-6 : # Check if not all zeros
                    if color_img_np.max() > 1.0:
                        color_img_np = color_img_np / color_img_np.max()
                color_img_disp = (np.clip(color_img_np, 0, 1) * 255).astype(np.uint8)
            else:
                color_img_disp = color_img_np

            if color_img_disp.ndim == 4 and color_img_disp.shape[0] == 1:
                color_img_disp = color_img_disp[0]
            if color_img_disp.ndim == 2:
                color_img_disp = cv2.cvtColor(color_img_disp, cv2.COLOR_GRAY2BGR)
            elif color_img_disp.shape[2] == 1:
                color_img_disp = cv2.cvtColor(color_img_disp, cv2.COLOR_GRAY2BGR)

            # Show live window
            cv2.imshow('3D Geometry Live Render', color_img_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t_anim += 0.05
            time.sleep(0.001)  # 1ms sleep

    except KeyboardInterrupt:
        print("Exiting live render loop.")
    finally: # Ensure cleanup is attempted for standalone mode
        if node:
            node._cleanup() # Call node's cleanup
        cv2.destroyAllWindows()
        # The atexit handlers should still run for profile saving if __main__ exits normally
        # or via sys.exit(). If KeyboardInterrupt is not caught gracefully by main Python interpreter,
        # atexit might not run. The explicit cleanup here is a fallback for the node instance.
        # The global _save_all_node_profiles_on_exit will also attempt to save.
