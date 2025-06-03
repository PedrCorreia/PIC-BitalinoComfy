import sys
import os
# Dynamically add project root and PIC-BitalinoComfy to sys.path for standalone script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
pic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if pic_root not in sys.path:
    sys.path.insert(0, pic_root)

import numpy as np
import torch

try:
    from src.geometry.render3d_isolated import render3d_color_subprocess
    from src.geometry.geom import Sphere, Cube
except ImportError as e:
    print("[ERROR] Could not import from src.geometry. sys.path is:", sys.path)
    print("[ERROR] Exception:", e)
    print("[ERROR] Make sure you are running this script from a location where the project root is in sys.path and that src/geometry exists.")
    raise

class GeometryRenderNode:
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

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image","depth_tensor")
    FUNCTION = "render"
    CATEGORY = "geometry"

    def render(self, object_type, center_x, center_y, center_z, size, rotation_deg_x, rotation_deg_y, rotation_deg_z, z_distance, img_size, color):
        # Camera setup: looking at origin from +z
        camera_position = [(0, 0, z_distance), (0, 0, 0), (0, 1, 0)]
        # Use isolated subprocess rendering for color only
        try:
            return self._render_with_renderer(
                None, object_type, center_x, center_y, center_z, size,
                rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color,
                img_size=img_size, background='white', show_edges=True
            )
        except Exception as e:
            print(f"Subprocess rendering failed: {e}")
            return self._create_fallback_tensors(img_size)
    
    def _render_with_renderer(self, renderer, object_type, center_x, center_y, center_z, size, 
                            rotation_deg_x, rotation_deg_y, rotation_deg_z, camera_position, color, img_size=None, background=None, show_edges=True):
        """Shared rendering logic for both GPU and CPU, now responsive to all node inputs"""
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
        # Use isolated subprocess rendering for color only
        geom_args = [(geom, color, 'black', 1.0)]
        # Use all node inputs for rendering
        color_img = render3d_color_subprocess(
            geom_args,
            img_size=img_size if img_size is not None else 512,
            background=background if background is not None else 'white',
            show_edges=show_edges,
            camera_position=camera_position
        )
        depth_img = None  # Depth is not supported in isolated mode
        return self._process_outputs(color_img, depth_img)
    
    def _process_outputs(self, color_img, depth_img):
        """Process rendered outputs into ComfyUI-compatible tensors with proper negative strides handling"""
        
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
        
        # Process depth image
        if depth_img is not None:
            # Convert to numpy array if needed
            if torch.is_tensor(depth_img):
                depth_img = depth_img.detach().cpu().numpy()
            
            # Ensure array is contiguous and make a copy to fix negative strides
            depth_img = np.ascontiguousarray(depth_img.copy())
            
            # Normalize depth values to [0, 1]
            if depth_img.max() > 1.0:
                depth_img = depth_img.astype(np.float32) / depth_img.max()
            else:
                depth_img = depth_img.astype(np.float32)
            
            # Convert to tensor
            depth_tensor = torch.from_numpy(depth_img).float()
            
            # Ensure proper shape: [1, H, W, 1] for depth
            if len(depth_tensor.shape) == 2:  # [H, W]
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(-1)  # -> [1, H, W, 1]
            elif len(depth_tensor.shape) == 3:
                if depth_tensor.shape[0] == 1:  # [1, H, W]
                    depth_tensor = depth_tensor.unsqueeze(-1)  # -> [1, H, W, 1]
                elif depth_tensor.shape[2] == 1:  # [H, W, 1]
                    depth_tensor = depth_tensor.unsqueeze(0)  # -> [1, H, W, 1]
                else:  # [H, W, C] where C > 1, take first channel
                    depth_tensor = depth_tensor[:, :, 0:1].unsqueeze(0)  # -> [1, H, W, 1]
        else:
            # Fallback: create zero depth
            depth_tensor = torch.zeros(1, 512, 512, 1, dtype=torch.float32)
        
        return color_tensor, depth_tensor
    
    def _create_fallback_tensors(self, img_size):
        """Create fallback tensors when rendering fails"""
        color_tensor = torch.zeros(1, img_size, img_size, 3, dtype=torch.float32)
        depth_tensor = torch.zeros(1, img_size, img_size, 1, dtype=torch.float32)
        return color_tensor, depth_tensor


if __name__ == "__main__":
    import cv2
    from PIL import Image
    
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
    
    print(f"Rendering color and depth with params: {params}")
    color_img, depth_tensor = node.render(**params)
    
    # Prepare color image - convert tensor to numpy first
    color_img_np = color_img.cpu().numpy() if hasattr(color_img, 'cpu') else color_img
    if color_img_np.dtype != np.uint8:
        color_img_disp = (np.clip(color_img_np, 0, 1) * 255).astype(np.uint8)
    else:
        color_img_disp = color_img_np
    
    # Remove batch dimension if present
    if color_img_disp.ndim == 4 and color_img_disp.shape[0] == 1:
        color_img_disp = color_img_disp[0]
    
    # Prepare depth image (grayscale)
    depth_np = depth_tensor.cpu().numpy()[0]  # (H, W, 1)
    depth_gray = (depth_np[..., 0] * 255).astype(np.uint8)  # Use one channel for grayscale
    
    # Stack color and depth images side by side
    if color_img_disp.ndim == 2:
        color_img_disp = cv2.cvtColor(color_img_disp, cv2.COLOR_GRAY2BGR)
    if len(depth_gray.shape) == 2:
        depth_gray_3c = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    else:
        depth_gray_3c = depth_gray
    
    # Resize if needed to match heights
    if color_img_disp.shape[0] != depth_gray_3c.shape[0]:
        h = min(color_img_disp.shape[0], depth_gray_3c.shape[0])
        color_img_disp = cv2.resize(color_img_disp, (color_img_disp.shape[1], h))
        depth_gray_3c = cv2.resize(depth_gray_3c, (depth_gray_3c.shape[1], h))
    
    side_by_side = np.concatenate([color_img_disp, depth_gray_3c], axis=1)
    cv2.imshow('3D Color (left) and Depth (right)', side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save both images
    Image.fromarray(color_img_disp).save("test_color_render.png")
    Image.fromarray(depth_gray).save("test_depth_render.png")
    Image.fromarray(side_by_side).save("test_color_and_depth_side_by_side.png")
    print("Saved color render to test_color_render.png")
    print("Saved depth render to test_depth_render.png")
    print("Saved side-by-side image to test_color_and_depth_side_by_side.png")
    print("Done.")
