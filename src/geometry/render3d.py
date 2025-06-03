import pyvista as pv
import numpy as np
import torch
import cv2
import hashlib
from .geom import Geometry3D, Cube

# Configure PyVista for headless/off-screen rendering and performance
pv.OFF_SCREEN = True
pv.set_plot_theme("document")
# Optimize PyVista settings for speed
if hasattr(pv, 'global_theme'):
    pv.global_theme.multi_samples = 1  # Reduce anti-aliasing for speed
    pv.global_theme.smooth_shading = False  # Disable smooth shading for speed
    pv.global_theme.depth_peeling.enable = False  # Disable depth peeling

# Cache computed meshes for reuse (speeds up repeated renders)
_mesh_cache = {}

class Render3D:
    """Simple 3D renderer with optimized performance"""
    
    def __init__(self, img_size=512, background='white', safe_mode=False):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.background = background
        self.geometries = []
        
        # Simple CUDA usage - just basic device selection, no interference with other scripts
        self.use_gpu = not safe_mode and torch.cuda.is_available()
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            print(f"Render3D: Using GPU acceleration on {torch.cuda.get_device_name()}")
            self._setup_gpu_matrices()
        else:
            self.device = torch.device('cpu')
            print("Render3D: Using CPU rendering")
            self._setup_cpu_matrices()
        
        # Cache the plotter for reuse - creates major speedup
        self._plotter = None
    
    def _setup_gpu_matrices(self):
        """Pre-create reusable matrices and tensors on GPU"""
        # Base cube vertices (centered at origin)
        self.base_cube_verts = torch.tensor([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], device=self.device, dtype=torch.float32)
        
        # Pre-create identity matrices for reuse
        self.identity_3x3 = torch.eye(3, device=self.device, dtype=torch.float32)
        self.identity_4x4 = torch.eye(4, device=self.device, dtype=torch.float32)
        
        # Pre-create commonly used values
        self.zeros_3 = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.ones_3 = torch.ones(3, device=self.device, dtype=torch.float32)
        
        # Cube faces (reusable)
        self.cube_faces = np.array([
            [4, 0, 1, 2, 3], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5], [4, 2, 3, 7, 6], [4, 3, 0, 4, 7]
        ])
        self.cube_faces_flat = np.hstack(self.cube_faces)
    
    def _setup_cpu_matrices(self):
        """Pre-create reusable matrices for CPU"""
        # Base cube vertices (centered at origin)
        self.base_cube_verts_cpu = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        # Pre-create identity matrices for reuse
        self.identity_3x3_cpu = np.eye(3, dtype=np.float32)
        
        # Cube faces (reusable)
        self.cube_faces = np.array([
            [4, 0, 1, 2, 3], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5], [4, 2, 3, 7, 6], [4, 3, 0, 4, 7]
        ])
        self.cube_faces_flat = np.hstack(self.cube_faces)
    
    def euler_matrix_gpu(self, rx, ry, rz):
        """Create Euler rotation matrix using pre-allocated GPU tensors"""
        # Convert to radians using pre-allocated tensors
        rx_rad = torch.deg2rad(torch.tensor(rx, device=self.device, dtype=torch.float32))
        ry_rad = torch.deg2rad(torch.tensor(ry, device=self.device, dtype=torch.float32))
        rz_rad = torch.deg2rad(torch.tensor(rz, device=self.device, dtype=torch.float32))
        
        # Pre-compute trig functions
        cos_rx, sin_rx = torch.cos(rx_rad), torch.sin(rx_rad)
        cos_ry, sin_ry = torch.cos(ry_rad), torch.sin(ry_rad)
        cos_rz, sin_rz = torch.cos(rz_rad), torch.sin(rz_rad)
        
        # Build rotation matrices using pre-allocated identity
        Rx = self.identity_3x3.clone()
        Rx[1, 1] = cos_rx; Rx[1, 2] = -sin_rx
        Rx[2, 1] = sin_rx; Rx[2, 2] = cos_rx
        
        Ry = self.identity_3x3.clone()
        Ry[0, 0] = cos_ry; Ry[0, 2] = sin_ry
        Ry[2, 0] = -sin_ry; Ry[2, 2] = cos_ry
        
        Rz = self.identity_3x3.clone()
        Rz[0, 0] = cos_rz; Rz[0, 1] = -sin_rz
        Rz[1, 0] = sin_rz; Rz[1, 1] = cos_rz
        
        # Efficient matrix multiplication chain
        return torch.matmul(torch.matmul(Rz, Ry), Rx)
    
    def euler_matrix_cpu(self, rx, ry, rz):
        """Create Euler rotation matrix using pre-allocated CPU arrays"""
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        
        # Pre-compute trig functions
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Build matrices efficiently
        Rx = self.identity_3x3_cpu.copy()
        Rx[1, 1] = cos_rx; Rx[1, 2] = -sin_rx
        Rx[2, 1] = sin_rx; Rx[2, 2] = cos_rx
        
        Ry = self.identity_3x3_cpu.copy()
        Ry[0, 0] = cos_ry; Ry[0, 2] = sin_ry
        Ry[2, 0] = -sin_ry; Ry[2, 2] = cos_ry
        
        Rz = self.identity_3x3_cpu.copy()
        Rz[0, 0] = cos_rz; Rz[0, 1] = -sin_rz
        Rz[1, 0] = sin_rz; Rz[1, 1] = cos_rz
        
        return Rz @ Ry @ Rx
    
    def add_geometry(self, geom: Geometry3D, color='white', edge_color='black', opacity=1.0):
        # Check cache key to avoid redundant geometry creation
        key = self._get_geometry_hash(geom, color)
        self.geometries.append((geom, color, edge_color, opacity, key))
    
    def _get_geometry_hash(self, geom, color):
        """Create a hash key for geometry caching"""
        if hasattr(geom, 'params'):
            # Create hash from essential properties
            key_parts = [
                str(geom.__class__.__name__),
                str(hash(str(geom.center))), 
                str(geom.params), 
                str(geom.rotation), 
                str(color)
            ]
            return hashlib.md5("_".join(key_parts).encode()).hexdigest()
        return None
    
    def _create_sphere_mesh(self, geom, cache_key=None):
        """Optimized sphere creation with caching"""
        global _mesh_cache
        
        # Check if mesh is in cache
        if cache_key and cache_key in _mesh_cache:
            return _mesh_cache[cache_key]
            
        # Adjust quality for performance
        quality_map = {'low': 6, 'medium': 10, 'high': 20}
        resolution = quality_map.get(getattr(geom, 'quality', 'medium'), 10)
        
        mesh = pv.Sphere(
            center=geom.center, 
            radius=geom.params['radius'],
            theta_resolution=resolution, 
            phi_resolution=resolution
        )
        
        # Cache the result
        if cache_key:
            _mesh_cache[cache_key] = mesh
            
            # Limit cache size
            if len(_mesh_cache) > 100:
                # Remove random items when cache gets too large
                for _ in range(10):
                    _mesh_cache.pop(next(iter(_mesh_cache)), None)
                
        return mesh
    
    def _create_cube_mesh_gpu(self, geom, cache_key=None):
        """Optimized GPU cube creation with caching"""
        global _mesh_cache
        
        # Check if mesh is in cache
        if cache_key and cache_key in _mesh_cache:
            return _mesh_cache[cache_key]
            
        width = geom.params['width']
        
        # Scale base vertices efficiently
        scaled_verts = self.base_cube_verts * width
        
        # Apply rotation using optimized matrix
        if not np.allclose(geom.rotation, [0, 0, 0]):
            R = self.euler_matrix_gpu(*geom.rotation)
            rotated_verts = torch.matmul(scaled_verts, R.T)
        else:
            rotated_verts = scaled_verts
        
        # Apply translation efficiently
        center_tensor = torch.tensor(geom.center, device=self.device, dtype=torch.float32)
        final_verts = rotated_verts + center_tensor
        
        # Convert to numpy for PyVista (optimized)
        verts_np = final_verts.detach().cpu().numpy()
        mesh = pv.PolyData(verts_np, self.cube_faces_flat)
        
        # Cache the result
        if cache_key:
            _mesh_cache[cache_key] = mesh
            
            # Limit cache size
            if len(_mesh_cache) > 100:
                # Remove random items when cache gets too large
                for _ in range(10):
                    _mesh_cache.pop(next(iter(_mesh_cache)), None)
        
        return mesh
    
    def _create_cube_mesh_cpu(self, geom, cache_key=None):
        """Optimized CPU cube creation with caching"""
        global _mesh_cache
        
        # Check if mesh is in cache
        if cache_key and cache_key in _mesh_cache:
            return _mesh_cache[cache_key]
            
        width = geom.params['width']
        
        # Scale base vertices efficiently
        scaled_verts = self.base_cube_verts_cpu * width
        
        # Apply rotation using optimized matrix
        if not np.allclose(geom.rotation, [0, 0, 0]):
            R = self.euler_matrix_cpu(*geom.rotation)
            rotated_verts = scaled_verts @ R.T
        else:
            rotated_verts = scaled_verts
        
        # Apply translation efficiently
        final_verts = rotated_verts + np.array(geom.center)
        
        mesh = pv.PolyData(final_verts, self.cube_faces_flat)
        
        # Cache the result
        if cache_key:
            _mesh_cache[cache_key] = mesh
            
            # Limit cache size
            if len(_mesh_cache) > 100:
                # Remove random items when cache gets too large
                for _ in range(10):
                    _mesh_cache.pop(next(iter(_mesh_cache)), None)
        
        return mesh
    
    def _process_geometries(self):
        """Optimized geometry processing with caching"""
        meshes_data = []
        
        for geom, color, edge_color, opacity, cache_key in self.geometries:
            if not isinstance(geom, Geometry3D):
                continue
            
            if 'radius' in geom.params:  # Sphere
                mesh = self._create_sphere_mesh(geom, cache_key)
            elif 'width' in geom.params:  # Cube
                if self.use_gpu:
                    mesh = self._create_cube_mesh_gpu(geom, cache_key)
                else:
                    mesh = self._create_cube_mesh_cpu(geom, cache_key)
            else:
                continue
                
            meshes_data.append({
                'mesh': mesh, 'color': color, 'edge_color': edge_color, 'opacity': opacity
            })
        
        return meshes_data
    
    def render(self, output=None, camera_position=None, show_edges=True, **kwargs):
        """Optimized rendering method"""
        plotter_img_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        
        # Reuse plotter if possible
        if self._plotter is None:
            pv.OFF_SCREEN = True
            self._plotter = pv.Plotter(window_size=plotter_img_size, off_screen=True)
        else:
            self._plotter.clear() 
        
        plotter = self._plotter
        plotter.set_background(self.background)
        
        # Process geometries with caching
        meshes_data = self._process_geometries()
        
        # Simplified rendering settings for better performance
        specular = 0.5  # Lower for faster rendering
        specular_power = 50  # Lower for faster rendering
        
        for mesh_info in meshes_data:
            plotter.add_mesh(
                mesh_info['mesh'],
                color=mesh_info['color'],
                show_edges=show_edges,
                edge_color=mesh_info['edge_color'],
                opacity=mesh_info['opacity'],
                specular=specular,
                specular_power=specular_power,
                smooth_shading=False  # Disable for performance
            )
        
        if camera_position:
            plotter.camera_position = camera_position
        
        plotter.show(auto_close=False)
        img = plotter.screenshot(None, transparent_background=True)
        
        # Don't close the plotter, reuse it for next time
        # plotter.close()
        
        if img is not None:
            img = self._process_image(img)
        
        if output and img is not None:
            cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"3D Render saved to {output}")
        
        return img
    
    def render_depth(self, camera_position=None, **kwargs):
        """Depth rendering disabled (returns None)"""
        return None
    
    def render_both(self, camera_position=None, show_edges=True, **kwargs):
        """Render both color and depth (returns color, None)"""
        color = self.render(camera_position=camera_position, show_edges=show_edges, **kwargs)
        return color, None
    
    def _process_image(self, img):
        """Optimized image processing"""
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        
        if img.shape[-1] == 4:
            img = img[..., :3]
        
        return img
    
    def _process_depth(self, depth_buffer):
        """Optimized depth processing"""
        if self.use_gpu:
            return self._process_depth_gpu(depth_buffer)
        else:
            return self._process_depth_cpu(depth_buffer)
    
    def _process_depth_gpu(self, depth_buffer):
        """GPU-optimized depth processing"""
        try:
            # Convert to tensor efficiently
            depth_tensor = torch.from_numpy(depth_buffer).to(self.device, dtype=torch.float32)
            
            # Handle invalid values
            mask = torch.isfinite(depth_tensor)
            if mask.any():
                valid_values = depth_tensor[mask]
                min_depth = torch.min(valid_values)
                max_depth = torch.max(valid_values)
                
                if max_depth > min_depth:
                    # Normalize in-place
                    depth_tensor.sub_(min_depth).div_(max_depth - min_depth)
                    depth_tensor[~mask] = 0
                else:
                    depth_tensor.zero_()
            else:
                depth_tensor.zero_()
            
            return depth_tensor.cpu().numpy()
            
        except Exception as e:
            print(f"GPU depth processing failed: {e}, falling back to CPU")
            return self._process_depth_cpu(depth_buffer)
    
    def _process_depth_cpu(self, depth_buffer):
        """CPU depth processing"""
        valid_mask = np.isfinite(depth_buffer)
        if valid_mask.any():
            valid_depths = depth_buffer[valid_mask]
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            
            if max_depth > min_depth:
                normalized = (depth_buffer - min_depth) / (max_depth - min_depth)
                normalized[~valid_mask] = 0
                return normalized
        
        return np.zeros_like(depth_buffer)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up plotter when done"""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        return False

# End of file
