import pyvista as pv
import numpy as np
import torch
import cv2
import hashlib
import os
import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import importlib.util
import threading
import queue
from .geom import Geometry3D, Cube

# Configure PyVista for headless/off-screen rendering and prevent conflicts
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'
os.environ['VTK_SILENCE_WARNING'] = '1'
pv.OFF_SCREEN = True
pv.set_plot_theme("document")

class Render3D:
    """3D renderer with isolation mechanisms to prevent OpenGL/CUDA context conflicts"""
    
    def __init__(self, img_size=512, background='white', safe_mode=True, process_isolation=True):
        """
        Initialize the 3D renderer
        
        Args:
            img_size: Size of the output image (int or tuple)
            background: Background color
            safe_mode: If True, use CPU rendering instead of GPU
            process_isolation: If True, use a separate process for rendering
        """
        # Force VTK to initialize here to avoid race conditions if we're not using process isolation
        if not process_isolation:
            import vtk
            vtk.vtkObject()
        
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.background = background
        self.geometries = []
        self._geometry_cache = {}  # Cache for geometry meshes
        self.process_isolation = process_isolation
        
        # Worker process management
        self._worker_process = None
        self._cmd_queue = None
        self._result_queue = None
        self._active_shm = None  # Keep track of active shared memory segments
        
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
            
        # Start worker process if using process isolation
        if self.process_isolation:
            self._start_worker_process()
    
    def _start_worker_process(self):
        """Start a separate process for rendering to isolate OpenGL contexts"""
        try:
            # Clean up any existing process first
            if self._worker_process is not None:
                self._stop_worker_process()
            
            # Create communication queues
            self._cmd_queue = mp.Queue()
            self._result_queue = mp.Queue()
            
            # Find the worker module path
            worker_module_path = os.path.join(os.path.dirname(__file__), "render3d_worker_entry.py")
            if not os.path.exists(worker_module_path):
                print(f"Worker module not found at {worker_module_path}, falling back to in-process rendering")
                self.process_isolation = False
                return
                
            # Import the worker module
            worker_module_spec = importlib.util.spec_from_file_location("render3d_worker", worker_module_path)
            if worker_module_spec is None:
                print("Could not load worker module spec, falling back to in-process rendering")
                self.process_isolation = False
                return
                
            worker_module = importlib.util.module_from_spec(worker_module_spec)
            worker_module_spec.loader.exec_module(worker_module)
            
            # Start the worker process with a non-daemon process
            # Using daemon=False ensures cleaner shutdown
            self._worker_process = mp.Process(
                target=worker_module.worker_process,
                args=(self._cmd_queue, self._result_queue),
                daemon=False
            )
            self._worker_process.start()
            
            # Wait briefly to ensure worker starts up
            time.sleep(0.1)
            
            if self._worker_process.is_alive():
                print(f"Started Render3D worker process (PID: {self._worker_process.pid})")
            else:
                print("Worker process failed to start")
                self.process_isolation = False
                
        except Exception as e:
            print(f"Failed to start worker process: {e}")
            import traceback
            traceback.print_exc()
            self.process_isolation = False
            
    def _stop_worker_process(self):
        """Stop the worker process if it's running"""
        if self._worker_process is not None and self._worker_process.is_alive():
            try:
                # Send exit command
                if self._cmd_queue is not None:
                    self._cmd_queue.put({'action': 'exit'})
                    
                # Wait for process to exit (with timeout)
                self._worker_process.join(timeout=2)
                
                # Force terminate if still running
                if self._worker_process.is_alive():
                    print("Worker process did not exit cleanly, terminating...")
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=1)
                    
                    # As a last resort, kill it
                    if self._worker_process.is_alive():
                        self._worker_process.kill()
            except Exception as e:
                print(f"Error stopping worker process: {e}")
            finally:
                self._worker_process = None
                
        # Clean up queues
        self._cmd_queue = None
        self._result_queue = None
        
        # Clean up any shared memory segments
        self._cleanup_shared_memory()
    
    def _cleanup_shared_memory(self):
        """Clean up any active shared memory segments"""
        if self._active_shm is not None:
            try:
                print(f"Cleaning up shared memory: {self._active_shm.name}")
                self._active_shm.close()
                try:
                    self._active_shm.unlink()
                except FileNotFoundError:
                    # This is ok, it means the segment was already removed
                    pass
            except Exception as e:
                print(f"Error cleaning up shared memory: {e}")
            self._active_shm = None
            
            # Force garbage collection to help release any remaining references
    
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
        self.base_cube_verts_cpu = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
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
        self.geometries.append((geom, color, edge_color, opacity))
        
    def clear_geometries(self):
        """Clear all geometries from the scene"""
        self.geometries = []
    
    def _create_sphere_mesh(self, geom):
        """Optimized sphere creation"""
        quality_map = {'low': 8, 'medium': 16, 'high': 32}
        resolution = quality_map.get(getattr(geom, 'quality', 'medium'), 16)
        
        return pv.Sphere(
            center=geom.center, 
            radius=geom.params['radius'],
            theta_resolution=resolution, 
            phi_resolution=resolution
        )
    
    def _create_cube_mesh_gpu(self, geom):
        """Optimized GPU cube creation using pre-allocated matrices"""
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
        return pv.PolyData(verts_np, self.cube_faces_flat)
    
    def _create_cube_mesh_cpu(self, geom):
        """Optimized CPU cube creation using pre-allocated matrices"""
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
        
        return pv.PolyData(final_verts, self.cube_faces_flat)
    
    def _hash_geometry(self, geom):
        """Create a hash of geometry properties for caching"""
        props = []
        props.append(str(geom.center))
        props.append(str(geom.rotation))
        
        if hasattr(geom, 'params'):
            for k, v in sorted(geom.params.items()):
                props.append(f"{k}:{v}")
                
        key = "|".join(props)
        return hashlib.md5(key.encode()).hexdigest()
        
    def _process_geometries(self):
        """Optimized geometry processing with caching and support for pre-defined meshes"""
        meshes_data = []
        
        for geom, color, edge_color, opacity in self.geometries:
            if not isinstance(geom, Geometry3D):
                # If geom is already a raw PyVista PolyData, we could potentially handle it here too.
                # For now, we expect Geometry3D objects from add_geometry.
                continue
                
            processed_mesh = None

            # Priority 1: Use geom.mesh if it's a valid, populated PyVista PolyData object.
            # This allows for pre-transformed or dynamically modified meshes to be rendered directly.
            # No caching is applied here, as these meshes are assumed to be externally managed.
            if hasattr(geom, 'mesh') and isinstance(geom.mesh, pv.PolyData) and geom.mesh.n_points > 0:
                processed_mesh = geom.mesh
                # Note: If geom.mesh.points are local, and geom.center/rotation are meant to be applied,
                # this direct usage bypasses that. For surface_eff.py, apply_deformation works on world coordinates.
            else:
                # Priority 2: Fallback to existing caching and generation logic
                # if geom.mesh is not a usable PolyData.
                hash_key = self._hash_geometry(geom) # Hash based on params, center, rotation
                
                if hash_key in self._geometry_cache:
                    processed_mesh = self._geometry_cache[hash_key]
                else:
                    # Create mesh if not in cache
                    new_mesh = None
                    if 'radius' in geom.params:  # Sphere
                        new_mesh = self._create_sphere_mesh(geom)
                    elif 'width' in geom.params:  # Cube
                        if self.use_gpu:
                            new_mesh = self._create_cube_mesh_gpu(geom)
                        else:
                            new_mesh = self._create_cube_mesh_cpu(geom)
                    else:
                        # Unknown geometry type based on params, skip
                        print(f"Render3D: Skipping geometry with unknown params: {geom.params}")
                        continue 
                    
                    if new_mesh:
                        self._geometry_cache[hash_key] = new_mesh # Cache the newly created mesh
                        processed_mesh = new_mesh
            
            if processed_mesh is not None:
                meshes_data.append({
                    'mesh': processed_mesh, 
                    'color': color, 
                    'edge_color': edge_color, 
                    'opacity': opacity
                })
            else:
                # Optional: Log if a geometry could not be processed into a mesh
                print(f"Render3D: Could not determine mesh for geometry: {geom}")
        
        return meshes_data
    
    def render(self, output=None, camera_position=None, show_edges=True, **kwargs):
        """
        Render the scene using either process isolation or in-process rendering
        """
        if self.process_isolation:
            return self._render_with_worker(output, camera_position, show_edges, **kwargs)
        else:
            return self._render_in_process(output, camera_position, show_edges, **kwargs)
        
    def _render_with_worker(self, output=None, camera_position=None, show_edges=True, **kwargs):
        """Render the scene using a separate worker process"""
        if self._worker_process is None or not self._worker_process.is_alive():
            print("Worker process not running, restarting...")
            # Make sure to clean up any existing worker process first
            self._stop_worker_process()
            self._start_worker_process()
            
            # If worker process still can't start, fall back to in-process rendering
            if self._worker_process is None or not self._worker_process.is_alive():
                print("Worker process couldn't be started, falling back to in-process rendering")
                return self._render_in_process(output, camera_position, show_edges, **kwargs)
        
        try:
            # Convert geometries to a serializable format
            geometries_data = []
            for geom, color, edge_color, opacity in self.geometries:
                if not isinstance(geom, Geometry3D):
                    continue
                    
                geom_data = {
                    'center': geom.center, # Still useful for potential fallback or context
                    'rotation': geom.rotation, # Still useful for potential fallback or context
                    'color': color,
                    'edge_color': edge_color,
                    'opacity': opacity,
                    'params': geom.params, # Original params
                }
                
                # Check if a pre-defined/deformed mesh exists and should be sent
                if hasattr(geom, 'mesh') and isinstance(geom.mesh, pv.PolyData) and geom.mesh.n_points > 0:
                    # Serialize mesh points and faces
                    # Ensure points are numpy arrays for consistent serialization
                    points = np.asarray(geom.mesh.points)
                    # Faces need to be in a flat format that PyVista can understand (e.g., [3, p1, p2, p3, 3, p4, p5, p6])
                    # or a list of lists/tuples if the worker reconstructs it carefully.
                    # pv.PolyData(points, faces_flat) is common.
                    # geom.mesh.faces is already in the flat format [n_points_face1, idx1, idx2, ..., n_points_face2, idxA, idxB, ...]
                    faces = np.asarray(geom.mesh.faces)
                    
                    geom_data['custom_mesh_data'] = {
                        'points': points.tolist(), # Convert to list for JSON serialization
                        'faces': faces.tolist()    # Convert to list for JSON serialization
                    }
                    geom_data['type'] = 'custom_mesh' # Indicate this is custom
                else:
                    # Determine geometry type for standard reconstruction if no custom mesh
                    if 'radius' in geom.params:
                        geom_data['type'] = 'sphere'
                        if hasattr(geom, 'quality'):
                            geom_data['quality'] = geom.quality
                    elif 'width' in geom.params:
                        geom_data['type'] = 'cube'
                    else:
                        print(f"Render3D: Skipping geometry with unknown params and no custom mesh: {geom.params}")
                        continue
                    
                geometries_data.append(geom_data)
            
            # Clear the result queue first to avoid reading old results
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except:
                    break
                    
            # Send render command to worker process
            # Send command to worker
            # print(f"Sending render command to worker (geometries: {len(geometries_data)})") # Commented out
            self._cmd_queue.put({
                'action': 'render',
                'img_size': self.img_size,
                'background': self.background,
                'geometries': geometries_data,
                'camera_position': camera_position,
                'show_edges': show_edges
            })
            
            # Wait for result from worker with a timeout
            # print("Waiting for worker result...") # Commented out
            try:
                result = self._result_queue.get(timeout=10.0)  # Increased timeout to 10s
            except Exception as e:
                print(f"Error getting result from worker: {e}")
                result = None  # Indicate failure for this render

            # print(f"Got worker result: success={result.get('success')}") # Commented out
            
            if result.get('success'):
                shm_name = result['shm_name']
                shape = result['shape']
                dtype_str = result['dtype']
                
                dtype = np.dtype(dtype_str)
                
                # print(f"Accessing shared memory: {shm_name}") # Commented out
                try:
                    # Connect to the existing shared memory
                    shm = shared_memory.SharedMemory(name=shm_name)
                    self._active_shm = shm  # Keep reference to close it later
                    
                    # Create numpy array using the shared memory buffer
                    img_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                    
                    # Create a copy IMMEDIATELY to avoid issues when shared memory is released
                    img_copy = img_array.copy()
                    
                    # print(f"Closing shared memory: {shm_name}") # Commented out
                    self._active_shm.close() # Close our handle to it
                    
                    # Request worker to unlink the shared memory
                    # print(f"Requesting worker to unlink shared memory: {shm_name}") # Commented out
                    self._cmd_queue.put({'action': 'cleanup_shared_memory', 'shm_name': shm_name})
                    self._active_shm = None # Mark as handled
                except Exception as e:
                    print(f"Error accessing shared memory: {e}")
                    img_copy = None
                
                # Save to output file if requested
                if output and img_copy is not None:
                    cv2.imwrite(output, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
                    print(f"3D Render saved to {output}")
                
                return img_copy
            else:
                print(f"Worker render failed: {result.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error in worker process rendering: {e}")
            return None
    
    def _render_in_process(self, output=None, camera_position=None, show_edges=True, **kwargs):
        """Simple rendering method (color only, depth ignored) with thorough cleanup"""
        try:
            # Ensure PyVista is in off-screen mode
            pv.OFF_SCREEN = True
            
            # Set up the plotter with explicit VTK parameters
            plotter_img_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
            plotter = pv.Plotter(window_size=plotter_img_size, off_screen=True)
            
            # Ensure VTK uses a dedicated render window
            plotter.renderer.SetUseDepthPeeling(True)
            plotter.renderer.SetMaximumNumberOfPeels(4)
            plotter.renderer.SetOcclusionRatio(0.0)
            
            meshes_data = self._process_geometries()
            
            for mesh_info in meshes_data:
                plotter.add_mesh(
                    mesh_info['mesh'],
                    color=mesh_info['color'],
                    show_edges=show_edges,
                    edge_color=mesh_info['edge_color'],
                    opacity=mesh_info['opacity'],
                    specular=1.0,
                    specular_power=100.0,
                    smooth_shading=True
                )
            
            plotter.set_background(self.background)
            if camera_position:
                plotter.camera_position = camera_position
            
            plotter.show(auto_close=False)
            img = plotter.screenshot(None, transparent_background=True)
            
            if img is not None:
                img = self._process_image(img)
            
            if output and img is not None:
                cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"3D Render saved to {output}")
            
            return img
            
        except Exception as e:
            print(f"Render3D: Rendering failed with error: {e}")
            return None
            
        finally:
            # Critical: ensure proper cleanup of VTK/PyVista resources
            if 'plotter' in locals():
                try:
                    plotter.close()
                    plotter.deep_clean()  # More thorough cleanup
                    del plotter
                except:
                    pass
                    
            # Force garbage collection to release GPU resources
            gc.collect()
    
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
    
    def cleanup(self):
        """Clean up resources, especially important for process isolation"""
        if self.process_isolation:
            self._stop_worker_process()
        
        # Clear caches and resources
        self._geometry_cache = {}
        self.geometries = []
        
        # Force garbage collection
        gc.collect()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.cleanup()
        return False
    
    def __del__(self):
        """Destructor to ensure worker process is terminated"""
        self.cleanup()

# End of file
