import pyvista as pv
import numpy as np
import torch
import cv2
import hashlib
import os
import gc
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import time
import json
import subprocess
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
            gc.collect()
    
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
        """Optimized geometry processing with caching"""
        meshes_data = []
        
        for geom, color, edge_color, opacity in self.geometries:
            if not isinstance(geom, Geometry3D):
                continue
                
            # Create a hash key for caching
            hash_key = self._hash_geometry(geom)
            
            # Try to get geometry from cache first
            if hash_key in self._geometry_cache:
                mesh = self._geometry_cache[hash_key]
            else:
                # Create mesh if not in cache
                if 'radius' in geom.params:  # Sphere
                    mesh = self._create_sphere_mesh(geom)
                elif 'width' in geom.params:  # Cube
                    if self.use_gpu:
                        mesh = self._create_cube_mesh_gpu(geom)
                    else:
                        mesh = self._create_cube_mesh_cpu(geom)
                else:
                    continue
                    
                # Store in cache for future use
                self._geometry_cache[hash_key] = mesh
                
            meshes_data.append({
                'mesh': mesh, 'color': color, 'edge_color': edge_color, 'opacity': opacity
            })
        
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
                    'center': geom.center,
                    'rotation': geom.rotation,
                    'color': color,
                    'edge_color': edge_color,
                    'opacity': opacity,
                    'params': geom.params,
                }
                
                # Determine geometry type
                if 'radius' in geom.params:
                    geom_data['type'] = 'sphere'
                    if hasattr(geom, 'quality'):
                        geom_data['quality'] = geom.quality
                elif 'width' in geom.params:
                    geom_data['type'] = 'cube'
                else:
                    continue
                    
                geometries_data.append(geom_data)
            
            # Clear the result queue first to avoid reading old results
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except:
                    break
                    
            # Print confirmation of render command
            print(f"Sending render command to worker (geometries: {len(geometries_data)})")
                
            # Send render command to worker process
            self._cmd_queue.put({
                'action': 'render',
                'img_size': self.img_size,
                'background': self.background,
                'geometries': geometries_data,
                'camera_position': camera_position,
                'show_edges': show_edges
            })
            
            # Wait for result with timeout but process UI events periodically
            result = None
            start_time = time.time()
            timeout = 5  # Reduced timeout to 5 seconds for better UI responsiveness
            
            try:
                print("Waiting for worker result...")
                # Use a non-blocking approach with shorter timeouts to allow UI updates
                while time.time() - start_time < timeout:
                    try:
                        # Try to get result with a shorter timeout
                        result = self._result_queue.get(timeout=0.01)  # Shorter timeout for more UI updates
                        print(f"Got worker result: success={result.get('success', False)}")
                        break
                    except queue.Empty:
                        # If queue is empty, yield control back to UI thread
                        time.sleep(0.005)  # Very short sleep for more responsive UI
                        
                        # Check worker process health every half second
                        if time.time() - start_time > 0.5 and (time.time() - start_time) % 0.5 < 0.01:
                            if not self._worker_process.is_alive():
                                print("Worker process died while waiting for results")
                                raise RuntimeError("Worker process died unexpectedly")
                        continue
                    except Exception as e:
                        print(f"Error getting result: {e}")
                        raise
                
                # If we got here without a result, we timed out
                if result is None:
                    print("Timeout waiting for worker result - restarting worker process")
                    # Force reset the worker process
                    self._stop_worker_process()
                    self._start_worker_process()
                    raise TimeoutError("Timeout waiting for worker result")
                    
            except Exception as e:
                print(f"Error waiting for render result: {e}")
                # If result is not received in time, restart worker
                self._stop_worker_process()
                self._start_worker_process()
                return None
            
            if not result.get('success', False):
                print(f"Worker render failed: {result.get('error', 'Unknown error')}")
                return None
            
            # Get image from shared memory
            shm_name = result.get('shm_name')
            shape = result.get('shape')
            dtype_str = result.get('dtype')
            
            if shm_name and shape and dtype_str:
                shm = None
                try:
                    # Clean up any previous shared memory first to avoid resource leaks
                    self._cleanup_shared_memory()
                    
                    # Access the shared memory segment with a timeout approach
                    print(f"Accessing shared memory: {shm_name}")
                    
                    # Try to access shared memory with retries for robustness
                    start_access_time = time.time()
                    max_retries = 3
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            shm = shared_memory.SharedMemory(name=shm_name)
                            break  # Success, exit retry loop
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"Failed to access shared memory after {max_retries} attempts: {e}")
                                raise
                            print(f"Retry {retry_count}/{max_retries} accessing shared memory: {e}")
                            time.sleep(0.1)  # Short delay before retry
                    
                    self._active_shm = shm  # Save reference for cleanup later
                    
                    # Create numpy array using the shared memory buffer
                    dtype = np.dtype(dtype_str.replace("'", ""))
                    img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                    
                    # Create a copy IMMEDIATELY to avoid issues when shared memory is released
                    img = np.copy(img)
                    
                    # Save to output file if requested
                    if output and img is not None:
                        cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        print(f"3D Render saved to {output}")
                    
                    # Send a cleanup command to the worker to explicitly unlink the shared memory
                    # Do this immediately after copying the data
                    try:
                        self._cmd_queue.put({
                            'action': 'cleanup_shared_memory',
                            'shm_name': shm_name
                        })
                    except Exception as cleanup_e:
                        print(f"Failed to send cleanup command: {cleanup_e}")
                    
                    # Close our handle immediately
                    if shm is not None:
                        try:
                            print(f"Closing shared memory: {shm_name}")
                            shm.close()
                            self._active_shm = None  # Clear reference
                        except Exception as close_e:
                            print(f"Error closing shared memory: {close_e}")
                    
                    # Explicitly force garbage collection to ensure resources are freed
                    gc.collect()
                    
                    return img
                    
                except Exception as e:
                    print(f"Error accessing shared memory: {e}")
                    
                    # Try to clean up any partially initialized resources
                    if shm is not None:
                        try:
                            shm.close()
                        except:
                            pass
                    
                    # Force garbage collection on error
                    self._active_shm = None
                    gc.collect()
                    
                    # Return None to indicate failure
                    return None
                finally:
                    # Final cleanup in case anything was missed
                    if shm is not None:
                        try:
                            shm.close()
                            self._active_shm = None
                        except:
                            pass
            
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
