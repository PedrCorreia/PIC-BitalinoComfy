"""
Worker process for handling PyVista 3D rendering in a separate process.
This isolates OpenGL/CUDA context and resource management from the main process.
"""
import os
import sys
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import json
import time
import traceback
import signal
import gc
import hashlib

# Force headless mode for the worker process
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'
os.environ['VTK_SILENCE_WARNING'] = '1'

# Import PyVista in worker process only
import pyvista as pv
pv.OFF_SCREEN = True
pv.set_plot_theme("document")

class Render3DWorker:
    """Worker process implementation of the Render3D functionality"""
    
    def __init__(self):
        self.geometries = []
        self._geometry_cache = {}
        self.img_size = (512, 512)
        self.background = 'white'
        
        # Force VTK initialization immediately
        import vtk
        vtk.vtkObject()
        print("Render3D worker: VTK initialized")
        
        # Pre-create cube faces for reuse
        self.cube_faces = np.array([
            [4, 0, 1, 2, 3], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5], [4, 2, 3, 7, 6], [4, 3, 0, 4, 7]
        ])
        self.cube_faces_flat = np.hstack(self.cube_faces)
        
        # Numpy arrays for cube geometry
        self.base_cube_verts = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        self.identity_3x3 = np.eye(3, dtype=np.float32)
    
    def euler_matrix(self, rx, ry, rz):
        """Create rotation matrix from Euler angles"""
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        
        # Pre-compute trig functions
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Build matrices efficiently
        Rx = self.identity_3x3.copy()
        Rx[1, 1] = cos_rx; Rx[1, 2] = -sin_rx
        Rx[2, 1] = sin_rx; Rx[2, 2] = cos_rx
        
        Ry = self.identity_3x3.copy()
        Ry[0, 0] = cos_ry; Ry[0, 2] = sin_ry
        Ry[2, 0] = -sin_ry; Ry[2, 2] = cos_ry
        
        Rz = self.identity_3x3.copy()
        Rz[0, 0] = cos_rz; Rz[0, 1] = -sin_rz
        Rz[1, 0] = sin_rz; Rz[1, 1] = cos_rz
        
        return Rz @ Ry @ Rx
    
    def _create_sphere_mesh(self, center, radius, quality='medium'):
        """Create a sphere mesh"""
        quality_map = {'low': 8, 'medium': 16, 'high': 32}
        resolution = quality_map.get(quality, 16)
        
        return pv.Sphere(
            center=center, 
            radius=radius,
            theta_resolution=resolution, 
            phi_resolution=resolution
        )
    
    def _create_cube_mesh(self, center, width, rotation):
        """Create a cube mesh with the given parameters"""
        # Scale base vertices
        scaled_verts = self.base_cube_verts * width
        
        # Apply rotation
        if not np.allclose(rotation, [0, 0, 0]):
            R = self.euler_matrix(*rotation)
            rotated_verts = scaled_verts @ R.T
        else:
            rotated_verts = scaled_verts
        
        # Apply translation
        final_verts = rotated_verts + np.array(center)
        
        return pv.PolyData(final_verts, self.cube_faces_flat)
    
    def _hash_geometry(self, geom_data):
        """Create a hash of geometry properties for caching"""
        center = geom_data['center']
        rotation = geom_data['rotation']
        geom_type = geom_data['type']
        
        props = []
        props.append(str(center))
        props.append(str(rotation))
        props.append(geom_type)
        
        for k, v in sorted(geom_data['params'].items()):
            props.append(f"{k}:{v}")
                
        key = "|".join(props)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _process_geometries(self, geometries_data):
        """Process geometry data received from the parent process"""
        meshes_data = []
        
        for geom_data in geometries_data:
            # Create a hash key for caching
            hash_key = self._hash_geometry(geom_data)
            
            # Try to get geometry from cache first
            if hash_key in self._geometry_cache:
                mesh = self._geometry_cache[hash_key]
            else:
                # Create mesh if not in cache
                if geom_data['type'] == 'sphere':
                    mesh = self._create_sphere_mesh(
                        geom_data['center'], 
                        geom_data['params']['radius'],
                        geom_data.get('quality', 'medium')
                    )
                elif geom_data['type'] == 'cube':
                    mesh = self._create_cube_mesh(
                        geom_data['center'], 
                        geom_data['params']['width'],
                        geom_data['rotation']
                    )
                else:
                    continue
                    
                # Store in cache for future use
                self._geometry_cache[hash_key] = mesh
                
            meshes_data.append({
                'mesh': mesh, 
                'color': geom_data['color'], 
                'edge_color': geom_data['edge_color'], 
                'opacity': geom_data['opacity']
            })
        
        return meshes_data
    
    def render(self, img_size, background, geometries_data, camera_position=None, show_edges=True):
        """Render the 3D scene with the given parameters"""
        try:
            # Update instance properties
            self.img_size = img_size
            self.background = background
            
            # Ensure PyVista is in off-screen mode
            pv.OFF_SCREEN = True
            
            # Set up the plotter with explicit VTK parameters
            plotter_img_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
            plotter = pv.Plotter(window_size=plotter_img_size, off_screen=True)
            
            # Ensure VTK uses a dedicated render window
            plotter.renderer.SetUseDepthPeeling(True)
            plotter.renderer.SetMaximumNumberOfPeels(4)
            plotter.renderer.SetOcclusionRatio(0.0)
            
            # Process all geometries
            meshes_data = self._process_geometries(geometries_data)
            
            # Add each mesh to the plotter
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
            
            # Set background and camera position
            plotter.set_background(self.background)
            if camera_position:
                plotter.camera_position = camera_position
            
            # Render the scene
            plotter.show(auto_close=False)
            img = plotter.screenshot(None, transparent_background=True)
            
            # Process the image
            if img is not None and img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            if img is not None and img.shape[-1] == 4:
                img = img[..., :3]
            
            return img
            
        except Exception as e:
            print(f"Render3DWorker: Rendering failed with error: {e}")
            traceback.print_exc()
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

def worker_process(cmd_queue, result_queue):
    """Main worker process function that receives commands and sends results back"""
    try:
        print("Render3D worker process started")
        
        # Initialize the worker
        worker = Render3DWorker()
        
        # Signal handler for clean shutdown
        def handle_signal(signum, frame):
            print(f"Render3D worker received signal {signum}, shutting down")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        
        # Active shared memory segments to clean up on exit
        active_shm = []
        
        # Main worker loop
        while True:
            try:
                # Get command from the queue with timeout
                try:
                    cmd = cmd_queue.get(timeout=5.0)  # Add timeout to prevent blocking forever
                except mp.queues.Empty:
                    # No commands for 5 seconds, just continue waiting
                    continue
                
                if cmd['action'] == 'exit':
                    print("Render3D worker: Exit command received")
                    break
                    
                elif cmd['action'] == 'render':
                    # Extract rendering parameters
                    img_size = cmd.get('img_size', (512, 512))
                    background = cmd.get('background', 'white')
                    geometries_data = cmd.get('geometries', [])
                    camera_position = cmd.get('camera_position', None)
                    show_edges = cmd.get('show_edges', True)
                    
                    # Perform the rendering
                    start_time = time.time()
                    img = worker.render(img_size, background, geometries_data, camera_position, show_edges)
                    end_time = time.time()
                    
                    # Create shared memory for the image result
                    if img is not None:
                        try:
                            # Clean up any old shared memory segments
                            for shm in active_shm:
                                try:
                                    shm.close()
                                    shm.unlink()
                                except:
                                    pass
                            active_shm = []
                            
                            # Create a new shared memory buffer
                            shm_name = f"render3d_img_{int(time.time() * 1000)}"
                            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img.nbytes)
                            active_shm.append(shm)
                            
                            # Copy the image data to shared memory
                            shm_array = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                            shm_array[:] = img[:]
                            
                            # Send result info back to parent process
                            result_queue.put({
                                'success': True,
                                'shm_name': shm_name,
                                'shape': img.shape,
                                'dtype': str(img.dtype),
                                'time_taken': end_time - start_time
                            })
                            print(f"Render3D worker: Image sent via shared memory {shm_name}")
                        except Exception as e:
                            print(f"Shared memory error: {e}")
                            traceback.print_exc()
                            result_queue.put({
                                'success': False,
                                'error': f"Shared memory error: {str(e)}"
                            })
                
                # Handle explicit shared memory cleanup command
                elif cmd['action'] == 'cleanup_shared_memory':
                    shm_name = cmd.get('shm_name')
                    if shm_name:
                        print(f"Render3D worker: Received cleanup request for {shm_name}")
                        # Find and clean up the specified shared memory segment
                        for i, shm in enumerate(active_shm):
                            if shm.name == shm_name:
                                try:
                                    print(f"Render3D worker: Unlinking shared memory {shm_name}")
                                    shm.close()
                                    shm.unlink()
                                    active_shm.pop(i)
                                    break
                                except Exception as e:
                                    print(f"Error unlinking shared memory {shm_name}: {e}")
                    else:
                        # If no specific name provided, clean up all segments
                        for shm in active_shm:
                            try:
                                shm.close()
                                shm.unlink()
                            except:
                                pass
                        active_shm = []
                        result_queue.put({
                            'success': False,
                            'error': "Rendering failed"
                        })
                
                # Add more commands as needed
                
            except Exception as e:
                print(f"Worker process error: {e}")
                traceback.print_exc()
                # Always send a response to avoid hanging the parent process
                try:
                    result_queue.put({
                        'success': False,
                        'error': str(e)
                    })
                except:
                    pass
    
    except Exception as e:
        print(f"Fatal worker error: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup any active shared memory
        for shm in active_shm:
            try:
                shm.close()
                shm.unlink()
            except:
                pass
        
        print("Render3D worker process exiting")
        sys.exit(0)
