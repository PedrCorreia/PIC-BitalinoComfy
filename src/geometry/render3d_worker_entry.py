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
# pv.set_plot_theme("document") # Theme can be set per plotter if needed

class Render3DWorker:
    """Worker process implementation of the Render3D functionality"""
    
    def __init__(self):
        self.geometries = []
        self._geometry_cache = {}
        # self.img_size = (512, 512) # Will be set by _ensure_plotter_initialized_and_sized
        # self.background = 'white' # Will be set by _ensure_plotter_initialized_and_sized
        
        self.plotter = None
        self.current_plotter_img_size = None
        self.current_plotter_background = None

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
        # Ensure rotation is a tuple for consistent hashing
        rotation_tuple = tuple(geom_data['rotation']) if isinstance(geom_data['rotation'], list) else geom_data['rotation']
        geom_type = geom_data['type']
        
        props = []
        props.append(str(center))
        props.append(str(rotation_tuple))
        props.append(geom_type)
        
        for k, v in sorted(geom_data['params'].items()):
            props.append(f"{k}:{v}")
                
        key = "|".join(props)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _process_geometries(self, geometries_data):
        """Process geometry data received from the parent process"""
        meshes_data = []
        # print(f"Worker: Processing {len(geometries_data)} geometries") # Debug
        for geom_data in geometries_data:
            mesh = None # Initialize mesh to None

            # Priority 1: Reconstruct from custom_mesh_data if provided
            if geom_data.get('type') == 'custom_mesh' and 'custom_mesh_data' in geom_data:
                custom_data = geom_data['custom_mesh_data']
                points = np.array(custom_data['points'])
                faces_flat = np.array(custom_data['faces'])
                if points.ndim == 2 and points.shape[1] == 3 and faces_flat.ndim == 1:
                    try:
                        mesh = pv.PolyData(points, faces_flat.astype(np.int_))
                    except Exception as e:
                        print(f"Worker: Error reconstructing custom mesh: {e}")
                        traceback.print_exc()
                else:
                    print(f"Worker: Invalid custom mesh data format. Points shape: {points.shape}, Faces shape: {faces_flat.shape}")
            
            # Priority 2: Fallback to caching and standard generation if no valid custom mesh
            if mesh is None: # If custom mesh reconstruction failed or wasn't provided
                # Create a hash key for caching (only for standard geometries)
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
                            geom_data['rotation'] # Rotation is already a tuple (or list of 3 floats)
                        )
                    else:
                        pass # Continue to next geometry if type is unknown and not custom
                        
                    # Store in cache for future use (only if standard geometry was created)
                    if mesh is not None:
                        self._geometry_cache[hash_key] = mesh
            
            if mesh is not None:
                meshes_data.append({
                    'mesh': mesh, 
                    'color': geom_data['color'], 
                    'edge_color': geom_data['edge_color'], 
                    'opacity': geom_data['opacity']
                })
            # else:
                # print(f\"Worker: Mesh could not be created or reconstructed for geom_data: {geom_data.get('type')}\") # Removed this print
        
        return meshes_data

    def _ensure_plotter_initialized_and_sized(self, target_img_size, target_background):
        """Initializes or resizes the plotter if necessary."""
        target_img_size_tuple = target_img_size if isinstance(target_img_size, tuple) else (target_img_size, target_img_size)
        
        if self.plotter is None:
            # print(f"Worker: Initializing plotter with size {target_img_size_tuple} and background '{target_background}'") # Debug
            pv.set_plot_theme("document") # Set theme once
            self.plotter = pv.Plotter(window_size=target_img_size_tuple, off_screen=True)
            self.plotter.renderer.SetUseDepthPeeling(True)
            self.plotter.renderer.SetMaximumNumberOfPeels(4)
            self.plotter.renderer.SetOcclusionRatio(0.0)
            self.plotter.set_background(target_background)
            self.current_plotter_img_size = target_img_size_tuple
            self.current_plotter_background = target_background
        elif self.current_plotter_img_size != target_img_size_tuple:
            # print(f"Worker: Resizing plotter from {self.current_plotter_img_size} to {target_img_size_tuple}") # Debug
            self.plotter.window_size = target_img_size_tuple
            # self.plotter.render() # Update window size if needed, might not be necessary if screenshot handles it
            self.current_plotter_img_size = target_img_size_tuple
        
        if self.current_plotter_background != target_background:
            # print(f"Worker: Changing background from '{self.current_plotter_background}' to '{target_background}'") # Debug
            self.plotter.set_background(target_background)
            self.current_plotter_background = target_background

    def render(self, img_size, background, geometries_data, camera_position=None, show_edges=True):
        """Render the 3D scene with the given parameters using a persistent plotter."""
        try:
            # Ensure PyVista is in off-screen mode (should be set globally, but good to be sure)
            pv.OFF_SCREEN = True 
            
            self._ensure_plotter_initialized_and_sized(img_size, background)
            
            # Clear previous actors from the plotter
            self.plotter.clear_actors()
            # print(f"Worker: Cleared actors. Plotter has {len(self.plotter.renderer.actors)} actors.") # Debug

            # Process all geometries
            meshes_data = self._process_geometries(geometries_data)
            # print(f"Worker: Processed {len(meshes_data)} meshes to add.") # Debug
            
            # Add each mesh to the plotter
            for mesh_info in meshes_data:
                self.plotter.add_mesh(
                    mesh_info['mesh'],
                    color=mesh_info['color'],
                    show_edges=show_edges,
                    edge_color=mesh_info['edge_color'],
                    opacity=mesh_info['opacity'],
                    specular=1.0,
                    specular_power=100.0,
                    smooth_shading=True,
                    name=f"geom_{id(mesh_info['mesh'])}" # Give unique name to help with updates/clearing
                )
            # print(f"Worker: Added meshes. Plotter has {len(self.plotter.renderer.actors)} actors.") # Debug
            
            # Set camera position (must be done after adding meshes if auto_adjust is on)
            if camera_position:
                self.plotter.camera_position = camera_position
            else:
                # Optional: reset camera if no specific position is given, to frame the new objects
                self.plotter.reset_camera(bounds=self.plotter.renderer.ComputeVisiblePropBounds())


            # Render the scene - show() might not be needed if screenshot triggers render
            # self.plotter.show(auto_close=False) # This might re-open a window if not careful
            img = self.plotter.screenshot(None, transparent_background=True) # screenshot should trigger a render
            
            # Process the image
            if img is not None and img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            if img is not None and img.shape[-1] == 4:
                img = img[..., :3]
            
            return img
            
        except Exception as e:
            print(f"Render3DWorker: Rendering failed with error: {e}")
            traceback.print_exc()
            # Attempt to clean up plotter if an error occurs to prevent state issues
            if self.plotter:
                try:
                    self.plotter.clear_actors() # Clear actors to be safe
                except Exception as ce:
                    print(f"Render3DWorker: Error clearing actors during exception handling: {ce}")
            return None
            
        # finally: # Removed plotter.close() and gc.collect() from here
            # No per-render cleanup of plotter

    def close(self):
        """Properly close the plotter and perform cleanup when the worker is shutting down."""
        print("Render3DWorker: Close method called.") # Debug
        if self.plotter:
            try:
                print("Render3DWorker: Closing plotter.") # Debug
                self.plotter.close()
                self.plotter.deep_clean()
                del self.plotter
                self.plotter = None
            except Exception as e:
                print(f"Render3DWorker: Error during plotter close: {e}")
                traceback.print_exc()
        
        # Force garbage collection once at the end
        print("Render3DWorker: Performing final garbage collection.") # Debug
        gc.collect()

def worker_process(cmd_queue, result_queue):
    """Main worker process function that receives commands and sends results back"""
    worker = None # Initialize worker to None for robust finally block
    active_shm = [] # Moved here to be accessible in finally block

    try:
        print("Render3D worker process started")
        
        # Initialize the worker
        worker = Render3DWorker()
        
        # Signal handler for clean shutdown
        def handle_signal(signum, frame):
            print(f"Render3D worker received signal {signum}, initiating shutdown...")
            # We want to break the loop and let the finally block handle cleanup
            # Setting a flag or putting an exit message in the queue might be safer
            # For now, direct sys.exit might be okay if finally block is robust
            # To ensure worker.close() is called, we can call it here before exiting
            # if worker:
            #    worker.close() # This might be problematic if called from signal handler context
            # sys.exit(0) # This will skip the finally block in the main loop
            # Instead, we'll rely on the main loop exiting and the outer finally block.
            # To break the loop, we can send an exit command to itself if the queue is used by this thread.
            # Or, more simply, raise an exception that the main loop can catch to exit gracefully.
            raise SystemExit(f"Signal {signum} received")

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal) # Catch Ctrl+C in worker too
        
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
                    break # Exit the loop, finally block will handle cleanup
                    
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
                            active_shm = [] # Reset active_shm before creating a new one
                            
                            # Create a new shared memory buffer
                            shm_name = f"render3d_img_{int(time.time() * 1000)}_{os.getpid()}" # Added PID for uniqueness
                            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img.nbytes)
                            active_shm.append(shm) # Add to list for cleanup
                            
                            # Copy the image data to shared memory
                            shm_array = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                            shm_array[:] = img[:]
                            
                            # Send result info back to parent process
                            result_queue.put({
                                'success': True,
                                'shm_name': shm_name,
                                'shape': img.shape,
                                'dtype': str(img.dtype),
                                'time_taken': end_time - start_time,
                                'pid': os.getpid() # Send worker PID for debugging
                            })
                            # print(f"Render3D worker (PID {os.getpid()}): Image sent via shared memory {shm_name}") # Commented out
                        except Exception as e:
                            print(f"Shared memory error: {e}")
                            traceback.print_exc()
                            result_queue.put({
                                'success': False,
                                'error': f"Shared memory error: {str(e)}",
                                'pid': os.getpid()
                            })
                
                # Handle explicit shared memory cleanup command
                elif cmd['action'] == 'cleanup_shared_memory':
                    shm_name = cmd.get('shm_name')
                    if shm_name:
                        # print(f"Render3D worker: Received cleanup request for {shm_name}") # Commented out
                        # Find and clean up the specified shared memory segment
                        for i, shm in enumerate(active_shm):
                            if shm.name == shm_name:
                                try:
                                    # print(f"Render3D worker: Unlinking shared memory {shm_name}") # Commented out
                                    shm.close()
                                    shm.unlink()
                                    active_shm.pop(i)
                                    break
                                except Exception as e:
                                    print(f"Error unlinking shared memory {shm_name} (PID {os.getpid()}): {e}")
                    else:
                        # If no specific name provided, clean up all segments
                        print(f"Render3D worker (PID {os.getpid()}): Cleaning up ALL active shared memory segments.")
                        for shm_item in list(active_shm): # Iterate over a copy
                            try:
                                print(f"Render3D worker (PID {os.getpid()}): Unlinking {shm_item.name} from active_shm list.")
                                shm_item.close()
                                shm_item.unlink()
                                if shm_item in active_shm: # Check if still present before removing
                                     active_shm.remove(shm_item)
                            except Exception as e:
                                print(f"Error cleaning up {shm_item.name} (PID {os.getpid()}): {e}")
                        # active_shm = [] # Already cleared by removing items

                        # This part seems to be for a failed render, not general cleanup.
                        # The original code had this under an 'else' for 'if img is not None:'
                        # which doesn't make sense for a 'cleanup_shared_memory' command.
                        # Assuming this was a copy-paste error and removing the result_queue.put for failure here.
                        # If cleanup is successful, no message is typically sent unless requested.
                        # If it's part of a failed render, that's handled elsewhere.
                        # For now, let's assume cleanup_shared_memory doesn't send a success message.
                        # result_queue.put({
                        #    'success': False, # This seems wrong for a cleanup command
                        #    'error': "Rendering failed" # This is also misleading
                        # })
            
            except SystemExit as se: # Catch SystemExit from signal handler
                print(f"Render3D worker (PID {os.getpid()}): SystemExit caught ({se}), preparing to shut down.")
                break # Exit the loop to trigger finally block
            except Exception as e:
                print(f"Worker process error (PID {os.getpid()}): {e}")
                traceback.print_exc()
                # Always send a response to avoid hanging the parent process
                try:
                    result_queue.put({
                        'success': False,
                        'error': str(e),
                        'pid': os.getpid()
                    })
                except:
                    pass
    
    except Exception as e: # Outer try-except for fatal errors during setup
        print(f"Fatal worker error (PID {os.getpid()})") # Corrected this line
        traceback.print_exc()
        # No result_queue here as it might not be initialized or worker might be None
        
    finally:
        print(f"Render3D worker (PID {os.getpid()}): Shutting down, cleaning up...")
        # Cleanup code here if needed
        # Attempt to close the worker gracefully
        if worker:
            try:
                worker.close()
            except Exception as e:
                print(f"Render3D worker (PID {os.getpid()}): Error during worker close in finally: {e}")
                traceback.print_exc()
        
        # Final shared memory cleanup
        print(f"Render3D worker (PID {os.getpid()}): Cleaning up shared memory segments...")
        for shm in active_shm:
            try:
                shm.close()
                shm.unlink()
                print(f"Render3D worker (PID {os.getpid()}): Unlinked shared memory {shm.name}")
            except Exception as e:
                print(f"Render3D worker (PID {os.getpid()}): Error unlinking shared memory {shm.name} in finally: {e}")
                traceback.print_exc()
        
        active_shm = [] # Clear the list
        print(f"Render3D worker (PID {os.getpid()}): Cleanup complete, exiting.")
