import numpy as np
import matplotlib.pyplot as plt
import cProfile # Added for profiling
import pstats # Added for reading profile stats
import io # Added for capturing pstats output
import noise # For Perlin noise
from opensimplex import OpenSimplex # For Simplex noise

class Surface_Noise:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 100000) # Default to a random seed
        self.seed = seed # Store the seed directly
        self.simplex_generator = OpenSimplex(seed=self.seed)
        # print(f"[Surface_Noise] Initialized with Simplex seed: {self.seed}")


    def perlin_noise(self, x, y, z, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        """Generate Perlin noise for a 3D surface using the 'noise' library."""
        octaves = max(1, int(octaves)) # Ensure octaves is an integer >= 1
        
        # noise.pnoise3 parameters:
        # x, y, z: coordinates
        # octaves: number of layers of noise
        # persistence: how much each octave contributes
        # lacunarity: how much frequency increases for each octave
        # repeatx, repeaty, repeatz: period for repeating noise (optional, makes it tileable)
        # base: seed for the noise
        return noise.pnoise3(x * scale, 
                             y * scale, 
                             z * scale, 
                             octaves=octaves, 
                             persistence=persistence, 
                             lacunarity=lacunarity,
                             repeatx=1024, # A large repeat period
                             repeaty=1024,
                             repeatz=1024,
                             base=self.seed) # Base is an integer seed for Perlin

    def simplex_noise(self, x, y, z, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        """Generate Simplex noise (fBm) for a 3D surface using 'opensimplex'."""
        octaves = max(1, int(octaves)) # Ensure octaves is an integer >= 1

        total = 0.0
        current_frequency = 1.0
        current_amplitude = 1.0
        max_amplitude = 0.0 # To normalize the result to approx [-1, 1]

        for _ in range(octaves):
            total += self.simplex_generator.noise3(x * scale * current_frequency, 
                                                    y * scale * current_frequency, 
                                                    z * scale * current_frequency) * current_amplitude
            max_amplitude += current_amplitude
            current_amplitude *= persistence
            current_frequency *= lacunarity
        
        if max_amplitude == 0: # Avoid division by zero
            return 0.0
        
        return total / max_amplitude


    def apply_deformation(self, geometry_object, deformation_type="none", strength=0.0, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        """
        Applies deformation to the vertices of a geometry object.

        Args:
            geometry_object: The geometry object (e.g., Sphere, Cube) with a 'vertices' attribute.
            deformation_type (str): Type of deformation ("none", "perlin", "simplex").
            strength (float): Magnitude of deformation (0.0 to 1.0).
            scale (float): Scale of the noise pattern.
            octaves (int): Number of octaves for noise.
            persistence (float): Persistence for noise.
            lacunarity (float): Lacunarity for noise.
        """
        #print(f"[apply_deformation] Called with: type={deformation_type}, strength={strength:.4f}, scale={scale:.4f}, octaves={octaves}, seed={self.seed}")

        if deformation_type == "none" or strength == 0.0:

            return geometry_object



        original_vertices = geometry_object.vertices.copy() # Work on a copy
        deformed_vertices = original_vertices.copy()
        
        if original_vertices.shape[0] < 0:
            print("[apply_deformation] No vertices to process.")
            return geometry_object

        centroid = np.mean(original_vertices, axis=0)
        #print(f"[apply_deformation] Calculated centroid: {centroid}")
        
        normals = original_vertices - centroid
        norm_magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Handle potential division by zero if a vertex is at the centroid
        safe_normals = np.zeros_like(normals)
        np.divide(normals, norm_magnitudes, out=safe_normals, where=norm_magnitudes!=0)
        normals = safe_normals
        



        max_displacement_val = -float('inf')
        min_displacement_val = float('inf')
        num_vertices_processed = 0

        for i, v in enumerate(original_vertices):
            x, y, z = v
            displacement_val = 0.0

            if deformation_type == "perlin":
                displacement_val = self.perlin_noise(x, y, z, scale, octaves, persistence, lacunarity)
            elif deformation_type == "simplex":
                displacement_val = self.simplex_noise(x, y, z, scale, octaves, persistence, lacunarity)
            
            max_displacement_val = max(max_displacement_val, displacement_val)
            min_displacement_val = min(min_displacement_val, displacement_val)
            num_vertices_processed +=1
            # Apply displacement along the normal
            displacement_vector = normals[i] * displacement_val * strength
            deformed_vertices[i] += displacement_vector
            

        

        geometry_object.vertices = deformed_vertices
        #print(f"[apply_deformation] Updated geometry_object.vertices. Sample (first 3):\\n{geometry_object.vertices[:3]}")
        
        if hasattr(geometry_object, 'mesh') and hasattr(geometry_object.mesh, 'points'):
            #print(f"[apply_deformation] Updating mesh.points. Current mesh points sample (first 3):\\n{geometry_object.mesh.points[:3]}")
            geometry_object.mesh.points = deformed_vertices
            #print(f"[apply_deformation] New mesh.points sample (first 3):\\n{geometry_object.mesh.points[:3]}")
            if hasattr(geometry_object.mesh, '_updated_points'): # For some PyVista versions/custom classes
                 geometry_object.mesh._updated_points()
                # print("[apply_deformation] Called mesh._updated_points()")
            elif hasattr(geometry_object.mesh, 'compute_normals'): # Recompute normals if they are used for shading
                 geometry_object.mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
     #            print("[apply_deformation] Called mesh.compute_normals()")
            
            # Explicitly mark the mesh as modified for the VTK pipeline
            if hasattr(geometry_object.mesh, 'Modified'):
                geometry_object.mesh.Modified()
      #          print("[apply_deformation] Called mesh.Modified()")
        else:
            print("[apply_deformation] Mesh or mesh.points attribute not found, skipping mesh update.")

        return geometry_object

if __name__ == "__main__":
    import sys
    import os
    import time
    import math
    import cv2 # For displaying the rendered image
    import cProfile
    import pstats
    import io

    # Ensure the project root (PIC-BitalinoComfy) is in sys.path for direct execution
    # to allow 'from src.geometry...' imports to work.
    current_file_path_eff = os.path.abspath(__file__)
    # current_file_path_eff is .../PIC-BitalinoComfy/src/geometry/surface_eff.py
    geometry_dir_eff = os.path.dirname(current_file_path_eff)
    # geometry_dir_eff is .../PIC-BitalinoComfy/src/geometry
    src_dir_eff = os.path.dirname(geometry_dir_eff)
    # src_dir_eff is .../PIC-BitalinoComfy/src
    pic_root_eff = os.path.dirname(src_dir_eff)
    # pic_root_eff is .../PIC-BitalinoComfy
    if pic_root_eff not in sys.path:
        sys.path.insert(0, pic_root_eff)

    # Moved imports here, after sys.path modification
    from src.geometry.render3d import Render3D
    from src.geometry.geom import Sphere, Cube

    # Profiling setup
    profiler = cProfile.Profile()

    # Initial setup
    surface_noise_gen = Surface_Noise()
    
    # Attempt to set up renderer, handle potential errors (e.g. no display)
    renderer = None
    try:
        renderer = Render3D(img_size=512, background='white', safe_mode=True, process_isolation=False) # Corrected instantiation
    except Exception as e:
        print(f"Failed to initialize Render3D, likely due to display issues: {e}")
        print("Will proceed without visual rendering if in a headless environment.")

    base_sphere = Sphere(center=(0, 0, 0), radius=1.5, quality='medium')
    original_vertices = base_sphere.vertices.copy() # Store original vertices
    # Ensure the mesh also has these original points if it's going to be reset
    if hasattr(base_sphere, 'mesh') and base_sphere.mesh is not None:
        original_mesh_points = base_sphere.mesh.points.copy()
    else:
        original_mesh_points = None # Should not happen if Sphere init is correct
        print("WARNING: base_sphere does not have a mesh after initialization for storing original_mesh_points.")

    # Animation parameters
    t_anim = 0.0
    # FOCUS ON PERLIN NOISE FOR PROFILING
    deformation_types = ["perlin"] 
    current_deformation_type_idx = 0 # Will always be Perlin

    # PROFILING PARAMETERS
    PROFILE_FRAMES = 100 # Limit frames for profiling
    frames_processed = 0

    print(f"Starting live deformation loop for profiling ({PROFILE_FRAMES} frames, Perlin noise only).")
    print("If no window appears, rendering might have failed (e.g., headless environment).")
    print("Press 'q' in the window to quit early if visuals are enabled.")

    profiler.enable() # Start profiling

    try:
        # Loop for a fixed number of frames for profiling
        for frame_num in range(PROFILE_FRAMES):
            # Reset base_sphere to its original state before applying deformation
            base_sphere.vertices = original_vertices.copy()
            if hasattr(base_sphere, 'mesh') and base_sphere.mesh is not None and original_mesh_points is not None:
                base_sphere.mesh.points = original_mesh_points.copy()

            # Animate deformation parameters
            deformation_strength = (math.sin(t_anim * 0.5) + 1) / 2  # Cycles 0.0 to 1.0
            
            # Cycle deformation type every few seconds - Not needed for Perlin-only profiling
            # if int(t_anim) % 20 == 0 and int(t_anim) != int(t_anim - 0.1): 
            #     current_deformation_type_idx = (current_deformation_type_idx + 1) % len(deformation_types)
            
            current_deformation_type = deformation_types[current_deformation_type_idx] # Should always be 'perlin'
            
            noise_scale = 0.8 + (math.cos(t_anim * 0.3) * 0.7) # Animate scale between 0.1 and 1.5
            noise_octaves = int(3 + (math.sin(t_anim * 0.2) * 2))

            # Apply deformation directly to base_sphere
            deformed_geom = surface_noise_gen.apply_deformation(
                geometry_object=base_sphere, 
                deformation_type=current_deformation_type,
                strength=deformation_strength,
                scale=noise_scale,
                octaves=noise_octaves,
                persistence=0.5,
                lacunarity=2.0
            )

            if renderer:
                renderer.clear_geometries()
                renderer.add_geometry(deformed_geom, color='cornflowerblue', edge_color='black', opacity=1.0)
                img = renderer.render( 
                    camera_position=[(0, 0, 7), (0, 0, 0), (0, 1, 0)], 
                )
                # Display the image
                if img is not None:
                    cv2.imshow('Surface Deformation Test', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested quit during profiling.")
                        break 
                else:
                    print("Rendered image is None.")
            else:
                time.sleep(0.001)

            t_anim += 0.1
            frames_processed += 1
            # time.sleep(0.01) # Commented out for profiling to measure actual work without artificial delay

    except KeyboardInterrupt:
        print("Profiling interrupted by user.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        profiler.disable() # Stop profiling
        
        if renderer:
            renderer.cleanup() # Corrected method call
        cv2.destroyAllWindows()
        print("Cleanup complete.")

        print(f"\nProfiling results for {frames_processed} frames (Perlin noise):")
        s = io.StringIO()
        # Sort stats by cumulative time
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        # Save profile data to a file
        profile_file_name = "surface_eff_perlin.prof"
        # Ensure the path is correct, placing it in the PIC-BitalinoComfy directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/geometry
        src_dir = os.path.dirname(script_dir) # .../src
        project_root = os.path.dirname(src_dir) # .../PIC-BitalinoComfy
        full_profile_path = os.path.join(project_root, profile_file_name)
        
        try:
            profiler.dump_stats(full_profile_path)
            print(f"Profile data saved to {full_profile_path}")
        except Exception as dump_exc:
            print(f"Error saving profile data to {full_profile_path}: {dump_exc}")
            # Fallback to saving in current dir if project root path fails for some reason
            try:
                profiler.dump_stats(profile_file_name)
                print(f"Profile data saved to {os.path.join(os.getcwd(), profile_file_name)} as a fallback.")
            except Exception as fallback_dump_exc:
                print(f"Fallback save also failed: {fallback_dump_exc}")


        # Optional: Print top N functions by cumulative time
        # print("\\nTop 20 functions by cumulative time:")
        # ps_sorted_cumtime = pstats.Stats(profiler, stream=io.StringIO()).sort_stats('cumulative')
        # ps_sorted_cumtime.print_stats(20)
        # print(ps_sorted_cumtime.stream.getvalue())
