import sys
import numpy as np
import os
import json

if __name__ == "__main__":
    # Arguments: <input_json> <output_npy>
    if len(sys.argv) != 3:
        print("Usage: python render3d_worker_entry.py <input_json> <output_npy>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_npy = sys.argv[2]
    
    # Default image size in case we can't read it from input
    default_img_size = 512
    
    try:
        # Load JSON data first to get img_size for potential fallback
        with open(input_json, 'r') as f:
            data = json.load(f)
        img_size = int(data.get('img_size', default_img_size))
        
        # Try relative import first (when run from the same directory)
        try:
            from render3d import Render3D
            from geom import Geometry3D, Sphere, Cube
        except ImportError:
            # If that fails, try to append parent directory to path for relative import
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from geometry.render3d import Render3D
                from geometry.geom import Geometry3D, Sphere, Cube
            except ImportError:
                # Last resort, try absolute import
                from src.geometry.render3d import Render3D
                from src.geometry.geom import Geometry3D, Sphere, Cube
        
        # Parse parameters
        background = data['background']
        show_edges = bool(data['show_edges'])
        camera_position = data['camera_position']
        use_cuda = bool(data['use_cuda'])
        
        # Create renderer
        renderer = Render3D(img_size=img_size, background=background, safe_mode=not use_cuda)
        
        # Recreate geometry objects from serialized data
        for geom_data in data['geom_args']:
            # Extract geometry data and rendering parameters
            geom_type = geom_data[0]['type']
            geom_params = geom_data[0]
            color = geom_data[1]
            edge_color = geom_data[2]
            opacity = float(geom_data[3])
            
            # Recreate geometry object based on type
            if geom_type == 'sphere':
                geom = Sphere(
                    center=tuple(geom_params['center']),
                    radius=float(geom_params['radius']),
                    quality=geom_params['quality'],
                    rotation=tuple(geom_params['rotation'])
                )
            elif geom_type == 'cube':
                geom = Cube(
                    center=tuple(geom_params['center']),
                    width=float(geom_params['width']),
                    quality=geom_params['quality'],
                    rotation=tuple(geom_params['rotation'])
                )
            else:
                continue
                
            # Add to renderer
            renderer.add_geometry(geom, color=color, edge_color=edge_color, opacity=opacity)
        
        # Render
        img = renderer.render(output=None, camera_position=camera_position, show_edges=show_edges)
        
        # Save result
        if img is not None:
            np.save(output_npy, img)
        else:
            np.save(output_npy, np.zeros((img_size, img_size, 3), dtype=np.uint8))
            
    except Exception as e:
        import traceback
        print(f"Error in worker: {e}")
        traceback.print_exc()
        
        # Get img_size from command line or use default if we couldn't load it
        if 'img_size' not in locals():
            img_size = default_img_size
            
        # Ensure we create an output file even on error
        np.save(output_npy, np.zeros((img_size, img_size, 3), dtype=np.uint8))
        sys.exit(1)
