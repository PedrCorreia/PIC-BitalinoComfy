import multiprocessing as mp
# Do NOT set start method at module level! Only set in main entry point if needed.
# mp.set_start_method('spawn', force=True)
import numpy as np
import os
import tempfile
import json
import subprocess
import sys

def geometry_to_dict(geom):
    """Convert geometry object to serializable dict"""
    try:
        if hasattr(geom, 'params') and 'radius' in geom.params:
            return {
                'type': 'sphere',
                'center': geom.center.tolist() if hasattr(geom.center, 'tolist') else list(geom.center),
                'radius': float(geom.params['radius']),
                'quality': geom.quality,
                'rotation': geom.rotation.tolist() if hasattr(geom.rotation, 'tolist') else list(geom.rotation)
            }
        elif hasattr(geom, 'params') and 'width' in geom.params:
            return {
                'type': 'cube',
                'center': geom.center.tolist() if hasattr(geom.center, 'tolist') else list(geom.center),
                'width': float(geom.params['width']),
                'quality': geom.quality,
                'rotation': geom.rotation.tolist() if hasattr(geom.rotation, 'tolist') else list(geom.rotation)
            }
        else:
            return {'type': 'unknown'}
    except Exception as e:
        print(f"Warning: Failed to serialize geometry object: {e}")
        return {'type': 'unknown'}

def render3d_color_subprocess(geom_args, img_size=512, background='white', show_edges=True, camera_position=None, use_cuda=True):
    """
    Render color image in a subprocess using Render3D, fully isolating PyVista/VTK from main process CUDA state.

    Args:
        geom_args: list of (geom, color, edge_color, opacity) tuples
        img_size: int, image size (default 512)
        background: str, background color (default 'white')
        show_edges: bool, whether to show edges (default True)
        camera_position: camera position for rendering (default None)
        use_cuda: if True, use GPU in subprocess if available (default True)

    Returns:
        np.ndarray: color image
    """
    # Convert geometry objects to serializable dicts
    serializable_geom_args = []
    for geom, color, edge_color, opacity in geom_args:
        geom_dict = geometry_to_dict(geom)
        serializable_geom_args.append([geom_dict, color, edge_color, opacity])
    
    # Prepare JSON data - handle None values for JSON serialization
    json_data = {
        'geom_args': serializable_geom_args,
        'img_size': img_size,
        'background': background if background is not None else 'white',
        'show_edges': show_edges if show_edges is not None else True,
        'camera_position': camera_position if camera_position is not None else None,
        'use_cuda': use_cuda if use_cuda is not None else True
    }
    
    try:
        # Write args to temp JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf_args:
            args_path = tf_args.name
            json.dump(json_data, tf_args)
            
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tf_out:
            out_path = tf_out.name
            
        # Find worker script path
        worker_path = os.path.join(os.path.dirname(__file__), 'render3d_worker_entry.py')
        
        # Set PYTHONPATH to help with module imports
        env = os.environ.copy()
        # Add multiple potential paths to PYTHONPATH to maximize chances of success
        module_paths = [
            # Direct parent directory (for relative imports)
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            # Project root
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')),
            # Custom nodes directory
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        ]
        
        # Convert paths to string with proper separator
        pythonpath = os.pathsep.join(module_paths + 
                                     [env['PYTHONPATH']] if 'PYTHONPATH' in env else [])
        env['PYTHONPATH'] = pythonpath
        
        # Call subprocess
        result = subprocess.run([sys.executable, worker_path, args_path, out_path], 
                               capture_output=True, env=env)
        
        if result.returncode != 0:
            print(f"Subprocess rendering failed: {result.stderr.decode()}")
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            try:
                img = np.load(out_path)
            except Exception as e:
                print(f"Failed to load output image: {e}")
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Clean up temp files
        if os.path.exists(args_path):
            os.remove(args_path)
        if os.path.exists(out_path):
            os.remove(out_path)
            
    except Exception as e:
        print(f"Error in subprocess rendering: {e}")
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
    return img