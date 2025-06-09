"""
Bridge module for ComfyUI integration with the 3D rendering system.
Provides isolated 3D rendering functions using the Render3D class with proper process isolation.
"""
import os
import sys
import numpy as np
import torch

# Import the Render3D class for process-isolated rendering
from .render3d import Render3D
from .geom import Geometry3D, Sphere, Cube

# Global renderer instance with process isolation
_global_renderer = None

def get_renderer(img_size=512, background='white', safe_mode=False, recreate=False):
    """
    Get or create a global renderer instance with process isolation
    
    Args:
        img_size: Size of the rendered image
        background: Background color for rendering
        safe_mode: If True, use CPU rendering instead of GPU
        recreate: If True, recreate the renderer even if it exists
        
    Returns:
        Render3D: A renderer instance with process isolation
    """
    global _global_renderer
    
    # Recreate renderer if requested or if it doesn't exist
    if _global_renderer is None or recreate:
        # Clean up any existing renderer first
        if _global_renderer is not None:
            _global_renderer.cleanup()
            _global_renderer = None
            
        # Create new renderer
        _global_renderer = Render3D(
            img_size=img_size,
            background=background,
            safe_mode=safe_mode,
            process_isolation=True # Reverted to True
        )
    
    return _global_renderer

def cleanup_renderer():
    """Clean up the global renderer instance"""
    global _global_renderer
    if _global_renderer is not None:
        _global_renderer.cleanup()
        _global_renderer = None

def render_geometry_for_comfy(geom_args, img_size=512, background='white', show_edges=True, camera_position=None, retry_on_failure=True):
    """
    Render geometry using the process-isolated Render3D for ComfyUI.
    
    Args:
        geom_args: List of tuples (geometry_object, color, edge_color, opacity)
        img_size: Size of the rendered image (int or tuple)
        background: Background color
        show_edges: Whether to show edges of the geometry
        camera_position: Camera position for rendering
        retry_on_failure: If True, retry rendering once with a fresh renderer on failure
        
    Returns:
        numpy.ndarray: The rendered image
    """
    try:
        # Get or create a renderer with process isolation - we don't recreate by default
        renderer = get_renderer(img_size=img_size, background=background, recreate=False)
        
        # Clear any previous geometries using a proper method
        if hasattr(renderer, 'clear_geometries'):
            renderer.clear_geometries()
        else:
            # Fallback: Reset the geometries list directly
            renderer.geometries = []
        
        # Add all geometries
        for geom, color, edge_color, opacity in geom_args:
            renderer.add_geometry(geom, color=color, edge_color=edge_color, opacity=opacity)
        
        # Render the scene
        image = renderer.render(
            camera_position=camera_position, 
            show_edges=show_edges,
            img_size=img_size
        )
        
        # If rendering succeeded, return the image
        if image is not None:
            return image
        
        # If we get here, rendering failed but we can try again
        if retry_on_failure:
            print("Render failed - recreating renderer and trying again...")
            # Force recreation of the renderer
            renderer = get_renderer(img_size=img_size, background=background, recreate=True)
            
            # Clear and add geometries
            renderer.clear_geometries()
            for geom, color, edge_color, opacity in geom_args:
                renderer.add_geometry(geom, color=color, edge_color=edge_color, opacity=opacity)
            
            # Try rendering again (without retry this time to avoid infinite recursion)
            return render_geometry_for_comfy(
                geom_args, img_size, background, show_edges, camera_position, 
                retry_on_failure=False
            )
        
        # If we still failed, return None
        return None
        
    except Exception as e:
        import traceback
        print(f"Error in render_geometry_for_comfy: {e}")
        traceback.print_exc()
        
        # Try once more with a fresh renderer if this is our first attempt
        if retry_on_failure:
            print("Exception during render - recreating renderer and trying once more...")
            try:
                # Force recreation of the renderer
                cleanup_renderer()  # Ensure a clean state
                
                # Try rendering again (without retry this time to avoid infinite recursion)
                return render_geometry_for_comfy(
                    geom_args, img_size, background, show_edges, camera_position, 
                    retry_on_failure=False
                )
            except Exception as e2:
                print(f"Second render attempt also failed: {e2}")
                traceback.print_exc()
                
        return None
