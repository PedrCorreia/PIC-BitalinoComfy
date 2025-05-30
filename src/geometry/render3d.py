import pyvista as pv
import numpy as np
from .geom import Geometry3D, Cube

def euler_matrix(rx, ry, rz):
    # Angles in degrees
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]] )
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]] )
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

class Render3D:
    def __init__(self, img_size=512, background='white'):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.background = background
        self.geometries = []

    def add_geometry(self, geom: Geometry3D, color='white', edge_color='black', opacity=1.0):
        self.geometries.append((geom, color, edge_color, opacity))

    def render(self, output="render3d.png", camera_position=None, show_edges=True, perspective_blur_z=None, perspective_blur_params=None, **kwargs):
        import cv2
        # Ensure img_size is a tuple for Plotter
        plotter_img_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        plotter = pv.Plotter(window_size=plotter_img_size, off_screen=True)
        for geom, color, edge_color, opacity in self.geometries:
            if isinstance(geom, Geometry3D):
                # Example: handle Sphere and Cube
                if 'radius' in geom.params:
                    mesh = pv.Sphere(center=geom.center, radius=geom.params['radius'],
                                     theta_resolution=self._res(geom), phi_resolution=self._res(geom))
                elif 'width' in geom.params and isinstance(geom, Cube):
                    # Create cube vertices, apply rotation, then make mesh
                    w = geom.params['width'] / 2.0
                    # Cube vertices centered at origin
                    verts = np.array([
                        [-w, -w, -w], [w, -w, -w], [w, w, -w], [-w, w, -w],
                        [-w, -w, w], [w, -w, w], [w, w, w], [-w, w, w]
                    ])
                    # Apply rotation
                    R = euler_matrix(*geom.rotation)
                    verts = verts @ R.T
                    # Move to center
                    verts = verts + geom.center
                    # Cube faces (PyVista expects 0-based indices)
                    faces = [
                        [4, 0, 1, 2, 3],  # bottom
                        [4, 4, 5, 6, 7],  # top
                        [4, 0, 1, 5, 4],  # side
                        [4, 1, 2, 6, 5],  # side
                        [4, 2, 3, 7, 6],  # side
                        [4, 3, 0, 4, 7],  # side
                    ]
                    faces_flat = np.hstack(faces)
                    mesh = pv.PolyData(verts, faces_flat)
                elif 'width' in geom.params:
                    # fallback: unrotated cube
                    mesh = pv.Cube(center=geom.center, x_length=geom.params['width'],
                                   y_length=geom.params['width'], z_length=geom.params['width'])
                else:
                    continue  # Extend for other shapes
                plotter.add_mesh(
                    mesh,
                    color=color,
                    show_edges=show_edges,
                    edge_color=edge_color,
                    opacity=opacity,
                    specular=1.0,         # maximum specular reflection for shininess
                    specular_power=100.0, # high value for sharp highlights
                    smooth_shading=True   # enable smooth shading for a shiny look
                )
        plotter.set_background(self.background)
        if camera_position:
            plotter.camera_position = camera_position
        img = plotter.screenshot(None, transparent_background=True)
        plotter.close()
        # Convert img to uint8 if needed
        if img is not None and img.dtype != np.uint8:
            img = (img * 255).clip(0,255).astype(np.uint8)
        # Remove alpha channel if present
        if img is not None and img.shape[-1] == 4:
            img = img[..., :3]
        # --- Perspective blur if requested ---
        if perspective_blur_z is not None and img is not None:
            params = perspective_blur_params or {}
            min_z = params.get('min_z', -15)
            max_blur = params.get('max_blur', 8)
            z = perspective_blur_z
            if z < 0:
                if z <= min_z:
                    blur_sigma = max_blur
                else:
                    blur_sigma = max_blur * (np.log1p(abs(z)) / np.log1p(abs(min_z)))
            else:
                blur_sigma = 0
            if blur_sigma > 0.2:
                ksize = int(2 * round(blur_sigma) + 1)
                img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
        if output and img is not None: # Check if img is not None before saving
            cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"3D Render saved to {output if output else 'memory'}") # Clarify if not saved to file
        return img

    def _res(self, geom):
        # Map quality to resolution
        quality_map = {'low': 8, 'medium': 16, 'high': 32}
        return quality_map.get(geom.quality, 16)

    def render_depth(self, camera_position=None, **kwargs):
        """
        Renders the depth map of the scene.
        """
        plotter_img_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        plotter = pv.Plotter(window_size=plotter_img_size, off_screen=True)
        
        for geom, color, edge_color, opacity in self.geometries:
            if isinstance(geom, Geometry3D):
                mesh = None
                if 'radius' in geom.params: # Sphere
                    mesh = pv.Sphere(center=geom.center, radius=geom.params['radius'],
                                     theta_resolution=self._res(geom), phi_resolution=self._res(geom))
                elif 'width' in geom.params and isinstance(geom, Cube): # Cube
                    w = geom.params['width'] / 2.0
                    verts = np.array([
                        [-w, -w, -w], [w, -w, -w], [w, w, -w], [-w, w, -w],
                        [-w, -w, w], [w, -w, w], [w, w, w], [-w, w, w]
                    ])
                    R = euler_matrix(*geom.rotation)
                    verts = verts @ R.T
                    verts = verts + geom.center
                    faces = [
                        [4, 0, 1, 2, 3], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4],
                        [4, 1, 2, 6, 5], [4, 2, 3, 7, 6], [4, 3, 0, 4, 7]
                    ]
                    faces_flat = np.hstack(faces)
                    mesh = pv.PolyData(verts, faces_flat)
                elif 'width' in geom.params: # Fallback unrotated cube
                     mesh = pv.Cube(center=geom.center, x_length=geom.params['width'],
                                   y_length=geom.params['width'], z_length=geom.params['width'])
                
                if mesh:
                    plotter.add_mesh(mesh, color=color, opacity=opacity) # Color/opacity don't affect depth but good to keep consistent

        plotter.set_background(self.background) 
        if camera_position:
            plotter.camera_position = camera_position
        
        # Force the rendering pass by taking a "screenshot" to memory
        plotter.screenshot(filename=None) # This ensures the scene is rendered

        # Get the depth image
        depth_buffer = plotter.get_image_depth()
        plotter.close()

        if depth_buffer is not None:
            min_depth = np.min(depth_buffer)
            max_depth = np.max(depth_buffer)
            if max_depth > min_depth: # Avoid division by zero
                depth_normalized = (depth_buffer - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = np.zeros_like(depth_buffer) # Or np.ones_like if far plane is 1 and all is far

            # PyVista's depth is typically 0 (near) to 1 (far).
            # If your node expects inverted depth (white=near), you might do:
            # depth_normalized = 1.0 - depth_normalized
            return depth_normalized 
        return None

class Render2D: #not working yet
    def __init__(self, img_size=512, background='white'):
        self.img_size = img_size
        self.background = background
        self.geometries = []

    def add_geometry(self, geom: Geometry3D, color='black', linewidth=2):
        self.geometries.append((geom, color, linewidth))

    def render(self, output="render2d.png", projection_axis='z', **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(self.img_size/100, self.img_size/100), dpi=100)
        ax.set_facecolor(self.background)
        for geom, color, linewidth in self.geometries:
            # Example: project vertices to 2D
            verts = np.array([v for v in getattr(geom, 'vertices', [])])
            if verts.size > 0:
                if projection_axis == 'z':
                    ax.plot(verts[:,0], verts[:,1], 'o-', color=color, linewidth=linewidth)
                elif projection_axis == 'y':
                    ax.plot(verts[:,0], verts[:,2], 'o-', color=color, linewidth=linewidth)
                elif projection_axis == 'x':
                    ax.plot(verts[:,1], verts[:,2], 'o-', color=color, linewidth=linewidth)
        ax.axis('equal')
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"2D Render saved to {output}")
