import pyvista as pv
import numpy as np
from geom import Geometry3D, Cube

def euler_matrix(rx, ry, rz):
    # Angles in degrees
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]] )
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]] )
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

class Render3D:
    def __init__(self, img_size=512, background='white'):
        self.img_size = img_size
        self.background = background
        self.geometries = []

    def add_geometry(self, geom: Geometry3D, color='white', edge_color='black', opacity=1.0):
        self.geometries.append((geom, color, edge_color, opacity))

    def render(self, output="render3d.png", camera_position=None, show_edges=True, perspective_blur_z=None, perspective_blur_params=None, **kwargs):
        import cv2
        plotter = pv.Plotter(window_size=(self.img_size, self.img_size), off_screen=True)
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
        if output and img is not None:
            cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"3D Render saved to {output}")
        return img

    def _res(self, geom):
        # Map quality to resolution
        return {'low': 8, 'medium': 32, 'high': 100}.get(geom.quality, 32)

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
