"""
Generate a 3D sphere with proper shading and lighting
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from PIL import Image
import io
import os
import argparse
import time
import pyvista as pv

class Sphere:
    def __init__(self, center=(0, 0, 0), radius=0.3, color='white', edge_color='black', theta_res=16, phi_res=16):
        self.center = center
        self.radius = radius
        self.color = color
        self.edge_color = edge_color
        self.theta_res = theta_res
        self.phi_res = phi_res

class SphereScene:
    def __init__(self, img_size=512, background='white'):
        self.spheres = []
        self.img_size = img_size
        self.background = background

    def add_sphere(self, sphere: Sphere):
        self.spheres.append(sphere)

    def render(self, output="multi_sphere_render.png", benchmark=True):
        start_time = time.time()
        plotter = pv.Plotter(window_size=(self.img_size, self.img_size), off_screen=True)
        mesh_times = []
        for s in self.spheres:
            mesh_start = time.time()
            sphere = pv.Sphere(
                center=s.center,
                radius=s.radius,
                theta_resolution=s.theta_res,
                phi_resolution=s.phi_res
            )
            plotter.add_mesh(
                sphere,
                color=s.color,
                smooth_shading=True,
                specular=0.7,
                specular_power=30,
                ambient=0.2,
                diffuse=0.8,
                show_edges=True,
                edge_color=s.edge_color,
                opacity=1.0
            )
            mesh_times.append(time.time() - mesh_start)
        plotter.set_background(self.background, top=self.background)
        plotter.show_axes()
        plot_start = time.time()
        # Dramatic perspective: camera far from the spheres, looking down the line
        plotter.camera_position = [(1, 1, 2), (0.2, 0.2, -1), (0, 0, 1)]
        img = plotter.screenshot(output, transparent_background=True)
        plotter.close()
        total_time = time.time() - start_time
        plot_time = time.time() - plot_start
        if benchmark:
            for i, t in enumerate(mesh_times):
                print(f"Mesh creation for sphere {i+1}: {t:.3f}s")
            print(f"Rendering time: {plot_time:.3f}s")
            print(f"Total time: {total_time:.3f}s")
            # Check if background is transparent
            from PIL import Image
            im = Image.open(output)
            if im.mode == 'RGBA':
                extrema = im.getchannel('A').getextrema()
                if isinstance(extrema, tuple) and len(extrema) == 2:
                    min_alpha, max_alpha = extrema
                    try:
                        if isinstance(max_alpha, (int, float)) and max_alpha < 255:
                            print("Background is transparent.")
                        else:
                            print("Background is not transparent (likely white).")
                    except Exception as e:
                        print(f"Could not determine transparency: {e}")
                        print("Background is not transparent (likely white).")
                else:
                    print("Background is not transparent (likely white).")
            else:
                print("Background is not transparent (likely white).")
        print(f"Rendered image saved to {output} (with transparent background if supported)")
        return img

# Example usage: create 3 spheres, add to scene, render
if __name__ == "__main__":
    scene = SphereScene(img_size=512, background='white')
    # Place spheres in a triangle, nearby but not in a line, with different depths
    scene.add_sphere(Sphere(center=(0, 0, 0), radius=0.3, color='white', edge_color='black', theta_res=16, phi_res=16))
    scene.add_sphere(Sphere(center=(0.5, 0.2, -0.5), radius=0.25, color='red', edge_color='black', theta_res=16, phi_res=16))
    scene.add_sphere(Sphere(center=(0.2, 0.6, -1), radius=0.2, color='blue', edge_color='black', theta_res=16, phi_res=16))
    # Camera set to show all spheres in a good perspective
    scene.render(output="multi_sphere_render.png", benchmark=True)

