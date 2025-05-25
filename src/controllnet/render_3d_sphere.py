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

class SphereRenderer:
    def __init__(self, 
                 center=(0.5, 0.5, 0.5), 
                 radius=0.5, 
                 img_size=(512, 512), 
                 color='red',
                 elevation=20, 
                 azimuth=30,
                 ambient_light=0.3,
                 specular_light=0.5):
        """
        Initialize a 3D sphere renderer with custom parameters
        
        Args:
            center: (x, y, z) coordinates of sphere center, range 0-1
            radius: sphere radius, range 0-1
            img_size: output image dimensions in pixels (width, height)
            color: sphere color (matplotlib color name or RGB tuple)
            elevation: camera elevation angle in degrees
            azimuth: camera azimuth angle in degrees
            ambient_light: intensity of ambient light (0-1)
            specular_light: intensity of specular highlights (0-1)
        """
        self.center = center
        self.radius = radius
        self.img_size = img_size
        self.color = color
        self.elevation = elevation
        self.azimuth = azimuth
        self.ambient_light = ambient_light
        self.specular_light = specular_light
        
    def render(self, benchmark=False):
        """Render the 3D sphere and return as numpy RGB array"""
        start_time = time.time()
        
        # Create figure with 3D axes
        fig = plt.figure(figsize=(self.img_size[0]/100, self.img_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color to black
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Define the view angle for camera
        ax.view_init(elev=self.elevation, azim=self.azimuth)
        
        # Hide axes and set limits
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Draw the sphere
        cx, cy, cz = self.center
        r = self.radius
        
        # Create sphere mesh with higher resolution for smoother sphere
        u = np.linspace(0, 2 * np.pi, 100)  # Increased from 50 to 100
        v = np.linspace(0, np.pi, 100)      # Increased from 50 to 100
        x = cx + r * np.outer(np.cos(u), np.sin(v))
        y = cy + r * np.outer(np.sin(u), np.sin(v))
        z = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        mesh_creation_time = time.time() - start_time
        plot_start_time = time.time()
        
        # Plot the sphere with lighting
        ax.plot_surface(
            x, y, z, 
            color=self.color, 
            shade=True,
            alpha=1.0,
            rstride=1,  # Reduced stride for higher detail
            cstride=1,  # Reduced stride for higher detail
            linewidth=0
        )
        
        plot_time = time.time() - plot_start_time
        render_start_time = time.time()
        
        # Render to numpy array
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        
        # Get the image data
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Convert to numpy array and ensure RGB (no alpha channel)
        img = np.array(Image.open(buf).convert('RGB'))
        
        # Close figure to prevent memory leak
        plt.close(fig)
        
        total_time = time.time() - start_time
        
        if benchmark:
            print(f"Benchmark results:")
            print(f"  Mesh creation: {mesh_creation_time:.3f}s")
            print(f"  Surface plotting: {plot_time:.3f}s")
            print(f"  Image rendering: {time.time() - render_start_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Resolution: {len(u)}x{len(v)} points, Image: {img.shape[1]}x{img.shape[0]} pixels")
        
        return img
    
    def save_image(self, output_path="sphere_render.png", benchmark=False):
        """Render and save the sphere image to a file"""
        img = self.render(benchmark=benchmark)
        Image.fromarray(img).save(output_path)
        print(f"Rendered image saved to {output_path}")
        return output_path

def main():
    """Command-line interface for rendering spheres"""
    parser = argparse.ArgumentParser(description="Render a 3D sphere with proper shading")
    parser.add_argument("--cx", type=float, default=0.5, help="X-coordinate of sphere center (0-1)")
    parser.add_argument("--cy", type=float, default=0.5, help="Y-coordinate of sphere center (0-1)")
    parser.add_argument("--cz", type=float, default=0.5, help="Z-coordinate of sphere center (0-1)")
    parser.add_argument("--radius", type=float, default=0.2, help="Sphere radius (0-1)")
    parser.add_argument("--color", type=str, default="red", help="Sphere color (matplotlib color name)")
    parser.add_argument("--elevation", type=float, default=20, help="Camera elevation angle (degrees)")
    parser.add_argument("--azimuth", type=float, default=30, help="Camera azimuth angle (degrees)")
    parser.add_argument("--output", type=str, default="sphere_render.png", help="Output image file path")
    parser.add_argument("--width", type=int, default=512, help="Output image width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Output image height in pixels")
    parser.add_argument("--benchmark", action="store_true", help="Run and display benchmarking information")
    
    args = parser.parse_args()
    
    # Create renderer with command line parameters
    renderer = SphereRenderer(
        center=(args.cx, args.cy, args.cz),
        radius=args.radius,
        img_size=(args.width, args.height),
        color=args.color,
        elevation=args.elevation,
        azimuth=args.azimuth
    )
    
    # Render and save the image
    renderer.save_image(args.output, benchmark=args.benchmark)

if __name__ == "__main__":
    main()
