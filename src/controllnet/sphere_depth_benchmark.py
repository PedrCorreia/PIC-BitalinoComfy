"""
sphere_depth_benchmark.py: Benchmark and visualize depth maps from fast MiDaS and SphereControl.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from fast_mida_depth import fast_mida_depth_map
from sphere_control import SphereControl

def add_far_axis(ax, z=10, length=1.5, color='red'):
    # Draw a 3D axis far away for perspective
    ax.plot([0, length], [0, 0], [z, z], color=color, linewidth=2)
    ax.plot([0, 0], [0, length], [z, z], color=color, linewidth=2)
    ax.plot([0, 0], [0, 0], [z, z+length], color=color, linewidth=2)

if __name__ == "__main__":
    shape = (256, 256)
    # Fast MiDaS depth
    t0 = time.time()
    mida_depth = fast_mida_depth_map(shape)
    t1 = time.time()
    mida_time = t1 - t0

    # SphereControl depth
    sc = SphereControl(radius=0.35, center=(0.5, 0.5, 0.5), arousal=0.5)
    t0 = time.time()
    sphere_depth = sc.get_depth_map(image_shape=shape)
    t1 = time.time()
    sphere_time = t1 - t0

    # Plot both depth maps
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(mida_depth, cmap='gray')
    axs[0].set_title(f'Fast MiDaS Depth\n{mida_time*1e3:.2f} ms')
    axs[0].axis('off')
    axs[1].imshow(sphere_depth, cmap='gray')
    axs[1].set_title(f'SphereControl Depth\n{sphere_time*1e3:.2f} ms')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    # 3D plot with far axis for perspective
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    ax.plot_surface(X, Y, sphere_depth, cmap='viridis', edgecolor='none', alpha=0.8)
    add_far_axis(ax, z=2.5, length=0.5)
    ax.set_title('SphereControl Depth Map with Far Axis')
    plt.show()
