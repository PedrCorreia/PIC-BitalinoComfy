"""
ControllNet Sphere Control Module

This module provides a minimal class for controlling a sphere (e.g., for diffusion guidance or visualization) in a 3D field.
You can set the sphere's center and radius, and generate a mask or field for use in downstream applications.


"""
import numpy as np

class SphereControl:
    def __init__(self, center=(0.5, 0.5, 0.5), radius=0.2, field_shape=(64, 64, 64)):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.field_shape = field_shape

    def set_radius(self, r):
        self.radius = float(r)

    def set_center(self, center):
        self.center = np.array(center, dtype=np.float32)

    def set_arousal(self, arousal):
        """
        Set arousal parameter (0 to 1). Higher arousal increases distortion.
        """
        self.arousal = float(np.clip(arousal, 0, 1))

    def set_git_branch(self, branch_name):
        """
        Set the git branch name for tracking or display purposes.
        """
        self.git_branch = branch_name

    def get_git_branch(self):
        """
        Get the current git branch name if set, else return None.
        """
        return getattr(self, 'git_branch', None)
        )
        cx, cy, cz = self.center
        r = self.radius
        if hasattr(self, 'arousal') and self.arousal > 0:
            rng = np.random.default_rng(42)
            spike_field = np.zeros_like(x)
            noise_field = np.zeros_like(x)
            if distortion_mode in ('both', 'spikes'):
                n_spikes = int(8 + 24 * self.arousal)
                spike_amp = self.arousal * 1.2 * r
                spike_sigma = 0.03 + 0.07 * self.arousal
                spike_centers = rng.uniform(0, 1, size=(n_spikes, 2))
                for cy_spike, cx_spike in spike_centers:
                    dist2 = (x - cx_spike) ** 2 + (y - cy_spike) ** 2
                    spike_field += spike_amp * np.exp(-dist2 / (2 * spike_sigma ** 2))
            if distortion_mode in ('both', 'noise'):
                noise = rng.normal(0, 1, x.shape)
                from scipy.ndimage import gaussian_filter
                noise = gaussian_filter(noise, sigma=4)
                noise_amp = self.arousal * 0.7 * r
                noise_field = noise_amp * noise
            distortion = spike_field + noise_field
            r_eff = r + distortion
        else:
            r_eff = r
        dx2 = (x - cx) ** 2
        dy2 = (y - cy) ** 2
        inside = dx2 + dy2 <= r_eff ** 2
        z_front = cz + np.sqrt(np.clip(r_eff ** 2 - dx2 - dy2, 0, None))
        depth = np.zeros_like(x)
        if np.any(inside):
            min_z = np.min(z_front[inside])
            max_z = np.max(z_front[inside])
            if max_z > min_z:
                depth[inside] = (z_front[inside] - min_z) / (max_z - min_z)
            else:
                depth[inside] = 1.0
        return depth

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    sc = SphereControl(center=(0.5, 0.5, 0.5), radius=0.2, field_shape=(32, 32, 32))
    sc.set_arousal(0.0)
    highres = (512, 512)
    depth_map = sc.get_depth_map(image_shape=highres)
    plt.imshow(depth_map, cmap='gray')
    plt.title('Sphere Depth Map (Arousal=0, High Resolution)')
    plt.colorbar()
    plt.show()
    # Animate sphere radius and arousal
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(np.zeros(highres), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Sinusoidal Sphere Depth Map with Arousal (High Resolution)')
    def update(frame):
        r = 0.25 + 0.2 * np.sin(2 * np.pi * frame / 60)
        arousal = 0.5 + 0.5 * np.sin(2 * np.pi * frame / 120)
        sc.set_radius(r)
        sc.set_arousal(arousal)
        depth_map = sc.get_depth_map(image_shape=highres)
        img.set_data(depth_map)
        ax.set_title(f'Sphere Depth Map (radius={r:.2f}, arousal={arousal:.2f})')
        return [img]
    ani = animation.FuncAnimation(fig, update, frames=120, interval=100, blit=True)
    plt.show()
