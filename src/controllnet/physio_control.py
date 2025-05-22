"""
PhysioShapeControl: Map physiological signals to dynamic geometric control maps for ControlNet/diffusion.

- Input: physiological signal (float or array)
- Output: 2D depth map (numpy array) for use as ControlNet input
- Example mapping: signal controls sphere radius and position

"""
import numpy as np

class PhysioShapeControl:
    def __init__(self, field_shape=(128, 128)):
        self.field_shape = field_shape
        self.center = np.array([0.5, 0.5], dtype=np.float32)
        self.radius = 0.2

    def update_from_signal(self, signal_value, arousal=0.0):
        """
        Map the physiological signal to sphere radius.
        signal_value: float in [0, 1] (normalized)
        arousal: float in [0, 1] (0=sphere, 1=fully distorted)
        """
        self.radius = 0.05 + 0.4 * signal_value
        self.center = np.array([0.5, 0.5], dtype=np.float32)
        self.arousal = float(np.clip(arousal, 0, 1))

    def get_depth_map(self):
        h, w = self.field_shape
        y, x = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )
        cx, cy = self.center
        r = self.radius
        # Add arousal-based geometric distortion
        if hasattr(self, 'arousal') and self.arousal > 0:
            # 3 random geometric distortions: spikes, waves, and bulges
            theta = np.arctan2(y - cy, x - cx)
            rng = np.random.default_rng(42)
            n_vertices = 8
            geo_pattern = np.zeros_like(x)
            # 1. Spikes (star-like)
            amplitude1 = 0.3 * self.arousal * r
            phase1 = rng.uniform(0, 2*np.pi)
            geo_pattern += amplitude1 * np.cos(n_vertices * theta + phase1)
            # 2. Waves (low-freq undulation)
            amplitude2 = 0.2 * self.arousal * r
            freq2 = 3
            phase2 = rng.uniform(0, 2*np.pi)
            geo_pattern += amplitude2 * np.sin(freq2 * theta + phase2)
            # 3. Bulges (random smooth noise)
            noise = rng.normal(0, 1, x.shape)
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=8)
            amplitude3 = 0.15 * self.arousal * r
            geo_pattern += amplitude3 * noise
            # Apply distortion
            r_eff = r + geo_pattern
        else:
            r_eff = r
        dx2 = (x - cx) ** 2
        dy2 = (y - cy) ** 2
        inside = dx2 + dy2 <= r_eff ** 2
        dist = np.sqrt(dx2 + dy2)
        depth = np.zeros_like(x)
        if np.any(inside):
            min_d = np.min(dist[inside])
            max_d = np.max(dist[inside])
            if max_d > min_d:
                depth[inside] = 1.0 - (dist[inside] - min_d) / (max_d - min_d)
            else:
                depth[inside] = 1.0
        return depth

    def get_3d_mask(self):
        """
        Returns a 3D numpy array (mask) with 1.0 inside the distorted sphere, 0.0 outside.
        The distortion is applied in 3D using the same geometric pattern as in 2D, but extended to all angles.
        """
        if len(self.field_shape) == 3:
            d, h, w = self.field_shape
        else:
            d, h, w = 64, 64, 64
        z, y, x = np.meshgrid(
            np.linspace(0, 1, d),
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )
        cx, cy, cz = 0.5, 0.5, 0.5
        r = self.radius
        # Spherical coordinates
        dx = x - cx
        dy = y - cy
        dz = z - cz
        r0 = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arctan2(dy, dx)
        phi = np.arccos(np.clip(dz / (r0 + 1e-8), -1, 1))
        # 3D geometric distortion
        if hasattr(self, 'arousal') and self.arousal > 0:
            rng = np.random.default_rng(42)
            n_vertices = 8
            geo_pattern = np.zeros_like(x)
            # 1. Spikes (star-like in 3D)
            amplitude1 = 0.3 * self.arousal * r
            phase1 = rng.uniform(0, 2*np.pi)
            geo_pattern += amplitude1 * np.cos(n_vertices * theta + phase1)
            # 2. Waves (low-freq undulation in phi)
            amplitude2 = 0.2 * self.arousal * r
            freq2 = 3
            phase2 = rng.uniform(0, 2*np.pi)
            geo_pattern += amplitude2 * np.sin(freq2 * phi + phase2)
            # 3. Bulges (random smooth noise in 3D)
            noise = rng.normal(0, 1, x.shape)
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=6)
            amplitude3 = 0.15 * self.arousal * r
            geo_pattern += amplitude3 * noise
            r_eff = r + geo_pattern
        else:
            r_eff = r
        mask = (r0 <= r_eff).astype(np.float32)
        return mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    frames = 120
    n_theta, n_phi = 128, 64
    img_res = 512  # Output depth map resolution
    psc = PhysioShapeControl(field_shape=(n_phi, n_theta))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.zeros((img_res, img_res)), cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal')
    ax.set_title('Depth Map (z-buffer) of Distorted Sphere')
    ax.axis('off')

    def update_depth(frame):
        # Animate radius and arousal
        signal = 0.5 + 0.1 * np.sin(2 * np.pi * frame / frames)
        arousal = 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * frame / (0.3*frames)))
        psc.update_from_signal(signal, arousal=arousal)
        # Spherical coordinates
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        r = psc.radius
        # Arousal-based geometric distortion (randomized per frame)
        rng = np.random.default_rng(frame)
        n_vertices = 8
        amplitude1 = 0.3 * psc.arousal * r
        phase1 = rng.uniform(0, 2*np.pi)
        spikes = amplitude1 * np.cos(n_vertices * theta_grid + phase1)
        amplitude2 = 0.2 * psc.arousal * r
        freq2 = 3
        phase2 = rng.uniform(0, 2*np.pi)
        waves = amplitude2 * np.sin(freq2 * phi_grid + phase2)
        amplitude3 = 0.15 * psc.arousal * r
        noise = rng.normal(0, 1, theta_grid.shape)
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=4)
        bulges = amplitude3 * noise
        r_eff = r + spikes + waves + bulges
        # Convert to cartesian (centered at 0,0,0)
        x = r_eff * np.sin(phi_grid) * np.cos(theta_grid)
        y = r_eff * np.sin(phi_grid) * np.sin(theta_grid)
        z = r_eff * np.cos(phi_grid)
        # Project to image plane (orthographic, x/y in [-r, r])
        img = np.zeros((img_res, img_res), dtype=np.float32)
        count = np.zeros((img_res, img_res), dtype=np.int32)
        # Normalize x/y to [0, img_res-1]
        x_img = ((x - x.min()) / (x.max() - x.min()) * (img_res - 1)).astype(int)
        y_img = ((y - y.min()) / (y.max() - y.min()) * (img_res - 1)).astype(int)
        # For each point, keep the max z (closest to camera)
        for i in range(x_img.shape[0]):
            for j in range(x_img.shape[1]):
                xi, yi = x_img[i, j], y_img[i, j]
                if 0 <= xi < img_res and 0 <= yi < img_res:
                    if count[yi, xi] == 0 or z[i, j] > img[yi, xi]:
                        img[yi, xi] = z[i, j]
                        count[yi, xi] = 1
        # Normalize depth to [0, 1] (1=closest, 0=farthest), background stays 0
        mask = count > 0
        if np.any(mask):
            zmin, zmax = img[mask].min(), img[mask].max()
            img[mask] = (img[mask] - zmin) / (zmax - zmin + 1e-8)
        img[~mask] = 0.0  # Ensure background is black
        im.set_data(img)
        ax.set_title(f'Depth Map (signal={signal:.2f}, arousal={arousal:.2f})')
        return [im]

    ani = animation.FuncAnimation(fig, update_depth, frames=frames, interval=60, blit=True)
    plt.show()
