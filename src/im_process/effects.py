import cv2
import numpy as np

class CannyEffects:

    @staticmethod
    def validate_canny_image(edge_img):
        """Ensure the input is a single-channel uint8 Canny edge image."""
        if edge_img.ndim != 2 or edge_img.dtype != np.uint8:
            raise ValueError("Input must be a single-channel uint8 Canny edge image.")

    def canny_with_variable_noise(self, edge_img, noise_level=10):
        """Add variable Gaussian noise only to the Canny edge pixels (expects Canny edge image)."""
        self.validate_canny_image(edge_img)
        noisy = edge_img.copy().astype(np.float32)
        mask = edge_img > 0
        noise = np.random.normal(0, noise_level, edge_img.shape)
        noisy[mask] += noise[mask]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def canny_edge_distance_map(self, edge_img):
        """Return a distance transform from the Canny edges (expects Canny edge image)."""
        self.validate_canny_image(edge_img)
        inv_edges = cv2.bitwise_not(edge_img)
        dist = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
        return dist_norm

    def canny_with_gaussian_peaks(self, edge_img, peak_height=255, peak_sigma=2):
        """Add Gaussian peaks centered at each edge pixel (expects Canny edge image)."""
        self.validate_canny_image(edge_img)
        h, w = edge_img.shape
        result = np.zeros_like(edge_img, dtype=np.float32)
        # Find edge coordinates
        ys, xs = np.where(edge_img > 0)
        # Precompute a small Gaussian kernel
        size = int(peak_sigma * 6) + 1
        gk = cv2.getGaussianKernel(size, peak_sigma)
        gauss2d = gk @ gk.T
        gauss2d = gauss2d / gauss2d.max() * peak_height
        half = size // 2
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-half), min(h, y+half+1)
            x0, x1 = max(0, x-half), min(w, x+half+1)
            gy0, gy1 = half-(y-y0), half+(y1-y)
            gx0, gx1 = half-(x-x0), half+(x1-x)
            result[y0:y1, x0:x1] += gauss2d[gy0:gy1, gx0:gx1]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def canny_with_fast_transform(self, edge_img, max_shift=5):
        """Randomly shift edge pixels within a max_shift radius to quickly distort the Canny edges."""
        self.validate_canny_image(edge_img)
        h, w = edge_img.shape
        result = np.zeros_like(edge_img)
        ys, xs = np.where(edge_img > 0)
        for y, x in zip(ys, xs):
            dy = np.random.randint(-max_shift, max_shift+1)
            dx = np.random.randint(-max_shift, max_shift+1)
            ny, nx = np.clip(y+dy, 0, h-1), np.clip(x+dx, 0, w-1)
            result[ny, nx] = 255
        return result

    def canny_pixelate(self, edge_img, block_size=8):
        """Pixelate the edge image for a digital blocky effect."""
        self.validate_canny_image(edge_img)
        h, w = edge_img.shape
        temp = cv2.resize(edge_img, (w//block_size, h//block_size), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated

    def canny_glitch_shift(self, edge_img, max_band_height=10, max_shift=30):
        """Randomly shift horizontal bands of the edge image for a glitch effect."""
        self.validate_canny_image(edge_img)
        result = edge_img.copy()
        h, w = edge_img.shape
        y = 0
        while y < h:
            band_height = np.random.randint(1, max_band_height+1)
            shift = np.random.randint(-max_shift, max_shift+1)
            y_end = min(y+band_height, h)
            band = result[y:y_end, :]
            band = np.roll(band, shift, axis=1)
            result[y:y_end, :] = band
            y += band_height
        return result
