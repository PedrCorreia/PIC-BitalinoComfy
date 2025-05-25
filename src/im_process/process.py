import cv2
import numpy as np
import torch

class Canny:
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, img):
        """Apply Canny edge detection to a grayscale or color image (numpy array)."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, self.low_threshold, self.high_threshold)
        return edges

class Midas:
    def __init__(self, model_type='MiDaS_small', device=None, force_perspective=False, min_z=-15, max_blur=8):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._loaded = False
        self.force_perspective = force_perspective
        self.min_z = min_z
        self.max_blur = max_blur

    def load_model(self):
        if self._loaded:
            return
        self.model = torch.hub.load('intel-isl/MiDaS', self.model_type)  # type: ignore
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')  # type: ignore
        if self.model_type == 'MiDaS_small':
            self.transform = getattr(transforms, 'small_transform', None)
        else:
            self.transform = getattr(transforms, 'default_transform', None)
        self._loaded = True

    def apply_perspective_blur(self, img, z):
        """Apply logarithmic Gaussian blur based on z position."""
        if z < 0:
            if z <= self.min_z:
                blur_sigma = self.max_blur
            else:
                blur_sigma = self.max_blur * (np.log1p(abs(z)) / np.log1p(abs(self.min_z)))
        else:
            blur_sigma = 0
        if blur_sigma > 0.2:
            ksize = int(2 * round(blur_sigma) + 1)
            img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
        return img

    def predict(self, img, z=None):
        """Predict depth map from an input image (numpy array, BGR or RGB). Optionally apply perspective blur if enabled."""
        self.load_model()
        import cv2
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        if self.transform is None:
            raise RuntimeError('MiDaS transform not loaded')
        input_tensor = self.transform(img_rgb).to(self.device)  # type: ignore
        if self.model is None:
            raise RuntimeError('MiDaS model not loaded')
        with torch.no_grad():
            prediction = self.model(input_tensor)  # type: ignore
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            output = prediction.cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        if self.force_perspective and z is not None:
            output = self.apply_perspective_blur(output, z)
        return output
