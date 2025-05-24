"""
Field transformation functions for use in the demo UI.
"""
import numpy as np
from scipy.ndimage import map_coordinates

def radial_distort(image, strength=0.5, center=None, order=1):
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    max_r = np.max(r)
    # Support per-pixel strength
    if isinstance(strength, np.ndarray):
        factor = 1 + strength * (r / max_r) ** 2 * 2
    else:
        factor = 1 + strength * (r / max_r) ** 2 * 2
    x_new = cx + x * factor
    y_new = cy + y * factor
    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)
    coords = np.stack([y_new, x_new], axis=-1)
    if image.ndim == 2:
        return map_coordinates(image, [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
    else:
        out = np.zeros_like(image)
        for c in range(image.shape[2]):
            out[...,c] = map_coordinates(image[...,c], [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
        return out

def spin_distort(image, strength=0.5, center=None, order=1):
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    max_r = np.max(r)
    # Support per-pixel strength
    if isinstance(strength, np.ndarray):
        theta_new = theta + strength * (r / max_r) * 2 * np.pi * 1.5
    else:
        theta_new = theta + strength * (r / max_r) * 2 * np.pi * 1.5
    x_new = r * np.cos(theta_new) + cx
    y_new = r * np.sin(theta_new) + cy
    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)
    coords = np.stack([y_new, x_new], axis=-1)
    if image.ndim == 2:
        return map_coordinates(image, [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
    else:
        out = np.zeros_like(image)
        for c in range(image.shape[2]):
            out[...,c] = map_coordinates(image[...,c], [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
        return out

def blackhole_distort(image, strength=0.5, center=None, order=1):
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x0 = x - cx
    y0 = y - cy
    r = np.sqrt(x0**2 + y0**2)
    max_r = np.max(r)
    ring_radius = 0.25 * max_r
    ring_width = 0.04 * max_r
    ring_mask = np.exp(-((r - ring_radius) ** 2) / (2 * ring_width ** 2))
    theta = np.arctan2(y0, x0)
    # Support per-pixel strength
    if isinstance(strength, np.ndarray):
        spin = strength * ring_mask
    else:
        spin = strength * ring_mask
    theta_new = theta + spin
    x_new = np.where(ring_mask > 0.01, r * np.cos(theta_new), x0) + cx
    y_new = np.where(ring_mask > 0.01, r * np.sin(theta_new), y0) + cy
    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)
    coords = np.stack([y_new, x_new], axis=-1)
    if image.ndim == 2:
        return map_coordinates(image, [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
    else:
        out = np.zeros_like(image)
        for c in range(image.shape[2]):
            out[...,c] = map_coordinates(image[...,c], [coords[...,0].ravel(), coords[...,1].ravel()], order=order).reshape(h, w)
        return out

# For vector field visualization (displacement)
def get_displacement_field(transform, shape, strength=0.5, center=None):
    h, w = shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x0 = x - cx
    y0 = y - cy
    if transform == 'radial':
        r = np.sqrt(x0**2 + y0**2)
        max_r = np.max(r)
        factor = 1 + strength * (r / max_r) ** 2
        x_new = x0 * factor
        y_new = y0 * factor
    elif transform == 'spin':
        r = np.sqrt(x0**2 + y0**2)
        theta = np.arctan2(y0, x0)
        max_r = np.max(r)
        theta_new = theta + strength * (r / max_r) * 2 * np.pi
        x_new = r * np.cos(theta_new)
        y_new = r * np.sin(theta_new)
    elif transform == 'blackhole':
        r = np.sqrt(x0**2 + y0**2)
        max_r = np.max(r)
        ring_radius = 0.25 * max_r
        ring_width = 0.04 * max_r
        ring_mask = np.exp(-((r - ring_radius) ** 2) / (2 * ring_width ** 2))
        theta = np.arctan2(y0, x0)
        spin = strength * ring_mask
        theta_new = theta + spin
        # Only apply spin at the ring, zero elsewhere
        x_new = np.where(ring_mask > 0.01, r * np.cos(theta_new), x0)
        y_new = np.where(ring_mask > 0.01, r * np.sin(theta_new), y0)
    else:
        x_new = x0
        y_new = y0
    dx = x_new - x0
    dy = y_new - y0
    return x, y, dx, dy
