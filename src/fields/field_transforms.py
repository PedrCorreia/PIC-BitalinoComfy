"""
Field transformation functions for use in the demo UI.
"""
import numpy as np
from scipy.ndimage import map_coordinates

def radial_distort(image, strength=0.5, center=None, order=1, object_mask=None, object_center_radius=None):
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x0 = x - cx
    y0 = y - cy
    r = np.sqrt(x0**2 + y0**2)
    if object_center_radius is not None:
        obj_cx, obj_cy, obj_radius = object_center_radius
    else:
        obj_cx, obj_cy, obj_radius = cx, cy, min(h, w) / 2
    obj_r = np.sqrt((x - obj_cx)**2 + (y - obj_cy)**2)
    # Allow a soft offset outside the object (10%)
    edge_offset = 0.1 * obj_radius
    # Ramp: 0 at center, 1 at edge, 0 outside edge+offset
    ramp = np.clip((obj_r - 0) / (obj_radius + edge_offset), 0, 1)
    mask = np.clip(1 - ((obj_r - (obj_radius + edge_offset)) / edge_offset), 0, 1)
    # Outward zoom, increasing from center to edge+offset
    factor = 1 + strength * ramp * mask
    x_new = obj_cx + (x - obj_cx) * factor
    y_new = obj_cy + (y - obj_cy) * factor
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

def spin_distort(image, strength=0.5, center=None, order=1, object_mask=None, object_center_radius=None):
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    x0 = x - cx
    y0 = y - cy
    r = np.sqrt(x0**2 + y0**2)
    theta = np.arctan2(y0, x0)
    if object_center_radius is not None:
        obj_cx, obj_cy, obj_radius = object_center_radius
    else:
        obj_cx, obj_cy, obj_radius = cx, cy, min(h, w) / 2
    obj_r = np.sqrt((x - obj_cx)**2 + (y - obj_cy)**2)
    edge_offset = 0.1 * obj_radius
    # Spin ramp: 0 at center, 1 at edge, 0 outside edge+offset
    ramp = np.clip((obj_r) / (obj_radius + edge_offset), 0, 1)
    # Fade out outside the edge
    mask = np.clip(1 - ((obj_r - (obj_radius + edge_offset)) / edge_offset), 0, 1)
    # Spin is strongest at edge, zero at center, fades outside
    spin_strength = strength * ramp * mask
    theta_new = theta + spin_strength * 2 * np.pi * 1.5
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
        # All vectors point outwards, magnitude increases with r
        ramp = np.clip(r / max_r, 0, 1)
        dx = x0 / (r + 1e-8) * ramp * strength * max_r
        dy = y0 / (r + 1e-8) * ramp * strength * max_r
        # Set zero at center to avoid NaN
        dx[r == 0] = 0
        dy[r == 0] = 0
        x_new = x0 + dx
        y_new = y0 + dy
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
