import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def bokeh_blur_on_edges(img, edge_map, blur_radius):
    # Blur only where edge_map is 1, keep other regions sharp
    import cv2, numpy as np
    if blur_radius < 1:
        return img
    # Create disk kernel
    ksize = int(2 * round(blur_radius) + 1)
    y, x = np.ogrid[-ksize//2+1:ksize//2+1, -ksize//2+1:ksize//2+1]
    disk = x**2 + y**2 <= (ksize//2)**2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[disk] = 1
    kernel /= kernel.sum()
    # Blur the whole image
    img_blur = np.zeros_like(img)
    for c in range(img.shape[2]):
        img_blur[..., c] = cv2.filter2D(img[..., c], -1, kernel)
    # Create mask from edge_map (dilate for visibility)
    mask = (edge_map > 0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    # Blend: blurred where edge, sharp elsewhere
    out = np.where(mask3, img_blur, img)
    return out

def blur_strength(z, min_z=-10, max_blur=10):
    # Smooth, photographic ramp: no blur at 0, saturate at min_z
    if z >= 0:
        return 0
    if z <= min_z:
        return max_blur
    t = abs(z) / abs(min_z)
    # Use a sigmoid for a gentle ramp, saturating at max_blur
    return max_blur * (1 / (1 + math.exp(-8*(t-0.5))))

def gradient_blur_on_edges(img, edge_map, blur_radius, min_blur=0.5, edge_offset=32, center_frac=0.2, z=0):
    import cv2, numpy as np
    if blur_radius < 1:
        return img
    h, w = edge_map.shape
    edge_mask = (edge_map > 0).astype(np.uint8)
    # Get object mask by flood fill from background
    filled = edge_mask.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(filled, mask, (0,0), (1,))
    obj_mask = 1 - filled
    if np.sum(obj_mask) == 0:
        obj_mask[h//2, w//2] = 1
    # Distance from edge (inside object)
    dist_in = cv2.distanceTransform(obj_mask, cv2.DIST_L2, 3)
    # Distance from edge (outside object)
    dist_out = cv2.distanceTransform(1-obj_mask, cv2.DIST_L2, 3)
    # Parameters for band thickness (grow with |z|)
    z_abs = abs(z)
    z_max = 20
    edge_band_thick = max(4, int(2 + 8 * (z_abs / z_max)))
    inner_band_thick = max(2, int(8 + 24 * (z_abs / z_max)))
    center_thick = max(2, int(min(h, w) * max(0.04, center_frac * (1 - 0.85 * min(z_abs, z_max) / z_max))))
    # Edge band: within edge_band_thick pixels of edge (both inside and outside)
    edge_band = (((dist_in <= edge_band_thick) | (dist_out <= edge_band_thick)) & (dist_in + dist_out > 0)).astype(np.uint8)
    # Inner band: inside object, between edge_band and center
    inner_band = ((dist_in > edge_band_thick) & (dist_in <= edge_band_thick + inner_band_thick) & (obj_mask == 1)).astype(np.uint8)
    # Center: deep inside object
    center = ((dist_in > edge_band_thick + inner_band_thick) & (dist_in <= edge_band_thick + inner_band_thick + center_thick) & (obj_mask == 1)).astype(np.uint8)
    # Outer bands for margin and far background (unchanged)
    margin = edge_offset
    obj_mask_dilated = cv2.dilate(obj_mask, np.ones((margin*2, margin*2), np.uint8), iterations=1)
    outband1 = np.bitwise_and(obj_mask_dilated, 1 - obj_mask)
    outband2 = np.bitwise_and(cv2.dilate(obj_mask_dilated, np.ones((margin*2, margin*2), np.uint8), iterations=1), 1 - obj_mask_dilated)
    masks = [edge_band, inner_band, center, outband1, outband2]
    # Blur levels for each band
    blur_edge = min_blur + (blur_radius - min_blur) * (z_abs / (z_max + 1e-8)) if z < 0 else 0
    blur_inner = blur_edge * 0.7
    blur_center = min_blur
    blur_out1 = blur_radius * 0.7
    blur_out2 = blur_radius * 0.3
    levels = [blur_edge, blur_inner, blur_center, blur_out1, blur_out2]
    # Apply blur for each band
    blurred = np.zeros_like(img, dtype=np.float32)
    for mask, r in zip(masks, levels):
        if r < 1:
            region = img.copy()
        else:
            ksize = int(2 * round(r) + 1)
            y, x = np.ogrid[-ksize//2+1:ksize//2+1, -ksize//2+1:ksize//2+1]
            disk = x**2 + y**2 <= (ksize//2)**2
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[disk] = 1
            kernel /= kernel.sum()
            region = np.zeros_like(img)
            for c in range(img.shape[2]):
                region[..., c] = cv2.filter2D(img[..., c], -1, kernel)
        for c in range(img.shape[2]):
            blurred[..., c][mask.astype(bool)] = region[..., c][mask.astype(bool)]
    # Fill far background and any remaining pixels with original image
    blurred = np.where(np.repeat((obj_mask_dilated > 0)[:, :, None], 3, axis=2), blurred, img)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return blurred

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.im_process.process import Midas, Canny
from geom import Sphere
from render3d import Render3D

# Output directory for temp renders
out_dir = os.path.dirname(__file__)

# Sphere positions along z axis
z_positions = np.arange(0, -13, -2.5)
x, y = 0, 0
radius = 2
img_size = 512

render_paths = []
imgs = []
blurred_imgs = []
canny_edges = []
# Fixed camera: position, focal point, view up
camera_position = [(0, 0, 8), (0, 0, 0), (0, 0, 0)]  # (pos, focal, up)
for i, z in enumerate(z_positions):
    r3d = Render3D(img_size=img_size, background='white')
    # Use a gold color and medium-poly sphere
    gold_hex = "#D43737"  # Hex code for gold
    r3d.add_geometry(Sphere(center=(x, y, z), radius=radius, quality='medium'), color=gold_hex, opacity=1)
    img = r3d.render(output="", show_edges=False, camera_position=camera_position)
    imgs.append(img)
    # Canny edge detection for blur mask
    edge = Canny().apply(img)
    canny_edges.append(edge)
    blur = blur_strength(z)*2
    img_blur = gradient_blur_on_edges(img, edge, blur, min_blur=0, edge_offset=32, center_frac=0.12, z=z)
    blurred_imgs.append(img_blur)

# Canny visualization (outer contours only, for display)
canny_imgs = []
for edge, img in zip(canny_edges, imgs):
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_only = np.zeros_like(img)
    cv2.drawContours(outer_only, contours, -1, (255,0,0), 1)
    canny_imgs.append(outer_only)

# Depth: use blurred images for MiDaS
midas = Midas()
depth_imgs = []
for img_blur in blurred_imgs:
    if img_blur is None:
        depth_imgs.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        continue
    depth = midas.predict(img_blur, optimize_size=True)
    depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_imgs.append(depth_vis)

# Mask visualization: use distance transform for edge band
mask_imgs = []
for edge, z in zip(canny_edges, z_positions):
    h, w = edge.shape
    edge_mask = (edge > 0).astype(np.uint8)
    filled = edge_mask.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(filled, mask, (0,0), (1,))
    obj_mask = 1 - filled
    if np.sum(obj_mask) == 0:
        obj_mask[h//2, w//2] = 1
    # Calculate centroid
    ys, xs = np.nonzero(obj_mask)
    if len(xs) == 0 or len(ys) == 0:
        cx, cy = w//2, h//2
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
    Y, X = np.ogrid[:h, :w]
    dist_to_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # r is max distance from centroid inside object
    r = dist_to_center[obj_mask.astype(bool)].max() if np.any(obj_mask) else min(h, w)//2
    # Edge: r*0.9 < d <= r
    edge_band = ((dist_to_center > r*0.9) & (dist_to_center <= r) & (obj_mask == 1)).astype(np.uint8)
    # Inner: r*0.4 < d <= r*0.9
    inner_band = ((dist_to_center > r*0.4) & (dist_to_center <= r*0.9) & (obj_mask == 1)).astype(np.uint8)
    # Center: d <= r*0.4
    center = ((dist_to_center <= r*0.4) & (obj_mask == 1)).astype(np.uint8)
    # Outer bands for margin and far background (unchanged)
    margin = 32
    obj_mask_dilated = cv2.dilate(obj_mask, np.ones((margin*2, margin*2), np.uint8), iterations=1)
    outband1 = np.bitwise_and(obj_mask_dilated, 1 - obj_mask)
    outband2 = np.bitwise_and(cv2.dilate(obj_mask_dilated, np.ones((margin*2, margin*2), np.uint8), iterations=1), 1 - obj_mask_dilated)
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    mask_img[edge_band.astype(bool)] = (255, 0, 0)      # red: edge band
    mask_img[inner_band.astype(bool)] = (255, 128, 0)   # orange: inner band
    mask_img[center.astype(bool)] = (0, 255, 0)         # green: center
    mask_img[outband1.astype(bool)] = (0, 255, 255)     # cyan: outer band
    mask_img[outband2.astype(bool)] = (0, 0, 255)       # blue: far bg
    mask_imgs.append(mask_img)

# Plot: 1st row renders, 2nd row blurred, 3rd row MiDaS, 4th row Canny outer, 5th row mask
num_renders = len(z_positions)
fig, axes = plt.subplots(5, num_renders, figsize=(3*num_renders, 15))
for i in range(num_renders):
    axes[0, i].imshow(imgs[i])
    axes[0, i].set_title(f"z={z_positions[i]:.1f}")
    axes[0, i].axis('off')
    axes[1, i].imshow(blurred_imgs[i])
    axes[1, i].axis('off')
    axes[2, i].imshow(depth_imgs[i], cmap='gray')
    axes[2, i].axis('off')
    axes[3, i].imshow(canny_imgs[i])
    axes[3, i].axis('off')
    axes[4, i].imshow(mask_imgs[i])
    axes[4, i].axis('off')
axes[0, 0].set_ylabel('Render', fontsize=14)
axes[1, 0].set_ylabel('Blurred', fontsize=14)
axes[2, 0].set_ylabel('MiDaS', fontsize=14)
axes[3, 0].set_ylabel('Canny Outer', fontsize=14)
axes[4, 0].set_ylabel('Mask', fontsize=14)
plt.tight_layout()
plt.show()
