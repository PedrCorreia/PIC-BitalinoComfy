"""
Field Transformations and Visualization UI

This script provides a simple dark mode UI for applying field transformations to images and visualizing the results live.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import RadioButtons
from field_transforms import radial_distort, spin_distort, blackhole_distort, get_displacement_field
from matplotlib.image import imread
import concurrent.futures
import threading
from skimage.color import rgb2gray
from skimage.feature import canny

mpl.rcParams['figure.facecolor'] = '#222'
mpl.rcParams['axes.facecolor'] = '#222'
mpl.rcParams['axes.edgecolor'] = '#888'
mpl.rcParams['axes.labelcolor'] = '#fff'
mpl.rcParams['xtick.color'] = '#fff'
mpl.rcParams['ytick.color'] = '#fff'
mpl.rcParams['text.color'] = '#fff'
mpl.rcParams['figure.edgecolor'] = '#222'
mpl.rcParams['savefig.facecolor'] = '#222'

def main():
    # Load image from images/ folder
    img_path = os.path.join(os.path.dirname(__file__), 'images', 'image.png')
    if os.path.exists(img_path):
        image = imread(img_path)
        if image.dtype == np.uint8:
            image = image / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3]  # drop alpha
    else:
        from skimage import data
        image = data.astronaut()
        image = image / 255.0

    # --- Add a grid overlay for visual testing ---
    def add_grid_overlay(img, grid_spacing=32, grid_color=(1,0,0), thickness=1):
        h, w = img.shape[:2]
        grid_img = img.copy()
        # Draw vertical lines
        for x in range(0, w, grid_spacing):
            grid_img[:, x:x+thickness, 0] = grid_color[0]
            grid_img[:, x:x+thickness, 1] = grid_color[1]
            grid_img[:, x:x+thickness, 2] = grid_color[2]
        # Draw horizontal lines
        for y in range(0, h, grid_spacing):
            grid_img[y:y+thickness, :, 0] = grid_color[0]
            grid_img[y:y+thickness, :, 1] = grid_color[1]
            grid_img[y:y+thickness, :, 2] = grid_color[2]
        return grid_img

    # --- Add depth map and edge detection utilities ---
    def get_depth_map(img):
        # Use grayscale intensity as a simple depth proxy (0=far, 1=near)
        if img.shape[-1] == 3:
            depth = rgb2gray(img)
        else:
            depth = img
        return depth
    def get_edge_map(img):
        # Use Canny edge detection
        if img.shape[-1] == 3:
            gray = rgb2gray(img)
        else:
            gray = img
        edges = canny(gray, sigma=2)
        return edges

    # Store the original image without grid
    orig_image = image.copy()
    
    # Calculate edge map and depth map from the original image (without grid)
    depth_map = get_depth_map(orig_image)
    edge_map = get_edge_map(orig_image)
    
    # Create display image with grid for visualization
    display_image = add_grid_overlay(image, grid_spacing=max(16, min(image.shape[0], image.shape[1])//20), grid_color=(1,0,0), thickness=1)

    # UI setup
    fig, ax = plt.subplots(figsize=(7,7))
    fig.patch.set_facecolor('#222')
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_disp = ax.imshow(display_image, vmin=0, vmax=1)
    ax.set_title('Field Transformation Demo', color='#fff')
    ax.axis('off')
    
    # Add center marker for visualization (hidden initially)
    center_marker = ax.plot([], [], 'o', color='#ff9800', markersize=10)[0]
    center_marker.set_visible(False)

    # Slider for distortion strength
    axcolor = '#333'
    ax_strength = plt.axes((0.3, 0.1, 0.6, 0.03), facecolor=axcolor)
    strength_slider = Slider(ax_strength, 'Strength', -2.0, 2.0, valinit=0.0, color='#0af', valstep=0.01)

    # Button to reset
    ax_reset = plt.axes((0.85, 0.02, 0.13, 0.05))
    reset_button = Button(ax_reset, 'Reset', color='#444', hovercolor='#666')

    # Button to toggle vector field visualization
    ax_toggle = plt.axes((0.65, 0.02, 0.18, 0.05))
    toggle_button = Button(ax_toggle, 'Show Vectors', color='#444', hovercolor='#666')
    show_vectors = [False]
    global_quiver_obj = None

    # Button to toggle noise for testing
    ax_noise = plt.axes((0.45, 0.02, 0.18, 0.05))
    noise_button = Button(ax_noise, 'Test with Noise', color='#444', hovercolor='#666')
    test_with_noise = [False]

    # Radio buttons for field type
    ax_radio = plt.axes((0.05, 0.4, 0.15, 0.2), facecolor='#222')
    radio = RadioButtons(ax_radio, ('radial', 'spin', 'blackhole'), label_props={'color': 'w'}, radio_props={'s': 0.8})
    field_type = ['radial']

    # UI state for overlays
    show_depth = [False]
    show_edges = [False]

    # Add buttons for depth and edge overlays
    ax_depth = plt.axes((0.05, 0.32, 0.15, 0.05))
    depth_button = Button(ax_depth, 'Show Depth', color='#444', hovercolor='#666')
    ax_edges = plt.axes((0.05, 0.25, 0.15, 0.05))
    edges_button = Button(ax_edges, 'Show Edges', color='#444', hovercolor='#666')

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    update_lock = threading.Lock()

    class PendingFuturesObj:
        def __init__(self):
            self.image = object()
            self.vector = object()
    pending_futures = PendingFuturesObj()

    # --- UI setup (move slider clear to here, not in update) ---
    ax_strength.clear()
    strength_slider = Slider(ax_strength, 'Strength', -2.0, 2.0, valinit=0.0, color='#0af', valstep=0.01)

    def compute_displacement_and_coords(tname, h, w, strength):
        # Always use the same logic for both image and vector field
        x, y, dx, dy = get_displacement_field(tname, (h, w), strength=strength)
        # For image warping, we need to find for each output pixel (i,j) the source pixel (i',j')
        # If the field is (dx, dy), then the source is (i - dy, j - dx)
        coords_y = (y - dy).clip(0, h-1)
        coords_x = (x - dx).clip(0, w-1)
        return x, y, dx, dy, coords_y, coords_x    
    def update(val=None):
        nonlocal global_quiver_obj
        strength = strength_slider.val
        tname = field_type[0]
        h, w = orig_image.shape[:2]

        # Compute object center for smart transformations and vector fields
        edges = edge_map
        if np.sum(edges) > 0:
            yx = np.argwhere(edges)
            center_yx = yx.mean(axis=0)
            obj_center = (center_yx[1], center_yx[0])  # (cx, cy)
        else:
            obj_center = (w/2, h/2)

        # Update center marker (shown only in edge mode)
        center_marker.set_data([obj_center[0]], [obj_center[1]])
        center_marker.set_visible(show_edges[0])

        if show_depth[0]:
            # Display grayscale depth map directly (brighter = closer)
            img_disp.set_data(depth_map)
            img_disp.set_cmap('gray')  # Use grayscale colormap
            img_disp.set_visible(True)
            center_marker.set_visible(False)
            fig.canvas.draw_idle()
            return

        if show_edges[0]:
            # Show only the outer edges as white on black, with center marker
            edge_img = np.zeros_like(orig_image)
            edge_img[edge_map, :] = 1.0  # white edges
            img_disp.set_data(edge_img)
            img_disp.set_cmap(None)
            img_disp.set_visible(True)
            center_marker.set_visible(True)
            fig.canvas.draw_idle()
            return

        if show_vectors[0]:
            img_disp.set_visible(False)
            if global_quiver_obj is not None:
                global_quiver_obj.remove()
                global_quiver_obj = None
            def compute_and_draw_vector():
                # Use detected object center for vector field
                # Only show vectors inside the object mask
                from scipy.ndimage import binary_dilation
                obj_mask = binary_dilation(edges, iterations=8)
                x, y, dx, dy = get_displacement_field(tname, (h, w), strength=strength, center=obj_center)
                # Mask out vectors outside the object
                dx = np.where(obj_mask, dx, 0)
                dy = np.where(obj_mask, dy, 0)
                step = max(1, int(h // 32))
                return x, y, dx, dy, step
            def on_vector_done(fut):
                nonlocal global_quiver_obj
                try:
                    x, y, dx, dy, step = fut.result()
                except concurrent.futures.CancelledError:
                    return
                with update_lock:
                    if show_vectors[0]:
                        if global_quiver_obj is not None and hasattr(global_quiver_obj, 'remove'):
                            global_quiver_obj.remove()
                        global_quiver_obj = ax.quiver(
                            x[::step,::step], y[::step,::step],
                            dx[::step,::step], dy[::step,::step],
                            color='#0af', angles='xy', scale_units='xy', scale=1
                        )
                        fig.canvas.draw_idle()
            if hasattr(pending_futures, 'vector') and pending_futures.vector:
                try:
                    pending_futures.vector.cancel()  # type: ignore
                except Exception:
                    pass
            fut = executor.submit(compute_and_draw_vector)
            fut.add_done_callback(on_vector_done)
            pending_futures.vector = fut  # type: ignore
        else:
            if global_quiver_obj is not None:
                global_quiver_obj.remove()
                global_quiver_obj = None
            img_disp.set_visible(True)
            def compute_and_draw_image():
                base_img = orig_image.copy()
                if test_with_noise[0]:
                    noise = np.random.normal(0, 0.08, base_img.shape)
                    color_bias = np.random.uniform(-0.15, 0.15, (1, 1, 3))
                    noise = noise + color_bias
                    base_img = base_img + noise
                    base_img = np.clip(base_img, 0, 1)
                # Use edge map to identify object region
                edges = edge_map
                from scipy.ndimage import binary_dilation
                obj_mask = binary_dilation(edges, iterations=8)
                # Use the same center as calculated globally
                cx, cy = obj_center
                depth = depth_map
                local_strength = strength * (1 - depth)
                # Apply transformation to base image, only inside object mask
                if tname == 'radial':
                    img_t = radial_distort(base_img, strength=local_strength, center=(cx, cy), order=0)
                elif tname == 'spin':
                    img_t = spin_distort(base_img, strength=local_strength, center=(cx, cy), order=0)
                elif tname == 'blackhole':
                    img_t = blackhole_distort(base_img, strength=local_strength, center=(cx, cy), order=0)
                else:
                    img_t = base_img
                img_t = np.where(obj_mask[...,None], img_t, base_img)
                if test_with_noise[0]:
                    post_noise = np.random.normal(0, 0.08, img_t.shape)
                    post_color_bias = np.random.uniform(-0.15, 0.15, (1, 1, 3))
                    post_noise = post_noise + post_color_bias
                    img_t = img_t + post_noise
                    img_t = np.clip(img_t, 0, 1)
                grid_spacing = max(16, min(img_t.shape[0], img_t.shape[1])//20)
                img_t_with_grid = add_grid_overlay(img_t, grid_spacing=grid_spacing, grid_color=(1,0,0), thickness=1)
                return img_t_with_grid

            def on_image_done(fut):
                try:
                    img_t = fut.result()
                except concurrent.futures.CancelledError:
                    return
                with update_lock:
                    if not show_vectors[0]:
                        img_disp.set_data(np.clip(img_t, 0, 1))
                        fig.canvas.draw_idle()
            if hasattr(pending_futures, 'image') and pending_futures.image:
                try:
                    pending_futures.image.cancel()  # type: ignore
                except Exception:
                    pass
            fut = executor.submit(compute_and_draw_image)
            fut.add_done_callback(on_image_done)
            pending_futures.image = fut  # type: ignore

    # --- Custom style for selected buttons ---
    SELECTED_COLOR = '#ff9800'  # orange highlight
    DEFAULT_BTN_COLOR = '#444'
    DEFAULT_BTN_HOVER = '#666'
    def update_radio_colors():
        for i, label in enumerate(radio.labels):
            if radio.value_selected == radio.labels[i].get_text():
                label.set_color(SELECTED_COLOR)
            else:
                label.set_color('w')
        fig.canvas.draw_idle()
    def update_toggle_color():
        if show_vectors[0]:
            toggle_button.color = SELECTED_COLOR
            toggle_button.hovercolor = SELECTED_COLOR
        else:
            toggle_button.color = DEFAULT_BTN_COLOR
            toggle_button.hovercolor = DEFAULT_BTN_HOVER
        fig.canvas.draw_idle()
    def update_noise_color():
        if test_with_noise[0]:
            noise_button.color = SELECTED_COLOR
            noise_button.hovercolor = SELECTED_COLOR
        else:
            noise_button.color = DEFAULT_BTN_COLOR
            noise_button.hovercolor = DEFAULT_BTN_HOVER
        fig.canvas.draw_idle()
    def update_overlay_colors():
        depth_button.color = SELECTED_COLOR if show_depth[0] else DEFAULT_BTN_COLOR
        depth_button.hovercolor = SELECTED_COLOR if show_depth[0] else DEFAULT_BTN_HOVER
        edges_button.color = SELECTED_COLOR if show_edges[0] else DEFAULT_BTN_COLOR
        edges_button.hovercolor = SELECTED_COLOR if show_edges[0] else DEFAULT_BTN_HOVER
        fig.canvas.draw_idle()
    # Patch radio callback to update color
    def radio_update(label):
        field_type[0] = label
        update_radio_colors()
        update()
    radio.on_clicked(radio_update)
    # Patch toggle callback to update color
    def toggle(event):
        show_vectors[0] = not show_vectors[0]
        toggle_button.label.set_text('Show Image' if show_vectors[0] else 'Show Vectors')
        update_toggle_color()
        update()
    toggle_button.on_clicked(toggle)
    # Patch noise toggle callback
    def toggle_noise(event):
        test_with_noise[0] = not test_with_noise[0]
        update_noise_color()
        update()
    noise_button.on_clicked(toggle_noise)
    # Patch depth toggle callback
    def toggle_depth(event):
        show_depth[0] = not show_depth[0]
        if show_depth[0]:
            show_edges[0] = False
        update_overlay_colors()
        update()
    depth_button.on_clicked(toggle_depth)
    # Patch edges toggle callback
    def toggle_edges(event):
        show_edges[0] = not show_edges[0]
        show_depth[0] = False
        update_overlay_colors()
        update()
    edges_button.on_clicked(toggle_edges)
    # Initial color update
    update_radio_colors()
    update_toggle_color()
    strength_slider.reset()
    update_noise_color()
    update_overlay_colors()

    def reset(_event):
        strength_slider.reset()

    # --- UI event bindings (after all UI elements created) ---
    strength_slider.on_changed(update)
    reset_button.on_clicked(reset)
    plt.show()

if __name__ == "__main__":
    main()
