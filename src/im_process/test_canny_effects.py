import cv2
import numpy as np
from process import Canny 
from effects import CannyEffects
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"C:/Users/corre/ComfyUI/custom_nodes/PIC-2025/image.png")
if img is None:
    raise FileNotFoundError("Image not found. Check the path.")
edge_img = Canny().apply(img)
# Create CannyEffects instance
canny_fx = CannyEffects()

# Apply each effect
noisy = canny_fx.canny_with_variable_noise(edge_img, noise_level=20)
dist_map = canny_fx.canny_edge_distance_map(edge_img)
gauss_peaks = canny_fx.canny_with_gaussian_peaks(edge_img, peak_height=255, peak_sigma=2)
fast_transform = canny_fx.canny_with_fast_transform(edge_img, max_shift=10)

# Prepare images for display (convert to 3-channel for stacking)
def to_bgr(im):
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im

# Add label to each tile
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 1
LABEL_BG = (0, 0, 0)

def add_label(im, text):
    im = im.copy()
    (w, h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(im, (0, 0), (w+6, h+6), LABEL_BG, -1)
    cv2.putText(im, text, (3, h+3), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return im

# Resize all images to a fixed size for display
DISPLAY_SIZE = (640, 640)

def resize(im):
    return cv2.resize(im, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)

def add_border(im, color=(0,255,0)):
    return cv2.copyMakeBorder(im, 2,2,2,2, cv2.BORDER_CONSTANT, value=color)

img_disp = add_label(add_border(resize(to_bgr(img))), "Original")
edge_disp = add_label(add_border(resize(to_bgr(edge_img))), "Canny")
noisy_disp = add_label(add_border(resize(to_bgr(noisy))), "Noisy")
dist_disp = add_label(add_border(resize(to_bgr(dist_map))), "Distance")
gauss_disp = add_label(add_border(resize(to_bgr(gauss_peaks))), "Gaussian Peaks")
fast_transform_disp = add_label(add_border(resize(to_bgr(fast_transform))), "Fast Transform")

# Stack all results horizontally
row1 = np.hstack([img_disp, edge_disp])
row2 = np.hstack([noisy_disp, dist_disp])
row3 = np.hstack([gauss_disp, fast_transform_disp])
all_disp = np.vstack([row1, row2, row3])

cv2.namedWindow("Canny Effects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny Effects", 1600, 1200)
cv2.imshow("Canny Effects", all_disp)
cv2.moveWindow("Canny Effects", 100, 100)
cv2.waitKey(1)  # Let the window appear
print("Press any key in the window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fallback: also show with matplotlib for better quality/zoom
plt.figure(figsize=(18, 18))
plt.imshow(cv2.cvtColor(all_disp, cv2.COLOR_BGR2RGB))
plt.title("Canny Effects")
plt.axis('off')
plt.show()
