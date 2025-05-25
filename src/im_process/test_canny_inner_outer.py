import cv2
import numpy as np
from process import Canny
from effects import CannyEffects
import matplotlib.pyplot as plt

# --- Canny on person.png ---
img_path = r"C:/Users/corre/ComfyUI/custom_nodes/PIC-2025/src/stock/person.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
edge = Canny().apply(img)

# --- Find only the outermost contours ---
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outer_only = np.zeros_like(img)
cv2.drawContours(outer_only, contours, -1, (255,0,0), 1)  # Blue for outer

# --- MiDaS on person.png ---
try:
    from process import Midas
    midas = Midas()
    print(f"Running MiDaS on {img_path}...")
    depth = midas.predict(img)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
except Exception as e:
    print(f"MiDaS test failed: {e}")
    depth_vis = np.zeros_like(img)

# --- Display all results in a grid: [original, outer, depth] ---
def resize(im, size=(256,256)):
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA)

row = [
    resize(img),
    resize(outer_only),
    resize(depth_vis)
]
all_disp = np.hstack(row)

cv2.namedWindow("Canny/MiDaS on person.png", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny/MiDaS on person.png", 900, 300)
cv2.imshow("Canny/MiDaS on person.png", all_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
