import cv2
import numpy as np
from PIL import Image
import time

def fast_canny(input_path, output_path=None, low_threshold=100, high_threshold=200):
    """
    Apply fast Canny edge detection to an image file and save or return the result.
    Args:
        input_path: Path to the input image (PNG, JPG, etc.)
        output_path: If provided, save the result to this path
        low_threshold: Lower bound for hysteresis thresholding
        high_threshold: Upper bound for hysteresis thresholding
    Returns:
        The edge-detected image as a numpy array (uint8)
    """
    img = np.array(Image.open(input_path).convert('L'))  # Convert to grayscale
    start=time.time()
    edges = cv2.Canny(img, low_threshold, high_threshold)
    if output_path:
        Image.fromarray(edges).save(output_path)
    elapsed = time.time() - start
    print(f"Fast Canny edge detection took {elapsed:.3f} seconds")
    return edges

if __name__ == "__main__":
    # Example usage: apply to the rendered sphere image
    edges = fast_canny("C:/Users/corre/ComfyUI/custom_nodes/PIC-2025/image.png", output_path="image_canny.png")
    print("Canny edge image saved as multi_sphere_render_canny.png")
#"src\controllnet\multi_sphere_render.png"