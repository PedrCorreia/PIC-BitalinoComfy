import cv2
import torch
import numpy as np
import time

def main():
    image_path = "src\controllnet\multi_sphere_render.png" #"src/controllnet/sphere_render.png"
    output_path = "depth_output2.png"
    print(f"Estimating depth for {image_path}...")
    try:
        estimate_depth(image_path, output_path)
    except Exception as e:
        print(f"Error: {e}")

def estimate_depth(image_path, output_path="depth_output.png"):
    # Use MiDaS DPT_Hybrid for better speed/accuracy tradeoff (or MiDaS v2.1 small for fastest)
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')  # Already using the fastest model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas.to(device)
    midas.eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found or cannot be opened: {image_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framebatch = transform(frame_rgb).to(device)
    with torch.no_grad():
        t0 = time.time()
        prediction = midas(framebatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()
        output = (output - np.min(output)) / (np.max(output) - np.min(output))
        t1 = time.time()
        print(f"Depth estimation time: {t1-t0:.3f}s")
    mask = output > 0
    if np.any(mask):
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = output[y0:y1, x0:x1]
    else:
        cropped = output
    small_output = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)
    output_img = (small_output * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_img)
    print(f"Depth map saved to {output_path}")
    cv2.imshow("Depth Output", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        estimate_depth(sys.argv[1])
    elif len(sys.argv) == 3:
        estimate_depth(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python fast_mida_depth.py [image_path] [output_path]")