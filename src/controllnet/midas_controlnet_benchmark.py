import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Import MiDaS depth estimator
sys.path.append(os.path.join(os.path.dirname(__file__), '../im_process'))
from process import Midas
# Import ControlNet minimal interface
from sdxl_controllnet_minimal import generate_with_controlnet, init_controlnet_pipeline

def midas_controlnet_benchmark(
    input_image_path=None,
    prompts=None,
    controlnet_scales=[0.2, 0.5, 0.8, 1.0],
    num_steps=2
):
    """
    Benchmark: Use MiDaS (small) to generate a depth map, then use ControlNet Depth SDXL 1.0 small with SDXL Turbo
    to generate images from the depth map at different scales.
    """
    if prompts is None:
        prompts = [
            "A 4k re-imagined restored greek beautiful temple with mountain landscape with a river with greek mitology creatures with galactic night sky "
        ]
    # Load input image
    if input_image_path and os.path.exists(input_image_path):
        print(f"Loading input image from: {input_image_path}")
        original_image = Image.open(input_image_path).convert("RGB")
    else:
        raise FileNotFoundError("Input image path must be provided and exist.")
    # Resize if needed
    if original_image.size != (512, 512):
        original_image = original_image.resize((512, 512), Image.LANCZOS)
    # Generate MiDaS depth map
    print("Generating MiDaS depth map...")
    midas = Midas()
    img_array = np.array(original_image)
    depth_map = midas.predict(img_array)
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map_uint8 = (depth_map_norm * 255).astype(np.uint8)
    depth_map_img = Image.fromarray(depth_map_uint8)
    depth_path = "midas_controlnet_input_depth.png"
    depth_map_img.save(depth_path)
    print(f"Saved input depth map to {depth_path}")
    # Initialize ControlNet pipeline (checkpoint)
    print("Initializing ControlNet pipeline (this may take a while the first time)...")
    pipeline = init_controlnet_pipeline(controlnet_type="depth")
    print("Pipeline loaded and ready.")
    # Generate images for each prompt and scale
    results = []
    for prompt in prompts:
        for scale in controlnet_scales:
            print(f"Generating image for prompt: '{prompt}' | scale: {scale}")
            start_time = time.time()
            gen_img = generate_with_controlnet(
                pipeline=pipeline,
                prompt=prompt,
                control_image=depth_map_img,
                num_steps=num_steps,
                controlnet_conditioning_scale=scale,
                output_path=None,
                apply_canny=False
            )
            elapsed = time.time() - start_time
            results.append((prompt, scale, gen_img, elapsed))
            print(f"Done in {elapsed:.2f}s")
    # Plot results
    n_prompts = len(prompts)
    n_scales = len(controlnet_scales)
    plt.figure(figsize=(4 + n_scales * 4, n_prompts * 4))
    for i, prompt in enumerate(prompts):
        plt.subplot(n_prompts, n_scales + 2, i * (n_scales + 2) + 1)
        plt.imshow(np.array(original_image))
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(n_prompts, n_scales + 2, i * (n_scales + 2) + 2)
        plt.imshow(np.array(depth_map_img), cmap='gray')
        plt.title("MiDaS Depth Map")
        plt.axis('off')
        for j, scale in enumerate(controlnet_scales):
            idx = i * n_scales + j
            _, _, gen_img, elapsed = results[idx]
            plt.subplot(n_prompts, n_scales + 2, i * (n_scales + 2) + 3 + j)
            plt.imshow(np.array(gen_img))
            plt.title(f"Scale: {scale} | {elapsed:.2f}s")  # <-- scale and time on same line
            plt.axis('off')
    # Set overall title and save the result
    if results:
        avg_time = sum([r[3] for r in results]) / len(results)
        plt.suptitle(
            f"ControlNet SDXL Turbo (Depth) with Different Scales\nPrompt: '{prompts[0]}'\n{num_steps} steps, Avg. time: {avg_time:.2f}s",
            fontsize=14
        )
    else:
        plt.suptitle(
            f"ControlNet SDXL Turbo (Depth) with Different Scales\nPrompt: '{prompts[0]}'\nBenchmark failed",
            fontsize=14
        )
    plt.tight_layout()
    plt.savefig("midas_controlnet_benchmark.png")
    print("Saved benchmark plot to midas_controlnet_benchmark.png")
    # plt.show()  # Comment out if running headless

if __name__ == "__main__":
    
    input_image_path = "/media/lugo/data/ComfyUI/custom_nodes/PIC-BitalinoComfy/src/controllnet/image.png"
    midas_controlnet_benchmark(input_image_path=input_image_path)
