import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import sys

# --- 1. Generate a static sphere depth map ---
def make_sphere_depth(size=512, radius=0.8, depth=0.7, device='cpu'):
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
    center = size // 2
    dx = x - center
    dy = y - center
    dist = torch.sqrt(dx.float()**2 + dy.float()**2)
    sphere_r = center * radius
    mask = dist <= sphere_r
    norm_dist = dist / sphere_r
    sphere_height = torch.zeros_like(dist)
    sphere_height[mask] = torch.sqrt(1 - norm_dist[mask]**2)
    depth_map = sphere_height * depth
    depth_map = depth_map.clamp(0, 1)
    depth_map = (depth_map * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(depth_map)


def main():
    if not torch.cuda.is_available():
        print("CUDA GPU is required to run this script.")
        sys.exit(1)
    device = 'cuda'
    # 1. Make depth map
    depth_img = make_sphere_depth(size=512, radius=0.8, depth=0.7, device=device)
    depth_img.save('sphere_depth.png')
    print('Saved sphere_depth.png')

    # 2. Load ControlNet and pipeline
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16 if device=='cuda' else torch.float32
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",  # Use your checkpoint here
        controlnet=controlnet,
        torch_dtype=torch.float16 if device=='cuda' else torch.float32
    )
    pipe = pipe.to(device)

    # 3. Prompts
    prompt = "a futuristic crystalline sphere, highly detailed, 8k, photorealistic"
    negative_prompt = "blurry, low quality, distorted, flat"

    # 4. Prepare depth image for pipeline
    depth_img = depth_img.convert("L").resize((512, 512), Image.LANCZOS)

    # 5. Generate
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=depth_img,
        num_inference_steps=4,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 1.2}
    ).images[0]
    result.save('sphere_controlnet_result.png')
    print('Saved sphere_controlnet_result.png')

if __name__ == "__main__":
    main()
