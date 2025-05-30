import numpy as np
from PIL import Image
import torch
from rtd_comfy.sdxl_turbo.diffusion_engine import DiffusionEngine

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

# --- 2. Setup diffusion engine and generate image ---
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1. Make depth map
    depth_img = make_sphere_depth(size=512, radius=0.8, depth=0.7, device=device)
    depth_img.save('sphere_depth.png')
    print('Saved sphere_depth.png')

    # 2. Prepare diffusion engine
    engine = DiffusionEngine(
        use_image2image=False,
        height_diffusion_desired=512,
        width_diffusion_desired=512,
        device=device,
        hf_model="stabilityai/sdxl-turbo"
    )

    # 3. Set ControlNet input (depth)
    arr = torch.from_numpy(np.array(depth_img)).float() / 255.0
    arr = arr.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    arr = arr.cpu().numpy()[0, 0]  # (H,W)
    engine.set_input_image(arr)

    # 4. Prompts
    prompt = "a futuristic crystalline sphere, highly detailed, 8k, photorealistic"
    negative_prompt = "blurry, low quality, distorted, flat"
    engine.set_embeddings([prompt]*4)
    if hasattr(engine, 'set_negative_embeddings'):
        engine.set_negative_embeddings([negative_prompt]*4)

    # 5. Generation params
    engine.set_num_inference_steps(4)
    engine.set_guidance_scale(7.5)
    engine.set_strength(1.2)

    # 6. Generate
    result = engine.generate()
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().clamp(0, 1)
        result = (result * 255).to(torch.uint8).numpy().transpose(1, 2, 0)
        result = Image.fromarray(result)
    result.save('sphere_controlnet_result.png')
    print('Saved sphere_controlnet_result.png')

if __name__ == "__main__":
    main()
