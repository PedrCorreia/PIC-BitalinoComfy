# !pip install opencv-python transformers accelerate
import matplotlib
matplotlib.use("Agg")   # <-- use a headless backend
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

# download an image
image = load_image("https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

# initialize the models and pipeline on GPU
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0-small", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    'stabilityai/sdxl-turbo',
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,           # let HF place submodules on GPU
)
# or if you want everything on GPU:
pipe = pipe.to("cuda")

# build canny control image
arr = np.array(image)
edges = cv2.Canny(arr, 100, 200)
edges = np.stack([edges]*3, axis=2)
canny_image = Image.fromarray(edges)

# run on GPU
canny_tensor = torch.from_numpy(np.array(canny_image)).to("cuda").permute(2,0,1)[None]/255.0
out = pipe(
    prompt,
    controlnet_conditioning_scale=0.5,
    image=canny_tensor
)
gen = out.images[0]

# display
fig, axs = plt.subplots(1,2,figsize=(12,6))
axs[0].imshow(canny_image); axs[0].set_title("Canny"); axs[0].axis("off")
axs[1].imshow(gen);       axs[1].set_title("Generated"); axs[1].axis("off")
plt.savefig("canny_vs_gen.png")