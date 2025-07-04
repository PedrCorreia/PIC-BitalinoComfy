from ..sdxl_turbo.diffusion_engine import DiffusionEngine, StereoDiffusionEngine
import numpy as np
from ..tools.input_image import AcidProcessor, tensor2image
import threading
import time
from PIL import Image  

class LRDiffusionEngineLoader:
    RETURN_TYPES = ("MODEL", )  
    RETURN_NAMES = ("diffusion_engine", )  
    FUNCTION = "load"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"
    DEFAULT_DO_COMPILE = False
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    DEFAULT_IMG2IMG = True
    DEFAULT_CONTROLNET_DEPTH = False

    def __init__(self):
        self.de = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_diffusion": ("FLOAT", {
                    "default": cls.DEFAULT_HEIGHT, 
                    "min": 128,
                    "max": 2048,
                    "step": 16,
                    "display": "number"
                }),
                "width_diffusion": ("FLOAT", {
                    "default": cls.DEFAULT_WIDTH, 
                    "min": 128,
                    "max": 2048,
                    "step": 16,
                    "display": "number"
                }),
                "do_compile": ("BOOLEAN", {
                    "default": cls.DEFAULT_DO_COMPILE, 
                }),
                "img2img": ("BOOLEAN", {
                    "default": cls.DEFAULT_IMG2IMG, 
                }),
                "controlnet_depth": ("BOOLEAN", {
                    "default": cls.DEFAULT_CONTROLNET_DEPTH, 
                }),
            },
            "optional": {
                "do_stereo_image": ("BOOLEAN", {"default": False}),
                }
        }

    def load(self, height_diffusion, width_diffusion, do_compile, img2img, do_stereo_image=False, auto_controlnet_depth=True, controlnet_depth=False):
        if not do_stereo_image:
            de = DiffusionEngine(use_image2image=img2img, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion, do_compile=do_compile, use_controllnet=controlnet_depth)
        else:
            de = StereoDiffusionEngine(use_image2image=img2img, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion, do_compile=do_compile, use_controllnet=controlnet_depth)
            if do_stereo_image is not None:
                de.set_stereo_image(do_stereo_image)
        
        # Store the auto_controlnet_depth setting on the engine
        de.auto_controlnet_depth = auto_controlnet_depth
        
        return ([de])
    

class LRDiffusionEngineAcid:
    DEFAULT_NUM_INFERENCE_STEPS = 2

    def __init__(self):
        self.ap = None
        self.last_diffusion_image = None

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_engine": ("MODEL", {}),
                "embeds": ("CONDITIONING", {}),
                },
            "optional": {
                "input_image": ("IMAGE", {}),
                "latents": ("LATENTS", {}),
                "decoder_embeds": ("CONDITIONING", {}),
                "num_inference_steps": ("FLOAT", {
                    "default": cls.DEFAULT_NUM_INFERENCE_STEPS, 
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "acid_strength": ("FLOAT", {
                    "default": 0.05, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),
                "acid_strength_foreground": ("FLOAT", {
                    "default": 0.01, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),                
                "coef_noise": ("FLOAT", {
                    "default": 0.15, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),
                "x_shift": ("FLOAT", {
                    "default": 0, 
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "y_shift": ("FLOAT", {
                    "default": 0, 
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "zoom_factor": ("FLOAT", {
                    "default": 1, 
                    "min": 0.5,
                    "max": 5,
                    "step": 0.01,
                    "display": "number"
                }),
                "rotation_angle": ("FLOAT", {
                    "default": 0, 
                    "min": -180,
                    "max": 180,
                    "step": 1,
                    "display": "number"
                }),
                "do_acid_tracers": ("BOOLEAN", {
                    "default": False, 
                }),
                "do_apply_humansegm_mask": ("BOOLEAN", {"default": False}),
                "human_segmentation_mask": ("IMAGE", {}),
                "do_flip_invariance": ("BOOLEAN", {"default": False}),
                "controlnet_scale": ("FLOAT", {"default": 0.5},)
                  
                }
            }
        

    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("image", )  
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def generate(
        self, 
        diffusion_engine, 
        embeds, 
        input_image=None, 
        latents=None,
        decoder_embeds=None,
        num_inference_steps=None,
        acid_strength_foreground=None,
        acid_strength=None,
        coef_noise=None,
        x_shift=None,
        y_shift=None,
        zoom_factor=None,
        rotation_angle=None,
        do_acid_tracers=None,
        do_apply_humansegm_mask=None,
        human_segmentation_mask=None,
        do_flip_invariance=None,
        controlnet_scale=None
    ):
        
        if self.ap is None:
            self.ap = AcidProcessor(device=diffusion_engine.device,
                           width_diffusion=diffusion_engine.width_diffusion,
                           height_diffusion=diffusion_engine.height_diffusion
                           )

        # Process acid first
        if acid_strength is not None:
            self.ap.set_acid_strength(acid_strength)
        if acid_strength_foreground is not None:
            self.ap.set_acid_strength_foreground(acid_strength_foreground)
        if coef_noise is not None:
            self.ap.set_coef_noise(coef_noise)
        if x_shift is not None:
            self.ap.set_x_shift(int(x_shift))
        if y_shift is not None:
            self.ap.set_y_shift(int(y_shift))
        if zoom_factor is not None:
            self.ap.set_zoom_factor(zoom_factor)
        if rotation_angle is not None:
            self.ap.set_rotation_angle(rotation_angle)
        if do_acid_tracers is not None:
            self.ap.set_do_acid_tracers(do_acid_tracers)
        if do_apply_humansegm_mask is not None:
            self.ap.set_apply_humansegm_mask(do_apply_humansegm_mask)
        if human_segmentation_mask is not None:
            self.ap.set_human_segmmask(human_segmentation_mask)      

        if do_flip_invariance is not None:
            self.ap.set_flip_invariance(do_flip_invariance)
            
        if input_image is not None:
            if hasattr(diffusion_engine, 'do_stereo_image') and diffusion_engine.do_stereo_image:
                self.ap.set_stereo_image(True)
            input_image = tensor2image(input_image)
            input_image = self.ap.process(input_image)
        else: # if no input image is provided, we take the first noise init_image that was automatically generated.
            input_image = self.ap.process(np.array(diffusion_engine.image_init))
        

        # Process DiffusionEngine
        diffusion_engine.set_embeddings(embeds)
        if input_image is not None:
            diffusion_engine.set_input_image(input_image)
        if latents is not None:
            diffusion_engine.latents(latents)
        if decoder_embeds is not None:
            diffusion_engine.set_decoder_embeddings(decoder_embeds)
        if num_inference_steps is not None:
            diffusion_engine.set_num_inference_steps(int(num_inference_steps))
        if controlnet_scale is not None:
            diffusion_engine.set_controllnet_scale(float(controlnet_scale))
        img = np.asarray(diffusion_engine.generate())
        
        # Update last image in acid processor

        if self.ap.do_flip_invariance:
            invariance_shift = np.random.randint(-10, high=10)
            img = np.roll(img, invariance_shift, axis=0)
            self.ap.flip_state -= invariance_shift
        
        self.ap.update(img.copy())
        
        if self.ap.do_flip_invariance:
            img = np.roll(img, self.ap.flip_state, axis=0)
                
        # print(f"diff size is: {img.shape}")
        img = [img]
        return img



class LRDiffusionEngineThreaded:
    DEFAULT_NUM_INFERENCE_STEPS = 2

    def __init__(self):
        self.diffusion_engine = None
        self.do_run = False
        self.num_inference_steps = self.DEFAULT_NUM_INFERENCE_STEPS
        self.embeds = None
        self.input_image = None
        self.latents = None
        self.decoder_embeds = None
        self.last_diffusion_img = Image.new("RGB", (512, 512))  # Initialize an empty PIL image with dimensions 512x512

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_engine": ("MODEL", {}),
                "embeds": ("CONDITIONING", {}),
                },
            "optional": {
                "input_image": ("IMAGE", {}),
                "latents": ("LATENTS", {}),
                "decoder_embeds": ("CONDITIONING", {}),
                "num_inference_steps": ("FLOAT", {
                    "default": cls.DEFAULT_NUM_INFERENCE_STEPS, 
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "do_run": ("BOOLEAN", {"default": True}),  # Added boolean input for do_run
            },
        }

    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("image", )  
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def _run_generation(self):
        while True:
            if not self.do_run:
                time.sleep(0.02)
                continue
            # print("starting _run_generation")
            if self.input_image is not None:
                input_image = self.input_image
                input_image = tensor2image(input_image)
            else:  # if no input image is provided, we take the first noise init_image that was automatically generated.
                input_image = np.array(self.diffusion_engine.image_init)

            # Process DiffusionEngine
            self.diffusion_engine.set_embeddings(self.embeds)
            if input_image is not None:
                self.diffusion_engine.set_input_image(input_image)
            if self.latents is not None:
                self.diffusion_engine.latents(self.latents)
            if self.decoder_embeds is not None:
                self.diffusion_engine.set_decoder_embeddings(self.decoder_embeds)
            if self.num_inference_steps is not None:
                self.diffusion_engine.set_num_inference_steps(int(self.num_inference_steps))
            img = np.asarray(self.diffusion_engine.generate())
            self.last_diffusion_img = img
            self.do_run = False
            # print('done _run_generation')


    def _init_diffusion_thread(self, diffusion_engine):
        self.diffusion_engine = diffusion_engine
        generation_thread = threading.Thread(target=self._run_generation)
        generation_thread.start()
        print("generation_thread started")

    def generate(
        self, 
        diffusion_engine,          
        embeds, 
        input_image=None, 
        latents=None,
        decoder_embeds=None,
        num_inference_steps=None,
        do_run=True,  
    ):
        # print("generate called")
        if self.diffusion_engine is None:
            self._init_diffusion_thread(diffusion_engine)
        self.do_run = do_run
        self.embeds = embeds
        self.input_image = input_image
        self.latents = latents
        self.decoder_embeds = decoder_embeds
        if num_inference_steps != self.num_inference_steps:
            self.num_inference_steps = num_inference_steps
            self.do_run = True
        return [self.last_diffusion_img]

# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")

