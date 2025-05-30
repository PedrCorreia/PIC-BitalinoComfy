import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from ..src.im_process.process import Midas, ImgUtils 

class PrintToolNode:
    """
    A node that prints any input data with a user-supplied comment.
    Accepts any Python data type as input and prints it with the comment.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Info": ("ANY", {}),
                "comment": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_tool"
    CATEGORY = "Pedro_PIC/🛠️ Tools"
    OUTPUT_NODE = True 

    def print_tool(self, thing, comment):
        print(f"{comment} {thing}")
        return (thing,)

class PrintMultiToolNode:
    """
    A node that prints multiple optional inputs (HR, RR, is_peak, etc.) with a user-supplied comment.
    Accepts any combination of these inputs and prints them for debugging.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comment": ("STRING", {"default": ""}),
            },
            "optional": {
                "HR": ("FLOAT", {}),
                "RR": ("FLOAT", {}),
                "is_peak": ("BOOLEAN", {}),
                "signal_id": ("STRING", {}),
                "thing": ("ANY", {}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_multi_tool"
    CATEGORY = "Pedro_PIC/🛠️ Tools"
    OUTPUT_NODE = True

    def print_multi_tool(self, comment, HR=None, RR=None, is_peak=None, signal_id=None, thing=None):
        msg = f"{comment}"
        if HR is not None:
            msg += f" | HR: {HR}"
        if RR is not None:
            msg += f" | RR: {RR}"
        if is_peak is not None:
            msg += f" | is_peak: {is_peak}"
        if signal_id is not None:
            msg += f" | signal_id: {signal_id}"
        if thing is not None:
            msg += f" | thing: {thing}"
        print(msg)
        return ()

class DepthModelLoaderNode:
    """
    Loads the MiDaS model encapsulated in the Midas class from src.im_process.process.
    """
    MODEL_TYPES = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"] # Add other MiDaS types as needed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (cls.MODEL_TYPES, {"default": "DPT_Large"}),
            }
        }

    RETURN_TYPES = ("MIDAS_INSTANCE",)
    RETURN_NAMES = ("midas_instance",)
    FUNCTION = "load_midas_model"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def load_midas_model(self, model_type):
        print(f"[DepthModelLoaderNode] Attempting to load MiDaS model: {model_type}")
        midas_instance = None # Initialize to None
        try:
            # Assuming Midas class is imported correctly from ..src.im_process.process
            midas_instance = Midas(model_type=model_type, device=None, use_half_precision=None)
            print(f"[DepthModelLoaderNode] Midas instance created: {type(midas_instance)}")
            print(f"[DepthModelLoaderNode] Calling midas_instance.load_model()...")
            midas_instance.load_model() 
            print(f"[DepthModelLoaderNode] midas_instance.load_model() completed.")

            if midas_instance.model is None:
                print(f"[DepthModelLoaderNode] CRITICAL WARNING: midas_instance.model is None after load_model().")
            else:
                print(f"[DepthModelLoaderNode] midas_instance.model loaded successfully.")

            if midas_instance.transform is None:
                print(f"[DepthModelLoaderNode] CRITICAL WARNING: midas_instance.transform is None after load_model().")
            else:
                print(f"[DepthModelLoaderNode] midas_instance.transform loaded successfully.")
            
            print(f"[DepthModelLoaderNode] Returning midas_instance.")
            return (midas_instance,)
        except Exception as e:
            print(f"[DepthModelLoaderNode] ERROR during load_midas_model: {e}")
            import traceback
            traceback.print_exc() # Print full traceback to console
            # Re-raising the exception is important for ComfyUI to see the node failed
            raise e 

class DepthMapNode:
    """
    Receives an image and a Midas instance, and returns a depth map
    using the Midas class from src.im_process.process.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "midas_instance": ("MIDAS_INSTANCE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("depth_map", )
    FUNCTION = "run_depth_prediction"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def run_depth_prediction(self, image: torch.Tensor, midas_instance: Midas):
        if not isinstance(midas_instance, Midas):
            raise TypeError("midas_instance must be an instance of the Midas class.")

        # Convert ComfyUI IMAGE tensor (B, H, W, C) [0,1] to NumPy array (H, W, C) [0,255] uint8
        # Midas.predict expects a NumPy array (BGR or RGB).
        # ImgUtils.tensor_to_numpy handles B,H,W,C -> H,W,C and scaling to 0-255.
        # It returns RGB by default if it converts from a tensor that could be RGB.
        numpy_image = ImgUtils.tensor_to_numpy(image)

        # Midas.predict returns a 2D NumPy array (H, W), float32, normalized [0,1]
        # It handles internal resizing and matches output to original input image dimensions.
        depth_map_np = midas_instance.predict(numpy_image, optimize_size=True) # optimize_size is True by default
        print(f"[DepthMapNode] depth_map_np from Midas.predict shape: {depth_map_np.shape}, dtype: {depth_map_np.dtype}")

        # Handle potential 1x1 output case to prevent Pillow error downstream
        if depth_map_np.shape == (1, 1):
            print(f"[DepthMapNode] Detected 1x1 depth_map_np. Upscaling to 2x2.")
            pixel_value = depth_map_np[0, 0] 
            depth_map_np = np.full((2, 2), pixel_value, dtype=np.float32) 
            print(f"[DepthMapNode] Upscaled depth_map_np shape: {depth_map_np.shape}, dtype: {depth_map_np.dtype}")
        
        final_numpy_depth = np.ascontiguousarray(depth_map_np) 
        
        depth_tensor = torch.from_numpy(final_numpy_depth).unsqueeze(0).unsqueeze(-1)
        print(f"[DepthMapNode] Returning depth_tensor shape: {depth_tensor.shape}, dtype: {depth_tensor.dtype}")
        return (depth_tensor,)

class CannyMapNode:
    """
    Receives an image and returns a canny edge map using OpenCV.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("canny_map", )
    FUNCTION = "run"
    CATEGORY = "Pedro_PIC/🧰 Tools"

    def run(self, image):
        # Convert ComfyUI IMAGE (Tensor) to NumPy array (H, W, C)
        img_numpy = ImgUtils.tensor_to_numpy(image)

        # Ensure it's grayscale for Canny
        if img_numpy.ndim == 3 and img_numpy.shape[2] == 3:
            gray = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
        elif img_numpy.ndim == 3 and img_numpy.shape[2] == 1:
            gray = img_numpy.squeeze(-1) # (H, W, 1) -> (H, W)
        elif img_numpy.ndim == 2:
            gray = img_numpy # Already (H, W)
        else:
            # Fallback: try to convert to PIL and then to grayscale, then back to numpy
            pil_img = ImgUtils.tensor_to_pil(image)
            gray_pil = pil_img.convert("L")
            gray = ImgUtils.pil_to_numpy(gray_pil)

        edges = cv2.Canny(gray, 100, 200)
        
        # Convert Canny output (H, W) NumPy to ComfyUI IMAGE tensor (1, H, W, 1)
        edges_tensor = torch.from_numpy(edges).float().unsqueeze(0).unsqueeze(-1) / 255.0
        return (edges_tensor,)

NODE_CLASS_MAPPINGS = globals().get("NODE_CLASS_MAPPINGS", {})
NODE_CLASS_MAPPINGS["PrintToolNode"] = PrintToolNode
NODE_CLASS_MAPPINGS["PrintMultiToolNode"] = PrintMultiToolNode
NODE_CLASS_MAPPINGS["DepthModelLoaderNode"] = DepthModelLoaderNode
NODE_CLASS_MAPPINGS["DepthMapNode"] = DepthMapNode
NODE_CLASS_MAPPINGS["CannyMapNode"] = CannyMapNode

NODE_DISPLAY_NAME_MAPPINGS = globals().get("NODE_DISPLAY_NAME_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS["PrintToolNode"] = "🛠️ Print Tool Node"
NODE_DISPLAY_NAME_MAPPINGS["PrintMultiToolNode"] = "🛠️ Print Multi Tool Node"
NODE_DISPLAY_NAME_MAPPINGS["DepthModelLoaderNode"] = "🧰 Depth Model Loader Node"
NODE_DISPLAY_NAME_MAPPINGS["DepthMapNode"] = "🧰 Depth Map Node"
NODE_DISPLAY_NAME_MAPPINGS["CannyMapNode"] = "🧰 Canny Map Node"
