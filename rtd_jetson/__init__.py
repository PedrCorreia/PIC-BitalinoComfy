NODE_CLASS_MAPPINGS = {}
from .comfy.diffusion_engine import LRDiffusionEngineAcid, LRDiffusionEngineLoader, LRDiffusionEngineThreaded
NODE_CLASS_MAPPINGS["PIC DiffusionEngineAcid"] = LRDiffusionEngineAcid
NODE_CLASS_MAPPINGS["PIC DiffusionEngineLoader"] = LRDiffusionEngineLoader
NODE_CLASS_MAPPINGS["PIC DiffusionEngineThreaded"] = LRDiffusionEngineThreaded

from .comfy.input_image import LRInputImageProcessor, LRFreezeImage, LRCropImage, LRImageGate, LRImageGateSelect, LRCropCoordinates
NODE_CLASS_MAPPINGS["PIC InputImageProcessor"] = LRInputImageProcessor
NODE_CLASS_MAPPINGS["PIC FreezeImage"] = LRFreezeImage
NODE_CLASS_MAPPINGS["PIC CropImage"] = LRCropImage
NODE_CLASS_MAPPINGS["PIC CropCoordinates"] = LRCropCoordinates
NODE_CLASS_MAPPINGS["PIC ImageGate"] = LRImageGate
NODE_CLASS_MAPPINGS["PIC ImageGateSelect"] = LRImageGateSelect

from .comfy.embeddings_mixer import LRPrompt2Embedding, LRBlend2Embeds, LRBlend4Embeds
NODE_CLASS_MAPPINGS["PIC Prompt2Embedding"] = LRPrompt2Embedding
NODE_CLASS_MAPPINGS["PIC PICBlend2Embeds"] = LRBlend2Embeds
NODE_CLASS_MAPPINGS["PIC PICBlend4Embeds"] = LRBlend4Embeds

from .comfy.segmentation_detection import LRHumanSeg, LRFaceCropper
NODE_CLASS_MAPPINGS["PIC HumanSeg"] = LRHumanSeg
NODE_CLASS_MAPPINGS["PIC FaceCropper"] = LRFaceCropper

from .comfy.controlnet_node import ControlNetNode
NODE_CLASS_MAPPINGS["ControlNetNode"] = ControlNetNode
NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetNode": "ControlNet Node",
}
NODE_CATEGORY_MAPPINGS = {
    "ControlNetNode": "LunarRing/visual",
}