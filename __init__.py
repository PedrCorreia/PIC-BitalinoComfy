import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CATEGORY_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

print("[DEBUG-INIT] Loading PIC-2025 custom nodes package")

# Define new categories with emojis
# Main categories: Pedro_PIC/🧰 Tools, Pedro_PIC/🔬 Processing, Pedro_PIC/🔬 Bio-Processing, Pedro_PIC/📡 Bitalino, Pedro_PIC/🌊 Signal Registry

# Signal Processing Nodes
try:
    from .comfy.signalprocessing import (
        MovingAverageFilter,
        SignalFilter,
        LoadSignalNode,
    )
    
    # Register nodes with consistent categories
    nodes = {
        "MovingAverageFilter": ("📉 Moving Average Filter", "Pedro_PIC/🔬 Processing"),
        "SignalFilter": ("🔍 Signal Filter", "Pedro_PIC/🔬 Processing"),
        "LoadSignalNode": ("📂 Load Signal", "Pedro_PIC/🔬 Processing"),
    }
    
    for node_name, (display_name, category) in nodes.items():
        node_class = locals()[node_name]
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name
        NODE_CATEGORY_MAPPINGS[node_name] = category
    
    print("Signal Processing Nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: {type(e).__name__} - {e}")

# Bio Processing Nodes
# ECG Node
try:
    from .comfy.phy.ecg import ECGNode
    NODE_CLASS_MAPPINGS["ECGNode"] = ECGNode
    NODE_DISPLAY_NAME_MAPPINGS["ECGNode"] = "❤️ ECG Processing"
    NODE_CATEGORY_MAPPINGS["ECGNode"] = "Pedro_PIC/🔬 Bio-Processing"
    print("ECG Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECG Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECG Node: {type(e).__name__} - {e}")

# RR Node
try:
    from .comfy.phy.rr import RRNode
    NODE_CLASS_MAPPINGS["RRNode"] = RRNode
    NODE_DISPLAY_NAME_MAPPINGS["RRNode"] = "🫁 RR Processing"
    NODE_CATEGORY_MAPPINGS["RRNode"] = "Pedro_PIC/🔬 Bio-Processing"
    print("RR Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} RR Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} RR Node: {type(e).__name__} - {e}")

# EDA Node
try:
    from .comfy.phy.eda import EDANode
    NODE_CLASS_MAPPINGS["EDANode"] = EDANode
    NODE_DISPLAY_NAME_MAPPINGS["EDANode"] = "💧 EDA Processing"
    NODE_CATEGORY_MAPPINGS["EDANode"] = "Pedro_PIC/🔬 Bio-Processing"
    print("EDA Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDA Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDA Node: {type(e).__name__} - {e}")

try:
    from .comfy.tools import PrintToolNode, PrintMultiToolNode, EnhancedPrintToolNode, DepthModelLoaderNode, DepthMapNode
    NODE_CLASS_MAPPINGS["PrintToolNode"] = PrintToolNode
    NODE_CLASS_MAPPINGS["PrintMultiToolNode"] = PrintMultiToolNode
    NODE_CLASS_MAPPINGS["EnhancedPrintToolNode"] = EnhancedPrintToolNode
    NODE_CLASS_MAPPINGS["DepthModelLoaderNode"] = DepthModelLoaderNode
    NODE_CLASS_MAPPINGS["DepthMapNode"] = DepthMapNode

    NODE_DISPLAY_NAME_MAPPINGS["PrintToolNode"] = "🛠️ Print Tool Node"
    NODE_DISPLAY_NAME_MAPPINGS["PrintMultiToolNode"] = "🛠️ Multi Print Tool"
    NODE_DISPLAY_NAME_MAPPINGS["EnhancedPrintToolNode"] = "🛠️ Enhanced Print Tool"
    NODE_DISPLAY_NAME_MAPPINGS["DepthModelLoaderNode"] = "🧰 Depth Model Loader Node"
    NODE_DISPLAY_NAME_MAPPINGS["DepthMapNode"] = "🧰 Depth Map Node"

    NODE_CATEGORY_MAPPINGS["PrintToolNode"] = "Pedro_PIC/🛠️ Tools"
    NODE_CATEGORY_MAPPINGS["PrintMultiToolNode"] = "Pedro_PIC/🛠️ Tools"
    NODE_CATEGORY_MAPPINGS["EnhancedPrintToolNode"] = "Pedro_PIC/🛠️ Tools"
    NODE_CATEGORY_MAPPINGS["DepthModelLoaderNode"] = "Pedro_PIC/🧰 Tools"
    NODE_CATEGORY_MAPPINGS["DepthMapNode"] = "Pedro_PIC/🧰 Tools"

    print("Print Tool Node and Depth Nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Print Tool/Depth Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Print Tool/Depth Nodes: {type(e).__name__} - {e}")

try:
    from .comfy.Registry.comfy_signal_generator_node import ComfySignalGeneratorNode
    from .comfy.Registry.comfy_signal_connector_node import ComfySignalConnectorNode
    from .comfy.plot.comfy_plot_registry_node import ComfyPlotRegistryNode
    NODE_CLASS_MAPPINGS["ComfySignalGeneratorNode"] = ComfySignalGeneratorNode
    NODE_DISPLAY_NAME_MAPPINGS["ComfySignalGeneratorNode"] = "🌊 Signal Generator (Registry)"
    NODE_CATEGORY_MAPPINGS["ComfySignalGeneratorNode"] = "Pedro_PIC/🌊 Signal Registry"
    NODE_CLASS_MAPPINGS["ComfySignalConnectorNode"] = ComfySignalConnectorNode
    NODE_DISPLAY_NAME_MAPPINGS["ComfySignalConnectorNode"] = "🔗 Signal Connector (Registry)"
    NODE_CATEGORY_MAPPINGS["ComfySignalConnectorNode"] = "Pedro_PIC/🌊 Signal Registry"
    NODE_CLASS_MAPPINGS["ComfyPlotRegistryNode"] = ComfyPlotRegistryNode
    NODE_DISPLAY_NAME_MAPPINGS["ComfyPlotRegistryNode"] = "📊 Plot Registry Node"
    NODE_CATEGORY_MAPPINGS["ComfyPlotRegistryNode"] = "Pedro_PIC/🌊 Signal Registry"
    print("Comfy Registry-based Signal Nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Comfy Registry-based Signal Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Comfy Registry-based Signal Nodes: {type(e).__name__} - {e}")
try:
    from .comfy.Registry.comfy_bitalino_generator_node import BITSignalGeneratorNode
    NODE_CLASS_MAPPINGS["BITSignalGeneratorNode"] = BITSignalGeneratorNode
    NODE_DISPLAY_NAME_MAPPINGS["BITSignalGeneratorNode"] = "📡 Bitalino Signal Generator "
    NODE_CATEGORY_MAPPINGS["BITSignalGeneratorNode"] = "Pedro_PIC/📡 Bitalino"
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Comfy Bitalino Signal Generator: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Comfy Bitalino Signal Generator: {type(e).__name__} - {e}")
# Add a final message confirming initialization
print("[DEBUG-INIT] PIC-2025 nodes loaded successfully")

try:
    from .comfy.geom.geometry_node import GeometryRenderNode
    NODE_CLASS_MAPPINGS["GeometryRenderNodeRT"] = GeometryRenderNode
    NODE_DISPLAY_NAME_MAPPINGS["GeometryRenderNode"] = "🧱 Geometry Render RT (3D)"
    NODE_CATEGORY_MAPPINGS["GeometryRenderNode"] = "Pedro_PIC/🧰 Tools"
    print("Geometry Render Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Geometry Render Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Geometry Render Node: {type(e).__name__} - {e}")

try:
    from .comfy.geom.geometry_node  import GeometryRenderNode
    NODE_CLASS_MAPPINGS["GeometryRenderNode"] = GeometryRenderNode
    NODE_DISPLAY_NAME_MAPPINGS["GeometryRenderNode"] = "🧱 Geometry Render (3D)"
    NODE_CATEGORY_MAPPINGS["GeometryRenderNode"] = "Pedro_PIC/🧰 Tools"
    print("Geometry Render Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Geometry Render Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Geometry Render Node: {type(e).__name__} - {e}")

try:
    from .comfy.overall_arousal import OverallArousalNode
    NODE_CLASS_MAPPINGS["OverallArousalNode"] = OverallArousalNode
    NODE_DISPLAY_NAME_MAPPINGS["OverallArousalNode"] = "🎯 Overall Arousal"
    NODE_CATEGORY_MAPPINGS["OverallArousalNode"] = "Pedro_PIC/🔬 Bio-Processing"
    print("Overall Arousal Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Overall Arousal Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Overall Arousal Node: {type(e).__name__} - {e}")

