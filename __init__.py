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
try:
    # Import all bio-processing nodes in a batch
    from .comfy.phy.ecg import ECGNode
    from .comfy.phy.rr import RRNode  # Removed RRLastValueNode
    from .comfy.phy.eda import EDANode
    
    # Register nodes with consistent categories
    bio_nodes = {
        "ECGNode": ("❤️ ECG Processing", "Pedro_PIC/🔬 Bio-Processing"),
        "RRNode": ("🫁 RR Processing", "Pedro_PIC/🔬 Bio-Processing"),
        "EDANode": ("💧 EDA Processing", "Pedro_PIC/🔬 Bio-Processing"),
    }
    
    for node_name, (display_name, category) in bio_nodes.items():
        node_class = locals()[node_name]
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name
        NODE_CATEGORY_MAPPINGS[node_name] = category
    
    print("Bio-Processing Nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bio-Processing Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bio-Processing Nodes: {type(e).__name__} - {e}")

try:
    from .comfy.tools import PrintToolNode, DepthModelLoaderNode, DepthMapNode
    NODE_CLASS_MAPPINGS["PrintToolNode"] = PrintToolNode
    NODE_DISPLAY_NAME_MAPPINGS["PrintToolNode"] = "🛠️ Print Tool Node"
    NODE_CATEGORY_MAPPINGS["PrintToolNode"] = "Pedro_PIC/🛠️ Tools"
    NODE_CLASS_MAPPINGS["DepthModelLoaderNode"] = DepthModelLoaderNode
    NODE_DISPLAY_NAME_MAPPINGS["DepthModelLoaderNode"] = "🧰 Depth Model Loader Node"
    NODE_CATEGORY_MAPPINGS["DepthModelLoaderNode"] = "Pedro_PIC/🧰 Tools"
    NODE_CLASS_MAPPINGS["DepthMapNode"] = DepthMapNode
    NODE_DISPLAY_NAME_MAPPINGS["DepthMapNode"] = "🧰 Depth Map Node"
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

