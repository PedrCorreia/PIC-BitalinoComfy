NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CATEGORY_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

# Define new categories with emojis
# Main categories: Pedro_PIC/🧰 Tools, Pedro_PIC/🔬 Processing, Pedro_PIC/🔬 Bio-Processing, Pedro_PIC/📡 Bitalino

try:
    from .src.plot import PygamePlot
    PYGAME_PLOT_AVAILABLE = True
    print("PygamePlot base class loaded successfully")
except ImportError as e:
    PYGAME_PLOT_AVAILABLE = False
    print(f"{IMPORT_ERROR_MESSAGE} PygamePlot base class: ImportError - {e}")
except Exception as e:
    PYGAME_PLOT_AVAILABLE = False
    print(f"{IMPORT_ERROR_MESSAGE} PygamePlot base class: {type(e).__name__} - {e}")

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
    from .comfy.ecg import ECGNode
    from .comfy.rr import RRNode, RRLastValueNode
    from .comfy.eda import EDANode
    
    # Register nodes with consistent categories
    bio_nodes = {
        "ECGNode": ("❤️ ECG Processing", "Pedro_PIC/🔬 Bio-Processing"),
        "RRNode": ("🫁 RR Processing", "Pedro_PIC/🔬 Bio-Processing"),
        "RRLastValueNode": ("🫁 RR Last Value", "Pedro_PIC/🔬 Bio-Processing"),
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

# Bitalino Receiver
try:
    from .comfy.bitalino_receiver_node import LRBitalinoReceiver
    NODE_CLASS_MAPPINGS["LR BitalinoReceiver_Alt"] = LRBitalinoReceiver
    NODE_DISPLAY_NAME_MAPPINGS["LR BitalinoReceiver_Alt"] = "📡 Bitalino Receiver"
    NODE_CATEGORY_MAPPINGS["LR BitalinoReceiver_Alt"] = "Pedro_PIC/📡 Bitalino"
    
    print("Bitalino Receiver loaded successfully")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bitalino Receiver: {e}")

# Import synthetic data generator node
try:
    from .comfy.synthetic_generator import SynthNode
    NODE_CLASS_MAPPINGS["SynthNode"] = SynthNode
    NODE_DISPLAY_NAME_MAPPINGS["SynthNode"] = "📊 Synthetic Data Generator"
    NODE_CATEGORY_MAPPINGS["SynthNode"] = "Pedro_PIC/🧰 Tools"
    
    print("Synthetic Data Generator loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: {type(e).__name__} - {e}")

# Tools and utility nodes
try:
    from .comfy.tools import CombineNode, SeparateNode, GetLastValueNode, IsPeakNode, PhysioNormalizeNode
    
    # Register tools with consistent categories
    tool_nodes = {
        "CombineNode": ("🔗 Combine Signal Components", "Pedro_PIC/🧰 Tools"),
        "SeparateNode": ("✂️ Separate Signal Components", "Pedro_PIC/🧰 Tools"),
        "GetLastValueNode": ("📊 Get Last Signal Value", "Pedro_PIC/🧰 Tools"),
        "IsPeakNode": ("⚡ Is Peak", "Pedro_PIC/🧰 Tools"),
        "PhysioNormalizeNode": ("📏 Physio Normalize", "Pedro_PIC/🧰 Tools"),
    }
    
    for node_name, (display_name, category) in tool_nodes.items():
        node_class = locals()[node_name]
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name
        NODE_CATEGORY_MAPPINGS[node_name] = category
    
    print("Tools nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Tools nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Tools nodes: {type(e).__name__} - {e}")