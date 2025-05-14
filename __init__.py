import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CATEGORY_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

print("[DEBUG-INIT] Loading PIC-2025 custom nodes package")

# Define new categories with emojis
# Main categories: Pedro_PIC/🧰 Tools, Pedro_PIC/🔬 Processing, Pedro_PIC/🔬 Bio-Processing, Pedro_PIC/📡 Bitalino, Pedro_PIC/🌊 Signal Registry

try:
    from .src.plot.plot import PygamePlot
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

# Bitalino Receiver
try:
    from .comfy.bitalino_receiver_node import LRBitalinoReceiver
    NODE_CLASS_MAPPINGS["LR BitalinoReceiver_Alt"] = LRBitalinoReceiver
    NODE_DISPLAY_NAME_MAPPINGS["LR BitalinoReceiver_Alt"] = "📡 Bitalino Receiver"
    NODE_CATEGORY_MAPPINGS["LR BitalinoReceiver_Alt"] = "Pedro_PIC/📡 Bitalino"
    
    print("Bitalino Receiver loaded successfully")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bitalino Receiver: {e}")

# Tools and utility nodes
try:
    from .comfy.tools import CombineNode, SeparateNode, GetLastValueNode, IsPeakNode
    
    # Register tools with consistent categories
    tool_nodes = {
        "CombineNode": ("🔗 Combine Signal Components", "Pedro_PIC/🧰 Tools"),
        "SeparateNode": ("✂️ Separate Signal Components", "Pedro_PIC/🧰 Tools"),
        "GetLastValueNode": ("📊 Get Last Signal Value", "Pedro_PIC/🧰 Tools"),
        "IsPeakNode": ("⚡ Is Peak", "Pedro_PIC/🧰 Tools"),
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

# Add Arousal nodes
try:
    from .comfy.arousal import PhysioNormalizeNode
    
    # Register arousal nodes with consistent categories
    arousal_nodes = {
        "PhysioNormalizeNode": ("📏 Physio Normalize", "Pedro_PIC/🧭 Arousal"),
    }
    
    for node_name, (display_name, category) in arousal_nodes.items():
        node_class = locals()[node_name]
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name
        NODE_CATEGORY_MAPPINGS[node_name] = category
    
    print("Arousal nodes loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Arousal nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Arousal nodes: {type(e).__name__} - {e}")

# Plot Unit Node - Latest version using standalone visualization hub
try:
    # Import the standalone visualization hub for better reliability
    from .comfy.Registry.standalone_visualization_hub import PlotUnitNode
    
    # Register Plot Unit node with consistent category
    NODE_CLASS_MAPPINGS["StandaloneVisualizationHub"] = PlotUnitNode
    NODE_DISPLAY_NAME_MAPPINGS["StandaloneVisualizationHub"] = "📊 Signal Visualization Hub (Live)"
    NODE_CATEGORY_MAPPINGS["StandaloneVisualizationHub"] = "Pedro_PIC/🌊 Signal Registry"
    
    print("Standalone Visualization Hub (Live) loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Standalone Visualization Hub: ImportError - {e}")
    
    # Fallback to the regular plot unit node if standalone fails
    try:
        from .comfy.Registry.plot_unit_node import PlotUnitNode
        
        # Register Plot Unit node with consistent category
        NODE_CLASS_MAPPINGS["PlotUnitNode"] = PlotUnitNode
        NODE_DISPLAY_NAME_MAPPINGS["PlotUnitNode"] = "📊 Signal Visualization Hub"
        NODE_CATEGORY_MAPPINGS["PlotUnitNode"] = "Pedro_PIC/🌊 Signal Registry"
        
        print("Plot Unit Visualization Hub loaded successfully (fallback)")
    except Exception as fallback_error:
        print(f"{IMPORT_ERROR_MESSAGE} Plot Unit Node (fallback): {type(fallback_error).__name__} - {fallback_error}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Standalone Visualization Hub: {type(e).__name__} - {e}")

# Mock Signal Node was moved to Signal Registry Nodes section

# Signal Input Node (Modern signal connection approach)
try:
    from .comfy.Registry.signal_input_node import SignalInputNode
    NODE_CLASS_MAPPINGS["SignalInputNode"] = SignalInputNode
    NODE_DISPLAY_NAME_MAPPINGS["SignalInputNode"] = "🔌 Signal Input"
    NODE_CATEGORY_MAPPINGS["SignalInputNode"] = "Pedro_PIC/🌊 Signal Registry"
    print("Signal Input Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Signal Input Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Signal Input Node: {type(e).__name__} - {e}")
# Signal Debug Node
try:
    from .comfy.Registry.signal_debug_node import SignalDebugNode
    NODE_CLASS_MAPPINGS["SignalDebugNode"] = SignalDebugNode
    NODE_DISPLAY_NAME_MAPPINGS["SignalDebugNode"] = "🔍 Signal Debug"
    NODE_CATEGORY_MAPPINGS["SignalDebugNode"] = "Pedro_PIC/🔬 Diagnostics"
    print("Signal Debug Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Signal Debug Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Signal Debug Node: {type(e).__name__} - {e}")

# Logger Node
try:
    from .comfy.Registry.logger_node import LoggerNode
    NODE_CLASS_MAPPINGS["LoggerNode"] = LoggerNode
    NODE_DISPLAY_NAME_MAPPINGS["LoggerNode"] = "📝 Signal Logger"
    NODE_CATEGORY_MAPPINGS["LoggerNode"] = "Pedro_PIC/🔬 Diagnostics"
    print("Logger Node loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Logger Node: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Logger Node: {type(e).__name__} - {e}")

# Unified Signal Generator (New consolidated generator)
try:
    from .comfy.Registry.unified_signal_generator import UnifiedSignalGenerator
    NODE_CLASS_MAPPINGS["UnifiedSignalGenerator"] = UnifiedSignalGenerator
    NODE_DISPLAY_NAME_MAPPINGS["UnifiedSignalGenerator"] = "🌊 Signal Generator"
    NODE_CATEGORY_MAPPINGS["UnifiedSignalGenerator"] = "Pedro_PIC/🌊 Signal Registry"
    print("Unified Signal Generator loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Unified Signal Generator: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Unified Signal Generator: {type(e).__name__} - {e}")

# Add a final message confirming initialization
print("[DEBUG-INIT] PIC-2025 nodes loaded successfully")

# Uncomment to reset the registry on import (useful for debugging)
# from .comfy.mock_signal_node import SignalRegistry
# SignalRegistry.reset()
