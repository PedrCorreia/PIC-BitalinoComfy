NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CATEGORY_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

# Define new categories with emojis
# Main categories: Pedro_PIC/🧰 Tools, Pedro_PIC/🔬 Processing, Pedro_PIC/📡 Bitalino


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

try:
    from .comfy.signalprocessing import (
        MovingAverageFilter,
        SignalFilter,
    )
    NODE_CLASS_MAPPINGS["MovingAverageFilter"] = MovingAverageFilter
    NODE_CLASS_MAPPINGS["SignalFilter"] = SignalFilter  # Add the threshold filter node
    
    # Add display names and categories
    NODE_DISPLAY_NAME_MAPPINGS["MovingAverageFilter"] = "📉 Moving Average Filter"
    NODE_DISPLAY_NAME_MAPPINGS["SignalFilter"] = "🔍 Signal Filter"
    NODE_CATEGORY_MAPPINGS["MovingAverageFilter"] = "Pedro_PIC/🔬 Processing"
    NODE_CATEGORY_MAPPINGS["SignalFilter"] = "Pedro_PIC/🔬 Processing"
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: {type(e).__name__} - {e}")

try:
    from .comfy.ecg import ECGNode
    NODE_CLASS_MAPPINGS["ECGNode"] = ECGNode
    NODE_DISPLAY_NAME_MAPPINGS["ECGNode"] = "❤️ ECG Processing"
    NODE_CATEGORY_MAPPINGS["ECGNode"] = "Pedro_PIC/🔬 Processing"
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECGNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECGNode: {type(e).__name__} - {e}")

try:
    from .comfy.rr import RRNode
    NODE_CLASS_MAPPINGS["RRNode"] = RRNode
    NODE_DISPLAY_NAME_MAPPINGS["RRNode"] = "🫁 RR Processing"
    NODE_CATEGORY_MAPPINGS["RRNode"] = "Pedro_PIC/🔬 Processing"
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} RRNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} RRNode: {type(e).__name__} - {e}")

try:
    from .comfy.eda import EDANode
    NODE_CLASS_MAPPINGS["EDANode"] = EDANode
    NODE_DISPLAY_NAME_MAPPINGS["EDANode"] = "💧 EDA Processing"
    NODE_CATEGORY_MAPPINGS["EDANode"] = "Pedro_PIC/🔬 Processing"
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDANode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDANode: {type(e).__name__} - {e}")

try:
    from .comfy.bitalino_receiver_node import LRBitalinoReceiver
    NODE_CLASS_MAPPINGS["LR BitalinoReceiver_Alt"] = LRBitalinoReceiver
    NODE_DISPLAY_NAME_MAPPINGS["LR BitalinoReceiver_Alt"] = "📡 Bitalino Receiver"
    NODE_CATEGORY_MAPPINGS["LR BitalinoReceiver_Alt"] = "Pedro_PIC/📡 Bitalino"
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bitalino Receiver: {e}")

# We no longer load any test/legacy versions of the plotting nodes
# as we have a single, working implementation now

# We already loaded PygamePlot at the top of the file, so no need to import it again here

# Import our synthetic data generator node (single implementation) 
try:
    from .comfy.synthetic_generator import SynthNode
    NODE_CLASS_MAPPINGS["SynthNode"] = SynthNode
    
    # Add display name and category
    NODE_DISPLAY_NAME_MAPPINGS["SynthNode"] = "📊 Synthetic Data Generator"
    NODE_CATEGORY_MAPPINGS["SynthNode"] = "Pedro_PIC/🧰 Tools"
    
    print("Synthetic Data Generator loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: {type(e).__name__} - {e}")

