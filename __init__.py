NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

# Use the PygamePlotNode from src as the main plotting method
try:
    from .src.plot import PygamePlotNode
    NODE_CLASS_MAPPINGS["PygamePlotNode"] = PygamePlotNode
    NODE_CLASS_MAPPINGS["SignalPlotter"] = PygamePlotNode  # Make this the default plotter
    print("PygamePlotNode from src successfully loaded as the primary plotting method")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} PygamePlotNode from src: ImportError - {e}")
    # Fallback to legacy PlotNode
    try:
        from .comfy.tools import PlotNode
        NODE_CLASS_MAPPINGS["SignalPlotter"] = PlotNode
        print("Falling back to legacy PlotNode for plotting")
    except ImportError as e2:
        print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: ImportError - {e2}")
    except Exception as e2:
        print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: {type(e2).__name__} - {e2}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} PygamePlotNode from src: {type(e).__name__} - {e}")

try:
    from .comfy.signalprocessing import (
        MovingAverageFilter,
        SignalFilter,
    )
    NODE_CLASS_MAPPINGS["MovingAverageFilter"] = MovingAverageFilter
    NODE_CLASS_MAPPINGS["SignalFilter"] = SignalFilter  # Add the threshold filter node
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: {type(e).__name__} - {e}")

try:
    from .comfy.ecg import ECGNode
    NODE_CLASS_MAPPINGS["ECGNode"] = ECGNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECGNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} ECGNode: {type(e).__name__} - {e}")

try:
    from .comfy.rr import RRNode
    NODE_CLASS_MAPPINGS["RRNode"] = RRNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} RRNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} RRNode: {type(e).__name__} - {e}")

try:
    from .comfy.eda import EDANode
    NODE_CLASS_MAPPINGS["EDANode"] = EDANode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDANode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} EDANode: {type(e).__name__} - {e}")

try:
    from .comfy.bitalino_receiver_node import LRBitalinoReceiver
    NODE_CLASS_MAPPINGS["LR BitalinoReceiver_Alt"] = LRBitalinoReceiver
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} Bitalino Receiver: {e}")

# Only try to load the test version of PygamePlotNode if we're using it as a fallback
try:
    from .comfy.test_comfy.plot_node import PygamePlotNode as TestPygamePlotNode
    NODE_CLASS_MAPPINGS["LegacyPygamePlotNode"] = TestPygamePlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} Legacy PygamePlotNode: ImportError - {e}")

try:   
    from .comfy.test_comfy.plot_node import PyQtGraphPlotNode
    NODE_CLASS_MAPPINGS["PyQtGraphPlotNode"] = PyQtGraphPlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} PyQtGraphPlotNode: ImportError - {e}")

try:
    from .comfy.test_comfy.plot_node import OpenCVPlotNode
    NODE_CLASS_MAPPINGS["OpenCVPlotNode"] = OpenCVPlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} OpenCVPlotNode: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} OpenCVPlotNode: {type(e).__name__} - {e}")  

# Import our synthetic data generator nodes (all node definitions are now in the comfy directory)
try:
    from .comfy.synthetic_generator import SynthNode
    NODE_CLASS_MAPPINGS["SynthNode"] = SynthNode
    NODE_CLASS_MAPPINGS["SyntheticDataNode"] = SynthNode  # Also register as SyntheticDataNode for compatibility
    print("Synthetic Data Generator loaded successfully")
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: ImportError - {e}")
    # Fallback to test version if new one fails
    try:
        from .comfy.test_comfy.Synth import SynthNode
        NODE_CLASS_MAPPINGS["SynthNode"] = SynthNode
        print("Falling back to test version of SynthNode")
    except ImportError as e2:
        print(f"{IMPORT_ERROR_MESSAGE} Fallback SynthNode: ImportError - {e2}")
    except Exception as e2:
        print(f"{IMPORT_ERROR_MESSAGE} Fallback SynthNode: {type(e2).__name__} - {e2}")
    
    # Additional fallback for SyntheticDataNode if needed
    try: 
        from .comfy.test_comfy.synthetic_data import SyntheticDataNode
        NODE_CLASS_MAPPINGS["SyntheticDataNode"] = SyntheticDataNode
        print("Falling back to test_comfy SyntheticDataNode") 
    except ImportError as e3:
        print(f"{IMPORT_ERROR_MESSAGE} SyntheticDataNode: ImportError - {e3}") 
    except Exception as e3:
        print(f"{IMPORT_ERROR_MESSAGE} SyntheticDataNode: {type(e3).__name__} - {e3}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SynthNode: {type(e).__name__} - {e}")

