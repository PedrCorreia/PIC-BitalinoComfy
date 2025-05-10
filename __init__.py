NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

try:
    from .comfy.tools import PlotNode
    NODE_CLASS_MAPPINGS["SignalPlotter"] = PlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: {type(e).__name__} - {e}")

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

try:
    from.comfy.test_comfy.plot_node import PygamePlotNode
    NODE_CLASS_MAPPINGS["PygamePlotNode"] = PygamePlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} PygamePlotNode: ImportError - {e}")

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

try: 
    from .comfy.test_comfy.synthetic_data import SyntheticDataNode
    NODE_CLASS_MAPPINGS["SyntheticDataNode"] = SyntheticDataNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SyntheticDataNode: ImportError - {e}") 
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SyntheticDataNode: {type(e).__name__} - {e}") 

