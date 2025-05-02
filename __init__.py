NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "PIC nodes: failed to import"

try:
    from .comfy.tools import PlotNode  # Ensure the file name matches the actual file
    NODE_CLASS_MAPPINGS["SignalPlotter"] = PlotNode
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalPlotter: {type(e).__name__} - {e}")

try:
    from .comfy.signalprocessing import (
        MovingAverageFilter,
        SignalFilter,
        SignalPLL,
        SignalThresholdFilter,  # Import the missing threshold filter
    )
    NODE_CLASS_MAPPINGS["MovingAverageFilter"] = MovingAverageFilter
    NODE_CLASS_MAPPINGS["SignalFilter"] = SignalFilter
    NODE_CLASS_MAPPINGS["SignalPLL"] = SignalPLL
    NODE_CLASS_MAPPINGS["SignalThresholdFilter"] = SignalThresholdFilter  # Add the threshold filter node
except ImportError as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: ImportError - {e}")
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} SignalProcessing Nodes: {type(e).__name__} - {e}")
