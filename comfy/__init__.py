from .mock_signal_node import MockSignalGenerator, NODE_CLASS_MAPPINGS as MOCK_SIGNAL_NODE_CLASS_MAPPINGS
from .signal_input_node import  SignalInputNode, NODE_CLASS_MAPPINGS as SIGNAL_INPUT_NODE_CLASS_MAPPINGS
from .plot_unit_node import PlotUnitNode, NODE_CLASS_MAPPINGS as PLOT_UNIT_NODE_CLASS_MAPPINGS

print("[DEBUG-INIT] Loading PIC-2025 custom nodes package")

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(MOCK_SIGNAL_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SIGNAL_INPUT_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PLOT_UNIT_NODE_CLASS_MAPPINGS)

print(f"[DEBUG-INIT] Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

# Combine all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MockSignalGenerator": "Mock Signal Generator",
    "SignalInputNode": "Signal Input",
    "PlotUnitNode": "Plot Unit"
}

print("[DEBUG-INIT] PIC-2025 nodes loaded successfully")

# Uncomment to reset the registry on import (useful for debugging)
# from .mock_signal_node import SignalRegistry
# SignalRegistry.reset()
