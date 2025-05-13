# DEPRECATED: This file is deprecated and will be removed in future versions.
# Please use unified_signal_connector.py instead.
# This file is only kept for backward compatibility.

import torch
import numpy as np
import logging
from ...src.plot.plot_registry import PlotRegistry
from ...src.plot.plot_registry_integration import PlotRegistryIntegration

# Configure logger
logger = logging.getLogger('SignalConnectorNode')

# Import from the unified file to maintain compatibility
from .unified_signal_connector import SignalConnectorNode, SignalRegistryConnector

# The following class is just a proxy and all real implementation is in unified_signal_connector.py
class SignalConnectorNode(SignalConnectorNode):
    """
    Node that connects signals to the registry for visualization
    
    This node follows the proper architecture pattern:
    1. Takes signal data as input
    2. Registers it with PlotRegistry
    3. Connects the node to the signal via PlotRegistryIntegration
    """
      # All implementation is in the unified file
    pass

# Node registration - import from unified file
from .unified_signal_connector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
