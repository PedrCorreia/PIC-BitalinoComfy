# DEPRECATED: This file is deprecated and will be removed in future versions.
# Please use unified_signal_connector.py instead.
# This file is only kept for backward compatibility.

import sys
import os
from ...src.plot.plot_registry import PlotRegistry

import torch
import numpy as np
import logging

# Configure logger
logger = logging.getLogger('SignalRegistryConnector')

# Import from the unified file to maintain compatibility
from .unified_signal_connector import SignalConnectorNode, SignalRegistryConnector

# The following class is just a proxy and all real implementation is in unified_signal_connector.py
class SignalRegistryConnector(SignalRegistryConnector):
    """
    A node that connects a signal to the visualization registry.
    This node controls whether a signal is passed to the registry or removed from it.
    When enabled, the signal ID is registered with the plot registry.
    When disabled, the signal ID is removed from the registry if it exists.
    """    # All implementation is in the unified file
    pass

# Node registration - import from unified file
from .unified_signal_connector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
