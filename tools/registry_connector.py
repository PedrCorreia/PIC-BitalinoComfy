"""
Utility script to connect signals between the signal generator and visualization system.
This script can be imported to reset the registry and establish proper connections.
"""

import sys
import os
import logging
import time
from importlib import reload

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RegistryUtils")

class RegistryConnector:
    """Utility class to connect signals between registries"""
    
    @staticmethod
    def get_signal_registry():
        """Get the signal registry instance"""
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.registry.signal_registry import SignalRegistry
            return SignalRegistry.get_instance()
        except ImportError as e:
            logger.error(f"Failed to import SignalRegistry: {e}")
            return None
    
    @staticmethod
    def get_plot_registry():
        """Get the plot registry instance"""
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.registry.plot_registry import PlotRegistry
            return PlotRegistry.get_instance()
        except ImportError as e:
            logger.error(f"Failed to import PlotRegistry: {e}")
            return None
            
    @staticmethod
    def reset_registries():
        """Reset both registries"""
        signal_registry = RegistryConnector.get_signal_registry()
        plot_registry = RegistryConnector.get_plot_registry()
        
        if signal_registry:
            # Check if reset method exists
            if hasattr(signal_registry, 'reset'):
                signal_registry.reset()
                logger.info("Signal registry reset")
            else:
                logger.warning("SignalRegistry has no reset method")
        
        if plot_registry:
            # Check if reset method exists
            if hasattr(plot_registry, 'reset'):
                plot_registry.reset()
                logger.info("Plot registry reset")
            else:
                logger.warning("PlotRegistry has no reset method")
    
    @staticmethod
    def connect_signals(signal_ids=None):
        """
        Connect signals from SignalRegistry to PlotRegistry
        
        Args:
            signal_ids: List of signal IDs to connect, or None to connect all
        """
        signal_registry = RegistryConnector.get_signal_registry()
        plot_registry = RegistryConnector.get_plot_registry()
        
        if not signal_registry or not plot_registry:
            logger.error("Cannot connect signals - registries not available")
            return False
        
        # Get all signals if none specified
        if signal_ids is None:
            try:
                signal_ids = signal_registry.get_all_signals()
            except Exception as e:
                logger.error(f"Failed to get signals: {e}")
                return False
        
        success = True
        for signal_id in signal_ids:
            try:
                # Get signal data
                signal_data = signal_registry.get_signal(signal_id)
                if signal_data is None:
                    logger.warning(f"Signal {signal_id} not found in registry")
                    continue
                
                # Get metadata
                metadata = signal_registry.get_signal_metadata(signal_id)
                
                # Register with plot registry
                plot_registry.register_signal(signal_id, signal_data, metadata)
                logger.info(f"Connected signal {signal_id} to plot registry")
            except Exception as e:
                logger.error(f"Failed to connect signal {signal_id}: {e}")
                success = False
                
        return success
        
    @staticmethod
    def list_signals():
        """List all signals in both registries"""
        signal_registry = RegistryConnector.get_signal_registry()
        plot_registry = RegistryConnector.get_plot_registry()
        
        logger.info("=== Signal Registry Contents ===")
        if signal_registry:
            try:
                signals = signal_registry.get_all_signals()
                logger.info(f"Found {len(signals)} signals: {signals}")
            except Exception as e:
                logger.error(f"Failed to list signals: {e}")
        else:
            logger.warning("SignalRegistry not available")
            
        logger.info("=== Plot Registry Contents ===")
        if plot_registry:
            try:
                signals = plot_registry.get_all_signals()
                logger.info(f"Found {len(signals)} signals: {signals}")
                
                # Get visualized signals
                if hasattr(plot_registry, 'get_visualized_signals'):
                    visualized = plot_registry.get_visualized_signals()
                    logger.info(f"Visualized signals: {visualized}")
            except Exception as e:
                logger.error(f"Failed to list signals: {e}")
        else:
            logger.warning("PlotRegistry not available")
            
# Command-line utility functions
def reset_registries():
    """Reset both registries"""
    RegistryConnector.reset_registries()
    
def connect_all_signals():
    """Connect all signals from SignalRegistry to PlotRegistry"""
    RegistryConnector.connect_signals()
    
def list_all_signals():
    """List all signals in both registries"""
    RegistryConnector.list_signals()
    
# Add these functions to the module namespace to make them importable
__all__ = ['RegistryConnector', 'reset_registries', 'connect_all_signals', 'list_all_signals']

if __name__ == "__main__":
    # If run directly, perform a full diagnostic
    logger.info("Running registry connector utility")
    reset_registries()
    time.sleep(0.5)  # Allow time for reset to complete
    list_all_signals()
    connect_all_signals()
    time.sleep(0.5)  # Allow time for connections to complete
    list_all_signals()
    logger.info("Registry connector utility completed")
