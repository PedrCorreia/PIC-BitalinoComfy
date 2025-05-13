"""
Signal Registry for PIC-2025.
Maintains a global registry of signals available for visualization.

This is a singleton class that stores signals by ID and provides
methods for registering, accessing, and managing signals.
"""

import numpy as np
import threading
import logging
import time
import torch
from collections import OrderedDict

# Configure logger
logger = logging.getLogger('SignalRegistry')

class SignalRegistry:
    """
    A dedicated registry for signal data.
    This acts as the central repository for all signals in the system.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the registry"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                logger.info("Created new SignalRegistry singleton instance")
            return cls._instance
    
    def __init__(self):
        """Initialize the registry with empty containers"""
        self.signals = OrderedDict()  # Signal data keyed by ID
        self.metadata = {}           # Signal metadata keyed by ID
        self.creation_time = time.time()
        
        # Thread-safety lock for registry operations
        self.registry_lock = threading.Lock()
        
        logger.info("SignalRegistry initialized")
    
    def register_signal(self, signal_id, signal_data, metadata=None):
        """
        Register a signal in the registry.
        
        Args:
            signal_id (str): Unique identifier for the signal
            signal_data (array): The signal data (numpy array, tensor, or list)
            metadata (dict, optional): Metadata for the signal (type, source, etc.)
        
        Returns:
            str: The signal ID (same as input if provided)
        """
        if not isinstance(signal_id, str):
            logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
            return None
            
        with self.registry_lock:
            # Convert tensors to numpy
            if isinstance(signal_data, torch.Tensor):
                signal_data = signal_data.detach().cpu().numpy()
            # Convert to numpy array if needed
            elif not isinstance(signal_data, np.ndarray):
                try:
                    signal_data = np.array(signal_data)
                except Exception as e:
                    logger.error(f"Failed to convert signal data to numpy array: {e}")
                    return None
                
            # Store the signal data
            self.signals[signal_id] = signal_data
            
            # Store metadata if provided
            if metadata:
                self.metadata[signal_id] = metadata
            elif signal_id not in self.metadata:
                # Default metadata if none exists
                self.metadata[signal_id] = {
                    'created': time.time(),
                    'source': 'unknown'
                }
                
            logger.info(f"Signal '{signal_id}' registered with shape {signal_data.shape}")
            return signal_id
            
    def get_signal(self, signal_id):
        """
        Get a signal by its ID
        
        Args:
            signal_id (str): ID of the signal to retrieve
            
        Returns:
            numpy.ndarray or None: The signal data or None if not found
        """
        with self.registry_lock:
            if signal_id in self.signals:
                return self.signals[signal_id]
            logger.warning(f"Signal '{signal_id}' not found in registry")
            return None
    
    def get_signal_metadata(self, signal_id):
        """
        Get metadata for a signal
        
        Args:
            signal_id (str): ID of the signal
            
        Returns:
            dict or None: The signal metadata or None if not found
        """
        with self.registry_lock:
            if signal_id in self.metadata:
                return self.metadata[signal_id]
            return None
    
    def remove_signal(self, signal_id):
        """
        Remove a signal from the registry
        
        Args:
            signal_id (str): ID of the signal to remove
            
        Returns:
            bool: True if removed, False otherwise
        """
        with self.registry_lock:
            if signal_id in self.signals:
                del self.signals[signal_id]
                if signal_id in self.metadata:
                    del self.metadata[signal_id]
                logger.info(f"Removed signal '{signal_id}' from registry")
                return True
            return False
    
    def get_all_signals(self):
        """
        Get all signal IDs in the registry
        
        Returns:
            list: List of all signal IDs
        """
        with self.registry_lock:
            return list(self.signals.keys())
    
    def clear_signals(self):
        """Clear all signals from the registry"""
        with self.registry_lock:
            signal_count = len(self.signals)
            self.signals.clear()
            self.metadata.clear()
            logger.info(f"Cleared {signal_count} signals from registry")
    
    def reset(self):
        """Reset the entire registry"""
        self.clear_signals()
        self.creation_time = time.time()
        logger.info("SignalRegistry completely reset")
