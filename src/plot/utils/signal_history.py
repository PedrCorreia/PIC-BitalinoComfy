"""
SignalHistoryManager - Manages signal history for the PlotUnit visualization system.

This module provides functionality for storing and retrieving signal history
for visualization purposes, without duplicating signal processing capabilities.
"""

import numpy as np
from collections import deque
import logging
from ...utils.signal_processing import NumpySignalProcessor

# Configure logger
logger = logging.getLogger('SignalHistoryManager')

class SignalHistoryManager:
    """
    Signal history manager for the PlotUnit visualization system.
    
    This class manages signal history buffers for visualization,
    providing storage, retrieval, and simple analysis of signal history.
    It delegates actual signal processing to the NumpySignalProcessor.
    
    Attributes:
        signal_history (dict): Dictionary of signal history buffers
        history_length (int): Length of the signal history buffer
        processor (NumpySignalProcessor): Reference to signal processor
    """
    
    def __init__(self, history_length=500):
        """
        Initialize the signal history manager.
        
        Args:
            history_length (int, optional): Length of the signal history buffer
        """
        self.signal_history = {}
        self.history_length = history_length
        self.processor = NumpySignalProcessor()
    
    def update_history(self, signal_id, data):
        """
        Update the signal history with new data.
        
        Args:
            signal_id (str): ID of the signal
            data (numpy.ndarray): New signal data
        """
        # Initialize history buffer if needed
        if signal_id not in self.signal_history:
            self.signal_history[signal_id] = deque(maxlen=self.history_length)
            
        # Add data to history
        if len(data) > 0:
            for sample in data:
                self.signal_history[signal_id].append(sample)
    
    def get_history(self, signal_id):
        """
        Get the signal history.
        
        Args:
            signal_id (str): ID of the signal
            
        Returns:
            numpy.ndarray: Signal history as array, or empty array if no history
        """
        if signal_id in self.signal_history:
            if isinstance(self.signal_history[signal_id], deque):
                # Use the utility function from NumpySignalProcessor to convert deque to numpy
                return NumpySignalProcessor.deque_to_numpy(self.signal_history[signal_id])
            return np.array(self.signal_history[signal_id])
        else:
            return np.array([])
    
    def clear_history(self, signal_id=None):
        """
        Clear the signal history.
        
        Args:
            signal_id (str, optional): ID of the signal to clear. 
                                      If None, clear all histories.
        """
        if signal_id is None:
            self.signal_history.clear()
        elif signal_id in self.signal_history:
            self.signal_history[signal_id].clear()
            
    def get_signal_info(self, signal_id):
        """
        Get information about a signal's history.
        
        Args:
            signal_id (str): ID of the signal
            
        Returns:
            dict: Signal information including length, min, max, etc.
        """
        if signal_id not in self.signal_history or len(self.signal_history[signal_id]) == 0:
            return {"error": "No signal history available"}
            
        data = self.get_history(signal_id)
        
        return {
            'length': len(data),
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }
    
    def apply_processing(self, signal_id, processing_type, **kwargs):
        """
        Apply processing to a signal history using NumpySignalProcessor.
        
        Args:
            signal_id (str): ID of the signal
            processing_type (str): Type of processing to apply
            **kwargs: Additional arguments for the processing function
            
        Returns:
            numpy.ndarray: Processed signal data
        """
        if signal_id not in self.signal_history:
            return np.array([])
            
        data = self.get_history(signal_id)
        
        # Apply processing using NumpySignalProcessor
        if processing_type == 'bandpass':
            return NumpySignalProcessor.bandpass_filter(data, **kwargs)
        elif processing_type == 'lowpass':
            return NumpySignalProcessor.lowpass_filter(data, **kwargs)
        elif processing_type == 'highpass':
            return NumpySignalProcessor.highpass_filter(data, **kwargs)
        elif processing_type == 'normalize':
            return NumpySignalProcessor.normalize_signal(data)
        elif processing_type == 'moving_average':
            return NumpySignalProcessor.moving_average(data, **kwargs)
        else:
            logger.warning(f"Unknown processing type: {processing_type}")
            return data
