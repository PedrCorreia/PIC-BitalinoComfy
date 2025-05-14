"""
Data conversion utilities for the PlotUnit visualization system.

This module provides data conversion utilities for the PlotUnit system,
handling conversions between different data formats for visualization.

Note: For signal processing functionality, use the SignalProcessingAdapter
or directly use the NumpySignalProcessor from src.utils.signal_processing.
"""

import numpy as np
import torch
import logging
from ...utils.signal_processing import NumpySignalProcessor

# Configure logger
logger = logging.getLogger('DataConverter')

def convert_to_numpy(data):
    """
    Convert various data types to numpy arrays for visualization.
    
    Handles conversions from:
    - PyTorch tensors
    - Lists
    - Numpy arrays
    - Scalar values
    
    Args:
        data: The input data to convert
        
    Returns:
        numpy.ndarray: The data as a numpy array
    """
    try:
        # Handle PyTorch tensors
        if isinstance(data, torch.Tensor):
            # Convert to CPU if needed
            if data.is_cuda:
                data = data.cpu()
                
            # Detach from computation graph if needed
            if data.requires_grad:
                data = data.detach()
                
            # Convert to numpy
            return data.numpy()
        
        # Handle lists and tuples
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        
        # Handle numpy arrays
        elif isinstance(data, np.ndarray):
            return data
        
        # Handle scalars
        elif isinstance(data, (int, float)):
            return np.array([data])
        
        # Handle unsupported types
        else:
            logger.warning(f"Unsupported data type for conversion: {type(data)}")
            return np.zeros(10)  # Return empty array
            
    except Exception as e:
        logger.error(f"Error converting data: {e}")
        return np.zeros(10)  # Return empty array on error

def normalize_signal(data, min_val=None, max_val=None):
    """
    Normalize signal data to a 0-1 range.
    
    Args:
        data (numpy.ndarray): The input data to normalize
        min_val (float, optional): Minimum value for normalization. If None, uses data minimum.
        max_val (float, optional): Maximum value for normalization. If None, uses data maximum.
        
    Returns:
        numpy.ndarray: The normalized data
    """
    if data is None or len(data) == 0:
        return np.zeros(10)
        
    # Use provided min/max or calculate from data
    if min_val is None:
        min_val = np.min(data)
    
    if max_val is None:
        max_val = np.max(data)
    
    # Handle case where min equals max
    if min_val == max_val:
        return np.zeros_like(data)
    
    # Normalize
    return (data - min_val) / (max_val - min_val)

def resample_signal(data, target_length):
    """
    Resample a signal to a target length.
    
    Args:
        data (numpy.ndarray): The input data to resample
        target_length (int): Target length for resampled signal
        
    Returns:
        numpy.ndarray: The resampled data
    """
    if data is None or len(data) == 0:
        return np.zeros(target_length)
        
    # If already the right length, return as is
    if len(data) == target_length:
        return data
        
    # Simple linear resampling using numpy
    indices = np.linspace(0, len(data) - 1, target_length)
    indices_floor = np.floor(indices).astype(int)
    indices_ceil = np.ceil(indices).astype(int)
    indices_remainder = indices - indices_floor
    
    # Handle edge case
    indices_ceil = np.minimum(indices_ceil, len(data) - 1)
    
    # Linear interpolation
    resampled = data[indices_floor] * (1 - indices_remainder) + data[indices_ceil] * indices_remainder
    
    return resampled
