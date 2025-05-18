"""
Debug plot utilities for the PlotUnit visualization system.

This module provides utility functions to generate static test plots
for debugging the UI layout and view modes using a simplified approach.
"""

import numpy as np
import logging
import time

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DebugPlots')

def generate_sine_wave(length=1000, frequency=1.0, amplitude=1.0, phase=0.0):
    """
    Generate a sine wave for test plots.
    
    Args:
        length (int): Length of the array
        frequency (float): Frequency of the sine wave
        amplitude (float): Amplitude of the sine wave
        phase (float): Phase offset in radians
        
    Returns:
        numpy.ndarray: Array containing the sine wave
    """
    t = np.linspace(0, 10, length)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def generate_debug_data():
    """
    Generate a dictionary of test signals.
    
    Returns:
        dict: Dictionary mapping signal IDs to numpy arrays
    """
    logger.info("Generating debug plot data")
    
    data = {}
    
    # Create time array
    samples = 1000
    t = np.linspace(0, 10, samples)
    
    try:
        # Raw signals (simpler naming to avoid import issues)
        data["ECG_RAW"] = 0.8 * np.sin(2 * np.pi * 1.0 * t)
        data["RESP_RAW"] = 0.7 * np.sin(2 * np.pi * 0.3 * t)
        data["EDA_RAW"] = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        
        # Add noise to raw signals
        data["ECG_RAW"] += 0.1 * np.random.randn(samples)
        data["RESP_RAW"] += 0.05 * np.random.randn(samples)
        data["EDA_RAW"] += 0.02 * np.random.randn(samples)
        
        # Processed signals
        data["ECG_PROCESSED"] = 0.8 * np.sin(2 * np.pi * 1.0 * t + 0.2)
        data["RESP_PROCESSED"] = 0.7 * np.sin(2 * np.pi * 0.3 * t + 0.1)
        data["EDA_PROCESSED"] = 0.5 * np.sin(2 * np.pi * 0.1 * t + 0.3)
        
        # Additional test signal
        data["COMPOSITE"] = data["ECG_RAW"] * 0.3 + data["RESP_RAW"] * 0.7
        
        logger.info(f"Generated {len(data)} debug signals")
        
    except Exception as e:
        logger.error(f"Error generating debug data: {e}")
    
    return data

def initialize_test_plots(plot_unit):
    """
    Initialize the PlotUnit with test plots.
    
    Args:
        plot_unit: PlotUnit instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Initializing test plots for PlotUnit")
        
        if not hasattr(plot_unit, 'queue_data'):
            logger.error("PlotUnit has no queue_data method")
            return False
            
        # Generate debug data
        data = generate_debug_data()
        
        # Feed data to PlotUnit
        for signal_id, signal_data in data.items():
            try:
                plot_unit.queue_data(signal_id, signal_data)
                logger.info(f"Added signal: {signal_id}")
            except Exception as e:
                logger.error(f"Error queuing data for {signal_id}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in initialize_test_plots: {e}")
        return False

def update_test_plots(plot_unit):
    """
    Update existing test plots with slightly modified data.
    
    Args:
        plot_unit: PlotUnit instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(plot_unit, 'queue_data'):
            logger.error("PlotUnit has no queue_data method")
            return False
            
        # Get existing data
        existing_data = {}
        if hasattr(plot_unit, 'data') and hasattr(plot_unit, 'data_lock'):
            with plot_unit.data_lock:
                existing_data = {k: np.copy(v) for k, v in plot_unit.data.items() if isinstance(v, np.ndarray)}
        
        # If no existing data, generate new data
        if not existing_data:
            return initialize_test_plots(plot_unit)
        
        # Update each signal
        phase_shift = np.random.uniform(0.05, 0.1)
        for signal_id, signal_data in existing_data.items():
            if len(signal_data) > 0:
                # Shift the signal slightly
                try:
                    rotated = np.roll(signal_data, int(len(signal_data) * 0.05))
                    # Add some random noise
                    noisy = rotated + 0.02 * np.random.randn(len(rotated))
                    plot_unit.queue_data(signal_id, noisy)
                except Exception as e:
                    logger.error(f"Error updating {signal_id}: {e}")
                    
        return True
        
    except Exception as e:
        logger.error(f"Error in update_test_plots: {e}")
        return False
