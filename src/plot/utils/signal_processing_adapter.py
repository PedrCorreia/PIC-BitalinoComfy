"""
Signal Processing Adapter - Provides easy access to signal processing utilities.

This module acts as an adapter to the main signal processing utilities
in the src/utils/signal_processing.py module, making them more easily
accessible to the plot module.
"""

import numpy as np
import logging
from ...utils.signal_processing import NumpySignalProcessor

# Configure logger
logger = logging.getLogger('SignalProcessingAdapter')

class SignalProcessingAdapter:
    """
    Adapter class for signal processing utilities.
    
    This class provides simplified access to the signal processing utilities
    in the NumpySignalProcessor class from the src/utils directory.
    """
    
    @staticmethod
    def process_signal(data, processing_type='default', **kwargs):
        """
        Process a signal based on the processing type.
        
        Args:
            data: The signal data
            processing_type (str): Type of processing to apply
            **kwargs: Additional arguments for the processing function
            
        Returns:
            numpy.ndarray: Processed signal data
        """
        # Process based on type
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
        elif processing_type == 'baseline_correction':
            return NumpySignalProcessor.correct_baseline(data, **kwargs)
        elif processing_type == 'find_peaks':
            return NumpySignalProcessor.find_peaks(data, **kwargs)
        else:
            # Default processing (just return the data)
            return data
    
    @staticmethod
    def analyze_signal(data, analysis_type='stats'):
        """
        Analyze a signal and return statistics.
        
        Args:
            data: The signal data
            analysis_type (str): Type of analysis to perform
            
        Returns:
            dict: Analysis results
        """
        if len(data) == 0:
            return {"error": "No signal data available"}
        
        if analysis_type == 'stats':
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'length': len(data)
            }
        elif analysis_type == 'psd':
            # Use the power spectral density analysis
            try:
                fs = 1000  # Default sampling frequency
                freqs, psd = NumpySignalProcessor.compute_psd_numpy(data, fs)
                return {
                    'frequencies': freqs,
                    'psd': psd
                }
            except Exception as e:
                logger.error(f"Error computing PSD: {e}")
                return {"error": f"Error computing PSD: {e}"}
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
