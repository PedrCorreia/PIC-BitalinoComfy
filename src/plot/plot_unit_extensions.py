"""
Enhanced plot unit functionality inspired by PlotPygameTemplate.
This module extends the base PlotUnit class with additional features.
"""

import numpy as np
import time
import threading
import logging
import pygame

# Configure logger
logger = logging.getLogger('PlotUnitExtensions')

class PlotUnitExtensions:
    """
    Extensions for the PlotUnit class with advanced plotting features
    inspired by the PygamePlotTemplate.
    
    This class provides methods that can be added to the PlotUnit
    without modifying its core functionality.
    """
    
    # Performance settings
    FPS = 60
    FPS_CAP_ENABLED = True
    PERFORMANCE_MODE = False
    SMART_DOWNSAMPLE = False  # Default to OFF for downsampling
    LINE_THICKNESS = 1
    
    # Color presets
    SIGNAL_COLORS = {
        "EDA": (0, 255, 0),      # Green for EDA
        "ECG": (255, 0, 0),      # Red for ECG
        "RR": (255, 165, 0),     # Orange for RR
        "R": (0, 0, 220),        # Blue for Raw signals
        "P": (220, 0, 0),        # Red for Processed signals
        "DEFAULT": (0, 180, 220)  # Cyan for default
    }
    
    def __init__(self, plot_unit):
        """
        Initialize the extension with a reference to the PlotUnit instance
        """
        self.plot_unit = plot_unit
        self._last_draw_time = time.time()
        self._real_time_start = time.time()
        self._last_latency = 0.0
        self.sampling_rate = None
        
    def update(self):
        """
        Force an update of the visualization
        This method was missing and causing errors
        """
        if not hasattr(self.plot_unit, 'event_queue'):
            logger.error("Cannot update: PlotUnit missing event_queue")
            return
            
        # Put a refresh message in the queue
        self.plot_unit.event_queue.put({
            'type': 'refresh',
            'timestamp': time.time()
        })
        
        # Log the update request
        logger.info("Plot update requested")
        
    def clear_plots(self):
        """
        Clear all plots from the visualization
        """
        if not hasattr(self.plot_unit, 'data'):
            logger.error("Cannot clear plots: PlotUnit missing data dictionary")
            return
            
        # Clear all data
        with self.plot_unit.data_lock:
            for key in list(self.plot_unit.data.keys()):
                self.plot_unit.data[key] = np.zeros(100)
        
        # Put a clear message in the queue
        self.plot_unit.event_queue.put({
            'type': 'clear_plots'
        })
        
        logger.info("All plots cleared")
        
    def _safe_min_max(self, arr):
        """
        Calculate min and max of array, safely handling empty arrays and NaN values
        """
        if arr is None or len(arr) == 0:
            return 0.0, 1.0

        # Filter out NaN values
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            return 0.0, 1.0

        return float(np.min(valid)), float(np.max(valid))
        
    def _smart_downsample(self, x_arr, y_arr, target_points=1000):
        """
        Intelligently downsample a signal based on its characteristics
        """
        if len(x_arr) <= target_points:
            return x_arr, y_arr
            
        # Get length of data
        data_len = len(x_arr)
        
        # Calculate downsample factor
        downsample_factor = max(1, data_len // target_points)
        
        # Different strategies based on signal type
        signal_type = getattr(self, 'signal_type', 'DEFAULT').upper()
        
        if signal_type == 'ECG':
            # For ECG, preserve peaks by taking local max values in windows
            result_x = []
            result_y = []
            
            for i in range(0, data_len, downsample_factor):
                end_idx = min(i + downsample_factor, data_len)
                window_x = x_arr[i:end_idx]
                window_y = y_arr[i:end_idx]
                
                if len(window_y) > 0:
                    max_idx = np.argmax(window_y)
                    result_x.append(window_x[max_idx])
                    result_y.append(window_y[max_idx])
                    
            return np.array(result_x), np.array(result_y)
            
        else:
            # Simple downsampling for other signal types
            return x_arr[::downsample_factor], y_arr[::downsample_factor]
            
    def _draw_smooth_line(self, surface, color, points, thickness=1):
        """
        Draw a smoother line by using anti-aliasing and optional interpolation
        """
        if len(points) < 2:
            return
            
        # For thick lines, use aalines with blend
        if thickness <= 1:
            pygame.draw.aalines(surface, color, False, points)
        else:
            # For thicker lines, first draw a regular line
            pygame.draw.lines(surface, color, False, points, thickness)
            # Then overlay an anti-aliased line for smoothing the edges
            pygame.draw.aalines(surface, color, False, points)
            
    def get_status_info(self):
        """
        Get current status information for display
        """
        now = time.time()
        
        # Real time tracking
        real_elapsed = now - self._real_time_start
        real_minutes = int(real_elapsed) // 60
        real_seconds = int(real_elapsed) % 60
        
        # Format status information
        status = {
            'real_time': f"{real_minutes:02}:{real_seconds:02}",
            'latency': getattr(self, '_last_latency', 0.0),
            'mode': 'Performance' if self.PERFORMANCE_MODE else 'Quality',
            'fps_cap': self.FPS if self.FPS_CAP_ENABLED else 'Unlimited',
            'sampling_rate': self.sampling_rate,
            'connected_nodes': self.plot_unit.settings['connected_nodes'] if hasattr(self.plot_unit, 'settings') else 0
        }
        
        return status
        
    def set_sampling_rate(self, rate):
        """
        Set the sampling rate for display
        """
        self.sampling_rate = rate
        logger.info(f"Sampling rate set to {rate} Hz")
