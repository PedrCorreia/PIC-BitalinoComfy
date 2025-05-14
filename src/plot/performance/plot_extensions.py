"""
Performance extensions for the PlotUnit visualization system.

This module provides extensions for enhanced performance monitoring,
settings management, and advanced visualization features.
"""

# =====IMPORTS=====
import numpy as np
import time
import logging
import pygame

# =====LOGGER CONFIGURATION=====
logger = logging.getLogger('PlotExtensions')

# =====CLASS: PlotExtensions=====
class PlotExtensions:
    """
    Extensions for the PlotUnit class with advanced plotting features
    inspired by the PygamePlotTemplate.

    This class provides methods that can be added to the PlotUnit
    without modifying its core functionality.

    Attributes:
        plot_unit (PlotUnit): Reference to the parent PlotUnit instance
        _last_draw_time (float): Timestamp of the last draw operation
        _real_time_start (float): Timestamp of when real-time tracking started
        _last_latency (float): Last measured latency value
        sampling_rate (float): Current sampling rate if available
        FPS (int): Target frames per second
        FPS_CAP_ENABLED (bool): Whether to cap FPS
        PERFORMANCE_MODE (bool): Whether to optimize for performance
        SMART_DOWNSAMPLE (bool): Whether to use smart downsampling
        LINE_THICKNESS (int): Thickness of plot lines
        SIGNAL_COLORS (dict): Color presets for different signal types
    """

    # =====PERFORMANCE SETTINGS=====
    FPS = 60
    FPS_CAP_ENABLED = True
    PERFORMANCE_MODE = False
    SMART_DOWNSAMPLE = False  # Default to OFF for downsampling
    LINE_THICKNESS = 1

    # =====COLOR PRESETS=====
    SIGNAL_COLORS = {
        "EDA": (0, 255, 0),      # Green for EDA
        "ECG": (255, 0, 0),      # Red for ECG
        "RR": (255, 165, 0),     # Orange for RR
        "R": (0, 0, 220),        # Blue for Raw signals
        "P": (220, 0, 0),        # Red for Processed signals
        "DEFAULT": (0, 180, 220)  # Cyan for default
    }

    # =====INITIALIZATION=====
    def __init__(self, plot_unit):
        """
        Initialize the extension with a reference to the PlotUnit instance

        Args:
            plot_unit (PlotUnit): Reference to the parent PlotUnit instance
        """
        self.plot_unit = plot_unit
        self._last_draw_time = time.time()
        self._real_time_start = time.time()
        self._last_latency = 0.0
        self.sampling_rate = None

    # =====UPDATE VISUALIZATION=====
    def update(self):
        """
        Force an update of the visualization.

        This method sends a refresh event to the PlotUnit's event queue
        to trigger a redraw of the visualization.
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

    # =====CLEAR PLOTS=====
    def clear_plots(self):
        """
        Clear all plots from the visualization.

        This method removes all data from the PlotUnit's data dictionary
        and replaces it with empty arrays.
        """
        if not hasattr(self.plot_unit, 'data'):
            logger.error("Cannot clear plots: PlotUnit missing data dictionary")
            return

        # Clear all data
        with self.plot_unit.data_lock:
            for key in list(self.plot_unit.data.keys()):
                self.plot_unit.data[key] = np.zeros((0, 2))  # Empty array with 2 columns (time, value)

    # =====PERFORMANCE MODE=====
    def set_performance_mode(self, enabled=True):
        """
        Enable or disable performance mode.

        In performance mode, visualizations are optimized for speed
        at the expense of some visual quality.

        Args:
            enabled (bool): Whether to enable performance mode
        """
        self.PERFORMANCE_MODE = enabled
        self.SMART_DOWNSAMPLE = enabled
        self.LINE_THICKNESS = 1 if enabled else 2
        logger.info(f"Performance mode {'enabled' if enabled else 'disabled'}")

    # =====FPS CAP=====
    def set_fps_cap(self, enabled=True, fps=60):
        """
        Enable or disable FPS cap and set the target FPS.

        Args:
            enabled (bool): Whether to enable FPS cap
            fps (int): Target FPS if cap is enabled
        """
        self.FPS_CAP_ENABLED = enabled
        self.FPS = fps if fps > 0 else 60
        logger.info(f"FPS cap {'enabled' if enabled else 'disabled'}, target: {self.FPS}")

    # =====LATENCY CALCULATION=====
    def calculate_latency(self, latest_timestamp):
        """
        Calculate the latency between the latest data point and current time.

        Args:
            latest_timestamp (float): Timestamp of the latest data point

        Returns:
            tuple: (latency, is_stable, is_acceptable) where
                latency (float): Latency in seconds
                is_stable (bool): Whether latency is stable
                is_acceptable (bool): Whether latency is within acceptable range
        """
        if latest_timestamp is None:
            return None, False, False

        now = time.time()
        latency = now - latest_timestamp

        # Check if latency is stable (not changing much)
        is_stable = abs(latency - self._last_latency) < 0.1 if hasattr(self, '_last_latency') else False

        # Check if latency is acceptable (<0.5s)
        is_acceptable = latency <= 0.5

        self._last_latency = latency

        return latency, is_stable, is_acceptable

    # =====DRAW PERFORMANCE METRICS=====
    def draw_performance_metrics(self, surface, font, x=10, y=10):
        """
        Draw performance metrics on the visualization surface.

        Args:
            surface (pygame.Surface): Surface to draw on
            font (pygame.font.Font): Font for text rendering
            x (int): X position for drawing
            y (int): Y position for drawing

        Returns:
            int: Updated Y position after drawing metrics
        """
        # Calculate elapsed time since start
        now = time.time()
        elapsed = now - self._real_time_start
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60

        # Real-time tracking
        real_time_str = f"Time: {minutes:02}:{seconds:02}"
        real_time_surface = font.render(real_time_str, True, (255, 255, 0))
        surface.blit(real_time_surface, (x, y))
        y += 25

        # Render sampling rate if available
        if self.sampling_rate:
            sr_str = f"Sample Rate: {self.sampling_rate:.1f} Hz"
            sr_surface = font.render(sr_str, True, (200, 200, 200))
            surface.blit(sr_surface, (x, y))
            y += 25

        # FPS calculation
        frame_time = now - self._last_draw_time
        fps = 1.0 / max(frame_time, 0.001)
        fps_str = f"FPS: {fps:.1f}"
        fps_color = (0, 255, 0) if fps >= 30 else (255, 165, 0) if fps >= 15 else (255, 0, 0)
        fps_surface = font.render(fps_str, True, fps_color)
        surface.blit(fps_surface, (x, y))
        y += 25

        # Update last draw time
        self._last_draw_time = now

        return y

    # =====SMART DOWNSAMPLING=====
    def smart_downsample(self, points, target_points=1000):
        """
        Intelligently downsample points for more efficient rendering.

        This method uses a combination of simple decimation and
        local extrema preservation to reduce the number of points
        while preserving the shape.

        Args:
            points (np.ndarray): Points to downsample (shape [N, 2])
            target_points (int): Approximate target number of points

        Returns:
            np.ndarray: Downsampled points
        """
        if not self.SMART_DOWNSAMPLE or len(points) <= target_points:
            return points

        # Simple decimation for large arrays
        if len(points) > target_points * 10:
            step = len(points) // target_points
            return points[::step]

        # For smaller arrays, use a simple but effective method
        # that preserves extremes
        indices = np.arange(len(points))
        if len(indices) <= target_points:
            return points

        # Always include first and last point
        selected = np.zeros(len(points), dtype=bool)
        selected[0] = True
        selected[-1] = True

        # Include local maxima and minima
        y = points[:, 1]
        for i in range(1, len(y) - 1):
            if (y[i] > y[i-1] and y[i] > y[i+1]) or (y[i] < y[i-1] and y[i] < y[i+1]):
                selected[i] = True

        # If still need more points, add some regularly spaced ones
        if selected.sum() < target_points:
            remaining = target_points - selected.sum()
            step = len(points) // remaining
            for i in range(1, len(points) - 1, step):
                if not selected[i]:
                    selected[i] = True
                    remaining -= 1
                if remaining <= 0:
                    break

        return points[selected]
