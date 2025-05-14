"""
LatencyMonitor module for the PlotUnit visualization system.

This module provides latency tracking and monitoring functionality
for the PlotUnit system, helping identify signal processing delays.
"""

import time
from collections import deque

class LatencyMonitor:
    """
    Latency monitor for tracking signal processing delays.
    
    This class tracks timestamps for signals and calculates latency
    metrics to help identify processing bottlenecks.
    
    Attributes:
        signal_times (dict): Dictionary mapping signal IDs to timestamps
        latency_history (deque): Queue of recent latency measurements
        last_update_time (float): Timestamp of the most recent signal update
    """
    
    def __init__(self, history_size=30):
        """
        Initialize the latency monitor.
        
        Args:
            history_size (int, optional): Size of the latency history queue
        """
        self.signal_times = {}
        self.latency_history = deque(maxlen=history_size)
        self.last_update_time = time.time()
        self.last_calculated_latency = 0.0
        
    def update_signal_time(self, signal_id):
        """
        Record a signal update timestamp.
        
        Args:
            signal_id (str): ID of the updated signal
        """
        current_time = time.time()
        self.signal_times[signal_id] = current_time
        self.last_update_time = current_time
        
        # Calculate and record latency if we have multiple signals
        if len(self.signal_times) > 1:
            self._calculate_latency()
    
    def _calculate_latency(self):
        """
        Calculate the current latency between signal updates.
        """
        if not self.signal_times:
            return
            
        # Find the oldest and newest signal timestamps
        oldest = min(self.signal_times.values())
        newest = max(self.signal_times.values())
        
        # Calculate latency between oldest and newest signal
        current_latency = newest - oldest
        
        # Store in history
        self.latency_history.append(current_latency)
        self.last_calculated_latency = current_latency
    
    def get_current_latency(self):
        """
        Get the current signal latency.
        
        Returns:
            float: Current latency in seconds
        """
        return self.last_calculated_latency
    
    def get_average_latency(self):
        """
        Get the average latency over recent history.
        
        Returns:
            float: Average latency in seconds
        """
        if not self.latency_history:
            return 0.0
            
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_signal_times(self):
        """
        Get the dictionary of signal timestamps.
        
        Returns:
            dict: Dictionary mapping signal IDs to timestamps
        """
        return self.signal_times
    
    def get_time_since_last_update(self):
        """
        Get the time elapsed since the last signal update.
        
        Returns:
            float: Time in seconds since last update
        """
        return time.time() - self.last_update_time
    
    def clear(self):
        """
        Clear all latency tracking data.
        """
        self.signal_times.clear()
        self.latency_history.clear()
        self.last_update_time = time.time()
        self.last_calculated_latency = 0.0
