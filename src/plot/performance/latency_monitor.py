"""
=====Overview=====
LatencyMonitor module for the PlotUnit visualization system.

This module provides latency tracking and monitoring functionality
for the PlotUnit system, helping identify signal processing delays.

=====Usage=====
Instantiate the LatencyMonitor and call update_signal_time(signal_id)
whenever a signal is processed. Retrieve latency metrics using the
provided getter methods.

Example:
    monitor = LatencyMonitor()
    monitor.update_signal_time("signal_1")
    monitor.update_signal_time("signal_2")
    print(monitor.get_current_latency())
    print(monitor.get_average_latency())

=====Classes=====
- LatencyMonitor: Tracks timestamps for signals and calculates latency
  metrics to help identify processing bottlenecks.

=====API Reference=====
"""

import time
from collections import deque

class LatencyMonitor:
    """
    =====Class: LatencyMonitor=====

    Latency monitor for tracking signal processing delays.
    
    This class tracks timestamps for signals and calculates latency
    metrics to help identify processing bottlenecks.
    
    Attributes:
        signal_times (dict): Dictionary mapping signal IDs to timestamps
        latency_history (deque): Queue of recent latency measurements
        last_update_time (float): Timestamp of the most recent signal update
        last_calculated_latency (float): Most recently calculated latency value
    """
    
    def __init__(self, history_size=30):
        """
        =====Method: __init__=====
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
        =====Method: update_signal_time=====
        Record a signal update timestamp.
        
        Args:
            signal_id (str): ID of the updated signal

        Side Effects:
            Updates signal_times, last_update_time, and latency metrics.
        """
        current_time = time.time()
        self.signal_times[signal_id] = current_time
        self.last_update_time = current_time
        
        # Calculate and record latency if we have multiple signals
        if len(self.signal_times) > 1:
            self._calculate_latency()
    
    def _calculate_latency(self):
        """
        =====Method: _calculate_latency (private)=====
        Calculate the current latency between signal updates.

        Finds the oldest and newest timestamps in signal_times and
        computes the difference as the current latency. Updates
        latency_history and last_calculated_latency.
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
        =====Method: get_current_latency=====
        Get the current signal latency.
        
        Returns:
            float: Current latency in seconds
        """
        return self.last_calculated_latency
    
    def get_latency(self):
        """
        Alias for get_current_latency() for compatibility with main app usage.
        
        Returns:
            float: Current latency in seconds
        """
        return self.get_current_latency()

    
    def get_average_latency(self):
        """
        =====Method: get_average_latency=====
        Get the average latency over recent history.
        
        Returns:
            float: Average latency in seconds
        """
        if not self.latency_history:
            return 0.0
            
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_signal_times(self):
        """
        =====Method: get_signal_times=====
        Get the dictionary of signal timestamps.
        
        Returns:
            dict: Dictionary mapping signal IDs to timestamps
        """
        return self.signal_times
    
    def get_time_since_last_update(self):
        """
        =====Method: get_time_since_last_update=====
        Get the time elapsed since the last signal update.
        
        Returns:
            float: Time in seconds since last update
        """
        return time.time() - self.last_update_time
    
    def clear(self):
        """
        =====Method: clear=====
        Clear all latency tracking data.

        Resets signal_times, latency_history, last_update_time, and last_calculated_latency.
        """
        self.signal_times.clear()
        self.latency_history.clear()
        self.last_update_time = time.time()
        self.last_calculated_latency = 0.0
