"""
=====Overview=====
EnhancedLatencyMonitor module for the PlotUnit visualization system.

This module provides improved latency tracking and monitoring functionality
for the PlotUnit system, helping identify signal processing delays with
better protection against latency issues.

=====Usage=====
Instantiate the EnhancedLatencyMonitor and call record_signal(signal_id, timestamp)
whenever a signal is processed. Retrieve latency metrics using the
provided getter methods.

Example:
    monitor = EnhancedLatencyMonitor()
    monitor.record_signal("signal_1", timestamp)
    monitor.record_signal("signal_2", timestamp)
    print(monitor.get_current_latency())
    print(monitor.get_average_latency())
    print(monitor.get_latency_status())

=====Classes=====
- EnhancedLatencyMonitor: Tracks timestamps for signals and calculates latency
  metrics with additional protection mechanisms.

=====API Reference=====
"""

import time
from collections import deque

class EnhancedLatencyMonitor:
    """
    =====Class: EnhancedLatencyMonitor=====

    Enhanced latency monitor for tracking signal processing delays with improved protection.
    
    This class tracks timestamps for signals and calculates latency
    metrics to help identify and mitigate processing bottlenecks.
    
    Attributes:
        signal_times (dict): Dictionary mapping signal IDs to timestamps
        latency_history (deque): Queue of recent latency measurements
        last_update_time (float): Timestamp of the most recent signal update
        last_calculated_latency (float): Most recently calculated latency value
        max_acceptable_latency (float): Maximum acceptable latency before warning or throttling
        latency_status (dict): Current status of latency metrics
    """
    
    # Latency thresholds in seconds
    LOW_LATENCY = 0.05      # 50ms - excellent
    MEDIUM_LATENCY = 0.1    # 100ms - acceptable
    HIGH_LATENCY = 0.2      # 200ms - problematic
    CRITICAL_LATENCY = 0.5  # 500ms - critical, requires intervention
    
    def __init__(self, history_size=30, max_acceptable_latency=0.5):
        """
        =====Method: __init__=====
        Initialize the enhanced latency monitor.
        
        Args:
            history_size (int, optional): Size of the latency history queue
            max_acceptable_latency (float, optional): Maximum acceptable latency in seconds
        """
        self.signal_times = {}
        self.latency_history = deque(maxlen=history_size)
        self.last_update_time = time.time()
        self.last_calculated_latency = 0.0
        self.max_acceptable_latency = max_acceptable_latency
        self.latency_status = {
            'status': 'unknown',
            'color': (200, 200, 200),  # Default gray
            'value': 0.0,
            'threshold_exceeded': False,
            'last_check': time.time()
        }
        
    def record_signal(self, signal_id, timestamp=None):
        """
        =====Method: record_signal=====
        Record a signal update with timestamp.
        
        Args:
            signal_id (str): ID of the updated signal
            timestamp (float, optional): Custom timestamp, or current time if None

        Returns:
            dict: Updated latency status

        Side Effects:
            Updates signal_times, last_update_time, and latency metrics.
        """
        current_time = timestamp or time.time()
        self.signal_times[signal_id] = current_time
        self.last_update_time = current_time
        
        # Calculate and record latency if we have multiple signals
        if len(self.signal_times) > 1:
            self._calculate_latency()
            
        # Update status
        self._update_latency_status()
        
        return self.latency_status
        
    # For backward compatibility with older code
    def update_signal_time(self, signal_id):
        """
        =====Method: update_signal_time=====
        Backward compatibility method that calls record_signal.
        
        Args:
            signal_id (str): ID of the updated signal
            
        Returns:
            dict: Updated latency status
        """
        return self.record_signal(signal_id)
    
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
            
        try:
            # Find the oldest and newest signal timestamps
            oldest = min(self.signal_times.values())
            newest = max(self.signal_times.values())
            
            # Calculate latency between oldest and newest signal
            current_latency = newest - oldest
            
            # Store in history
            self.latency_history.append(current_latency)
            self.last_calculated_latency = current_latency
        except Exception as e:
            # Ensure error handling doesn't break visualization
            print(f"Error calculating latency: {e}")
    
    def _update_latency_status(self):
        """
        =====Method: _update_latency_status (private)=====
        Update the latency status based on current values.
        
        This determines the severity of any latency issues and updates
        the status color and threshold flags.
        """
        latency = self.last_calculated_latency
        threshold_exceeded = latency > self.max_acceptable_latency
        
        if latency <= self.LOW_LATENCY:
            status = 'excellent'
            color = (0, 255, 0)  # Green
        elif latency <= self.MEDIUM_LATENCY:
            status = 'good'
            color = (200, 255, 0)  # Yellow-green
        elif latency <= self.HIGH_LATENCY:
            status = 'warning'
            color = (255, 255, 0)  # Yellow
        elif latency <= self.CRITICAL_LATENCY:
            status = 'high'
            color = (255, 165, 0)  # Orange
        else:
            status = 'critical'
            color = (255, 0, 0)  # Red
            
        self.latency_status = {
            'status': status,
            'color': color,
            'value': latency,
            'threshold_exceeded': threshold_exceeded,
            'last_check': time.time()
        }
    
    def get_latency_status(self):
        """
        =====Method: get_latency_status=====
        Get the current latency status information.
        
        Returns:
            dict: Dictionary with status, color, value and threshold info
        """
        return self.latency_status
    
    def get_current_latency(self):
        """
        =====Method: get_current_latency=====
        Get the current signal latency.
        
        Returns:
            float: Current latency in seconds
        """
        return self.last_calculated_latency
    
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
    
    def should_throttle(self):
        """
        =====Method: should_throttle=====
        Determine if processing should be throttled due to high latency.
        
        Returns:
            bool: True if processing should be throttled
        """
        return self.latency_status['threshold_exceeded']    
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
        
        # Update latency status
        if hasattr(self, '_update_latency_status'):
            self._update_latency_status()
