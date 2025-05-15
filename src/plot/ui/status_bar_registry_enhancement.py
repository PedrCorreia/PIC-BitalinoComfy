#!/usr/bin/env python
"""
Status Bar Enhancement for Registry Integration

This module enhances the StatusBar class with additional registry signal information.
"""

import pygame
import time
from src.plot.ui.status_bar import StatusBar

# Store the original draw method
original_draw = StatusBar.draw

def enhanced_draw(self, fps, latency, signal_times=None):
    """
    Enhanced version of the status bar draw method with better registry signal info.
    
    Args:
        fps (float): Current frames per second
        latency (float): Current latency in seconds
        signal_times (dict): Dictionary mapping signal IDs to last update times
    """
    # Draw the basic status bar using the original method
    if signal_times is None:
        signal_times = {}
        
    # Count active signals
    signal_count = len(signal_times)
    
    # Calculate update status
    if signal_count > 0:
        latest_time = max(signal_times.values()) if signal_times else time.time()
        time_since_update = time.time() - latest_time
        
        if time_since_update < 1.0:
            update_text = "Just now"
            update_color = (0, 220, 0)  # Bright green
        elif time_since_update < 5.0:
            update_text = f"{time_since_update:.1f}s ago"
            update_color = (0, 220, 0)  # Green
        elif time_since_update < 30.0:
            update_text = f"{time_since_update:.1f}s ago"
            update_color = (220, 220, 0)  # Yellow
        else:
            update_text = f"{int(time_since_update)}s ago"
            update_color = (220, 0, 0)  # Red
    else:
        update_text = "No updates"
        update_color = (150, 150, 150)  # Gray
    
    # Pass the signal count and update text to the original method
    runtime = ""
    if hasattr(self, 'start_time') and self.start_time:
        elapsed = time.time() - self.start_time
        runtime = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
          # Call the original method with appropriate parameters
    try:
        # First try with enhanced parameters
        original_draw(self, fps, latency, signal_times)
    except TypeError:
        # Fallback to original method's signature
        original_draw(self, fps, latency)

# Apply the enhancement
try:
    # Replace the method with our enhanced version
    StatusBar.draw = enhanced_draw
    print("Successfully enhanced StatusBar for registry signals")
except Exception as e:
    print(f"Warning: Could not enhance StatusBar for registry signals: {e}")
