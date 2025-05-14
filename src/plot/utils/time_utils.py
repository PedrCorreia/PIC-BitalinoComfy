"""
Time utilities for the PlotUnit visualization system.

This module provides time-related utilities for the PlotUnit system,
handling timing operations and formatting.
"""

import time
from datetime import datetime

def get_timestamp():
    """
    Get the current timestamp.
    
    Returns:
        float: Current timestamp in seconds
    """
    return time.time()

def format_time(seconds):
    """
    Format a time duration in seconds to a human-readable string.
    
    Args:
        seconds (float): Time duration in seconds
        
    Returns:
        str: Formatted time string (e.g., "MM:SS")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def format_time_with_ms(seconds):
    """
    Format a time duration in seconds to a human-readable string with milliseconds.
    
    Args:
        seconds (float): Time duration in seconds
        
    Returns:
        str: Formatted time string (e.g., "MM:SS.mmm")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"

def format_elapsed_time(start_time):
    """
    Format the elapsed time since a start time.
    
    Args:
        start_time (float): Start timestamp in seconds
        
    Returns:
        str: Formatted elapsed time string
    """
    return format_time(time.time() - start_time)

def timestamp_to_str(timestamp):
    """
    Convert a timestamp to a human-readable string.
    
    Args:
        timestamp (float): Timestamp in seconds
        
    Returns:
        str: Human-readable timestamp string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%H:%M:%S')

def get_time_since(timestamp):
    """
    Get the time elapsed since a timestamp.
    
    Args:
        timestamp (float): Reference timestamp in seconds
        
    Returns:
        float: Time elapsed in seconds
    """
    return time.time() - timestamp

def format_time_since(timestamp):
    """
    Format the time elapsed since a timestamp as a human-readable string.
    
    Args:
        timestamp (float): Reference timestamp in seconds
        
    Returns:
        str: Human-readable elapsed time string
    """
    elapsed = get_time_since(timestamp)
    
    if elapsed < 1.0:
        return "Just now"
    elif elapsed < 60.0:
        return f"{elapsed:.1f} seconds ago"
    elif elapsed < 3600.0:
        minutes = int(elapsed // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        hours = int(elapsed // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
