"""
=====Overview=====
FPSCounter module for the PlotUnit visualization system.

This module provides FPS (frames per second) tracking functionality
for the PlotUnit system, helping monitor visualization performance.

=====Usage=====
Create an instance of FPSCounter and call update() each time a frame is rendered.
Access current FPS with get_fps(), average frame time with get_frame_time(),
and time since the last frame with get_time_since_last_frame().

Example:
    fps_counter = FPSCounter()
    while rendering:
        fps = fps_counter.update()
        print(f"Current FPS: {fps}")

=====Classes=====
- FPSCounter: Tracks frame rendering times and calculates FPS metrics.

=====API Reference=====
"""

import time
from collections import deque

class FPSCounter:
    """
    =====Class: FPSCounter=====
    FPS counter for tracking visualization performance.
    
    This class tracks frame rendering times and calculates FPS
    metrics to monitor visualization performance.
    
    Attributes:
        frame_times (deque): Queue of recent frame timestamps
        last_frame_time (float): Timestamp of the most recently rendered frame
        current_fps (float): Current calculated FPS
    """
    
    def __init__(self, history_size=60):
        """
        =====Method: __init__=====
        Initialize the FPS counter.
        
        Args:
            history_size (int, optional): Size of the frame history queue
        """
        self.frame_times = deque(maxlen=history_size)
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        
    def update(self):
        """
        =====Method: update=====
        Record a new frame and update FPS calculation.
        
        Returns:
            float: The current FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Calculate FPS if we have enough history
        if len(self.frame_times) > 1:
            # Calculate time elapsed for the frames in our history
            elapsed = self.frame_times[-1] - self.frame_times[0]
            
            if elapsed > 0:
                # Calculate FPS as (number of frames - 1) / elapsed time
                self.current_fps = (len(self.frame_times) - 1) / elapsed
            
        self.last_frame_time = current_time
        return self.current_fps
    
    def get_fps(self):
        """
        =====Method: get_fps=====
        Get the current frames per second.
        
        Returns:
            float: Current FPS
        """
        return self.current_fps
    
    def get_frame_time(self):
        """
        =====Method: get_frame_time=====
        Get the average time per frame.
        
        Returns:
            float: Average time per frame in milliseconds
        """
        if self.current_fps <= 0:
            return 0
            
        return 1000.0 / self.current_fps
    
    def get_time_since_last_frame(self):
        """
        =====Method: get_time_since_last_frame=====
        Get time elapsed since the last frame was rendered.
        
        Returns:
            float: Time in seconds since last frame
        """
        return time.time() - self.last_frame_time
