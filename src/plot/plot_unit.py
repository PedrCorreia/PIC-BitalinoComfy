"""
PlotUnit - Main visualization class for the plot module.

This module provides the main PlotUnit class that serves as the entry point
for the visualization system, integrating all components.
"""

import pygame
import threading
import numpy as np
import torch
import queue
import time
import logging
import os
import atexit
from collections import deque

# Import components from restructured modules
from .constants import *
from .view import RawView, ProcessedView, TwinView, SettingsView
from .ui import Sidebar, StatusBar
from .event_handler import EventHandler
from .performance import LatencyMonitor, FPSCounter, PlotExtensions
from .controllers import ButtonController
from .utils import convert_to_numpy

# Import ViewMode enum
from .view_mode import ViewMode

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlotUnit')

class PlotUnit:
    """
    Main visualization class for real-time signal plotting.
    
    This class provides a singleton instance for visualizing signals
    from the PIC-2025 system. It handles window creation, event processing,
    data management, and rendering.
    
    Attributes:
        width (int): Width of the visualization window
        height (int): Height of the visualization window
        running (bool): Flag indicating if the visualization is running
        initialized (bool): Flag indicating if PyGame is initialized
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of PlotUnit.
        
        Returns:
            PlotUnit: The singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """
        Initialize the PlotUnit instance.
        
        Note: Use get_instance() to access the singleton instance.
        """
        # Ensure this is only initialized once
        if PlotUnit._instance is not None:
            raise RuntimeError("Use PlotUnit.get_instance() to get the singleton instance")
        
        # Window properties
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.sidebar_width = SIDEBAR_WIDTH
        self.status_bar_height = STATUS_BAR_HEIGHT
        self.plot_width = self.width - self.sidebar_width
        
        # UI State
        self.current_mode = ViewMode.RAW
        self.running = False
        self.initialized = False
        
        # Thread-safe communication
        self.event_queue = queue.Queue()
        self.data_lock = threading.Lock()
        self.data = {}
        
        # Settings
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Performance monitoring
        self.start_time = time.time()
        self.latency_monitor = LatencyMonitor()
        self.fps_counter = FPSCounter()
        
        # Setup thread and window
        self.thread = None
        self.surface = None
        self.font = None
        self.icon_font = None
        
        # UI Components
        self.sidebar = None
        self.status_bar = None
        self.event_handler = None
        self.button_controller = None
        self.views = {}
        
        # Extensions
        self.extensions = None
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def start(self):
        """
        Start the visualization in a separate thread.
        
        This method launches the visualization window in a background thread
        if it's not already running.
        """
        with self._lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_visualization, daemon=True)
                self.thread.start()
                logger.info("PlotUnit visualization started")
    
    def cleanup(self):
        """
        Clean up resources when application exits.
        """
        logger.info("Cleaning up PlotUnit resources")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        pygame.quit()
    
    def _run_visualization(self):
        """
        Main visualization loop running in separate thread.
        
        This method initializes PyGame, sets up the window, and runs
        the main event loop for visualization.
        """
        try:
            logger.info("Starting visualization thread")
            os.environ['SDL_VIDEODRIVER'] = 'windib'  # For Windows compatibility
            pygame.init()
            pygame.display.set_caption("ComfyUI - PlotUnit")
            
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont(DEFAULT_FONT, DEFAULT_FONT_SIZE)
            self.icon_font = pygame.font.SysFont(DEFAULT_FONT, ICON_FONT_SIZE)
            
            # Initialize UI components
            self._initialize_components()
            
            self.initialized = True
            logger.info("PlotUnit window initialized successfully")
            clock = pygame.time.Clock()
            
            while self.running:
                # Process events
                self.running = self.event_handler.process_events()
                
                # Update current mode from event handler
                self.current_mode = self.event_handler.get_current_mode()
                
                # Handle incoming messages
                self._process_queue()
                
                # Render the interface
                self._render()
                
                # Update FPS counter
                self.fps_counter.update()
                
                # Update the display
                pygame.display.flip()
                
                # Cap the frame rate if enabled
                if self.settings['caps_enabled']:
                    clock.tick(TARGET_FPS)
                
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
        finally:
            pygame.quit()
            self.initialized = False
            self.running = False
            logger.info("Visualization thread terminated")
    
    def _initialize_components(self):
        """
        Initialize UI components and views.
        """
        # Create UI components
        self.sidebar = Sidebar(
            self.surface, 
            self.sidebar_width, 
            self.height, 
            self.font, 
            self.icon_font,
            self.current_mode
        )
        
        self.status_bar = StatusBar(
            self.surface,
            self.width,
            self.status_bar_height,
            self.font,
            self.start_time
        )
        
        # Create views
        self.views = {
            ViewMode.RAW: RawView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font
            ),
            ViewMode.PROCESSED: ProcessedView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font
            ),
            ViewMode.TWIN: TwinView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font
            ),
            ViewMode.SETTINGS: SettingsView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font,
                self.settings
            ),
        }
          # Create button controller first
        self.button_controller = ButtonController(self)
        
        # Create event handler with button controller
        self.event_handler = EventHandler(
            self.sidebar,
            self.views[ViewMode.SETTINGS],
            self.button_controller
        )
        
        # Create extensions
        self.extensions = PlotExtensions(self)
    
    def _process_queue(self):
        """
        Process events from the queue.
        
        This method handles events from the queue, including data updates
        and control messages.
        """
        # Process all events in the queue
        try:
            while True:
                event = self.event_queue.get_nowait()
                
                if "type" in event:
                    event_type = event["type"]
                    
                    if event_type == "data":
                        # Handle data update
                        data_type = event.get("data_type", "raw")
                        data = event.get("data")
                        
                        if data is not None:
                            self._update_data(data_type, data)
                            
                    elif event_type == "command":
                        # Handle command
                        command = event.get("command")
                        if command == "clear":
                            self.clear_plots()
                
                self.event_queue.task_done()
                
        except queue.Empty:
            pass
    
    def _update_data(self, data_type, data):
        """
        Update data for visualization.
        
        Args:
            data_type (str): Type/ID of the data
            data: The data to update
        """
        # Convert to numpy if needed
        np_data = convert_to_numpy(data)
        
        # Update the data
        with self.data_lock:
            self.data[data_type] = np_data
            
            # Track when this signal was last updated
            self.latency_monitor.update_signal_time(data_type)
    
    def _render(self):
        """
        Render the visualization interface.
        
        This method clears the screen and renders the appropriate view
        based on the current mode.
        """
        # Clear the screen
        self.surface.fill(BACKGROUND_COLOR)
        
        # Draw the sidebar
        self.sidebar.draw()
        
        # Draw the appropriate view based on mode
        if self.current_mode in self.views:
            self.views[self.current_mode].draw()
        
        # Draw buttons
        if self.button_controller:
            self.button_controller.draw(self.surface)
            
        # Draw additional performance metrics from extensions
        if self.extensions and self.settings.get('show_extended_metrics', True):
            y_pos = 40  # Below the buttons
            y_pos = self.extensions.draw_performance_metrics(self.surface, self.font, 10, y_pos)
        
        # Draw status bar with performance metrics
        self.status_bar.draw(
            self.fps_counter.get_fps(),
            self.latency_monitor.get_current_latency(),
            self.latency_monitor.get_signal_times()
        )
    
    def queue_data(self, data_type, data):
        """
        Queue data for visualization.
        
        This method allows external code to send data to the visualization.
        
        Args:
            data_type (str): Type/ID of the data
            data: The data to visualize
        """
        if not self.running:
            self.start()
            
        event = {
            "type": "data",
            "data_type": data_type,
            "data": data
        }
        self.event_queue.put(event)
    
    def queue_command(self, command):
        """
        Queue a command for the visualization.
        
        Args:
            command (str): Command to execute
        """
        event = {
            "type": "command",
            "command": command
        }
        self.event_queue.put(event)
    
    def clear_plots(self):
        """
        Clear all plot data.
        """
        with self.data_lock:
            self.data.clear()
            self.latency_monitor.clear()
        
        logger.info("Cleared all plot data")
    
    def is_running(self):
        """
        Check if the visualization is running.
        
        Returns:
            bool: True if the visualization is running, False otherwise
        """
        return self.running and self.initialized
