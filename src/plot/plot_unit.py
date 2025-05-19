"""
PlotUnit - Main visualization class for the plot module.

This module provides the main PlotUnit class that serves as the entry point
for the visualization system, integrating all components.
"""

import pygame
import threading
import queue
import time
import logging
import os
import atexit

# Import components from restructured modules
from . import constants
from .view import RawView, ProcessedView, TwinView,  SettingsView
from .ui import Sidebar, StatusBar
from .ui.plot_container import PlotContainer
from .utils.event_handler import EventHandler
from .performance import EnhancedLatencyMonitor, LatencyMonitor, FPSCounter, PlotExtensions
from .utils.signal_generator import generate_test_signals, update_test_signals

# Import ViewMode enum
from .view_mode import ViewMode

# Import constants
from .constants import (WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, STATUS_BAR_HEIGHT, 
                          DEFAULT_SETTINGS, PLOT_PADDING, DEFAULT_FONT, DEFAULT_FONT_SIZE, 
                          ICON_FONT_SIZE, TARGET_FPS, BACKGROUND_COLOR)

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
        self.status_bar_height = STATUS_BAR_HEIGHT        # Update plot_width for just the left sidebar (no right sidebar anymore)
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
        self.latency_monitor = EnhancedLatencyMonitor()
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
            self._initialize_components()
            self.initialized = True
            logger.info("PlotUnit window initialized successfully")
            clock = pygame.time.Clock()
            
            # Main visualization loop
            while self.running:
                # Process events
                if self.event_handler is not None:
                    self.running = self.event_handler.process_events()
                else:
                    logger.error("Event handler not initialized; terminating visualization loop.")
                    break
                # Update current mode from event handler
                if hasattr(self.event_handler, 'get_current_mode'):
                    self.current_mode = self.event_handler.get_current_mode()
                elif hasattr(self.event_handler, 'current_mode'):
                    self.current_mode = self.event_handler.current_mode
                
                # Handle incoming messages
                self._process_queue()
                
                # Get current latency status
                latency_status = self.latency_monitor.get_latency_status() if hasattr(self.latency_monitor, 'get_latency_status') else None
                
                # Check if we need to throttle rendering due to high latency
                should_throttle = latency_status and latency_status.get('threshold_exceeded', False)
                
                # Render the interface (unless we're throttling due to high latency)
                if not should_throttle:
                    self._render()
                    # Update the display
                    if self.surface is not None:
                        pygame.display.flip()
                else:
                    # Reduced rendering during high latency - just basic UI
                    if self.surface is not None:
                        self.surface.fill(BACKGROUND_COLOR)
                    if self.sidebar is not None:
                        self.sidebar.draw()
                    if self.status_bar is not None:
                        self.status_bar.draw(
                            self.fps_counter.get_fps(), 
                            self.latency_monitor.get_current_latency(),
                            self.latency_monitor.get_signal_times()
                        )
                    if self.surface is not None:
                        pygame.display.flip()
                    logger.warning(f"High latency detected: {latency_status.get('value', 0):.3f}s - throttling rendering" if latency_status is not None else "High latency detected: unknown value - throttling rendering")
                
                # Update FPS counter
                self.fps_counter.update()
                
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
            self.current_mode,
            self.settings
        )
        
        self.status_bar = StatusBar(
            self.surface,
            self.width,
            self.status_bar_height,
            self.font
        )
        
        # Create views
        self.views = {
            ViewMode.RAW: RawView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font,
                
            ),
            ViewMode.PROCESSED: ProcessedView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font,
                
            ),
            ViewMode.TWIN: TwinView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font,
                
            ),
            ViewMode.SETTINGS: SettingsView(
                self.surface, 
                self.data_lock, 
                self.data, 
                self.font,
                self.settings
            ),
        }
    
        # --- PlotContainer integration ---
        self.plot_container = PlotContainer(
            pygame.Rect(0, 0, self.width, self.height),
            self.sidebar_width,
            margin=PLOT_PADDING
        )
        
        # Initialize PlotContainer with the view for current mode
        if self.current_mode == ViewMode.RAW:
            self.plot_container.add_plot(self.views[ViewMode.RAW])
        elif self.current_mode == ViewMode.PROCESSED:
            self.plot_container.add_plot(self.views[ViewMode.PROCESSED])
        elif self.current_mode == ViewMode.TWIN:
            self.plot_container.add_plot(self.views[ViewMode.TWIN])
            self.plot_container.set_twin_view(True)
        elif self.current_mode == ViewMode.STACKED:
            self.plot_container.add_plot(self.views[ViewMode.STACKED])
            self.plot_container.set_twin_view(False)
        
        # Create event handler
        self.event_handler = EventHandler(
            self.sidebar,
            self.views[ViewMode.SETTINGS],
            None  # Previously used for button_controller, now always None
        )
        
        # Create extensions
        self.extensions = PlotExtensions(self)
        
        # Add debug plots for UI testing
        

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
        Update the data dictionary with new data.
        
        Args:
            data_type (str): Type/ID of the data
            data: The data to visualize
        """
        timestamp = time.time()
        
        with self.data_lock:
            # Initialize the data type if it doesn't exist
            if data_type not in self.data:
                self.data[data_type] = {
                    'values': [],
                    'timestamps': []
                }
            
            # Add the new data
            self.data[data_type]['values'].append(data)
            self.data[data_type]['timestamps'].append(timestamp)
            
            # Keep data within limits if needed
            max_points = self.settings.get('max_data_points', 1000)
            if len(self.data[data_type]['values']) > max_points:
                self.data[data_type]['values'] = self.data[data_type]['values'][-max_points:]
                self.data[data_type]['timestamps'] = self.data[data_type]['timestamps'][-max_points:]
        
        # Update latency monitoring
        self.latency_monitor.record_signal(data_type, timestamp)
    
    def _set_mode(self, mode):
        """
        Set the current visualization mode.
        
        Args:
            mode (ViewMode): The visualization mode to set
        """
        if not isinstance(mode, ViewMode):
            logger.error(f"Invalid mode: {mode}")
            return
            
        self.current_mode = mode
        if hasattr(self, 'sidebar') and self.sidebar:
            self.sidebar.current_mode = mode
            
        # Update the event handler if available
        if hasattr(self, 'event_handler') and self.event_handler:
            self.event_handler.current_mode = mode
        
        # Update the plot container layout for the new mode
        if hasattr(self, 'plot_container') and self.plot_container:
            if mode == ViewMode.RAW:
                self.plot_container.set_twin_view(False)
                self.plot_container.plots = [self.views[ViewMode.RAW]]
            elif mode == ViewMode.PROCESSED:
                self.plot_container.set_twin_view(False)
                self.plot_container.plots = [self.views[ViewMode.PROCESSED]]
            elif mode == ViewMode.TWIN:
                self.plot_container.set_twin_view(True)
                self.plot_container.plots = [self.views[ViewMode.TWIN]]
            elif mode == ViewMode.STACKED:
                self.plot_container.set_twin_view(False)
                self.plot_container.plots = [self.views[ViewMode.STACKED]]
            
            self.plot_container.update_layout()
        
        logger.info(f"Mode set to {mode.name}")
    
    def _render(self):
        """
        Render the visualization interface.
        This method clears the screen and renders the appropriate view
        based on the current mode. All UI components are checked for initialization.
        """
        # Clear the screen
        if self.surface is not None:
            self.surface.fill(BACKGROUND_COLOR)
        # Draw the sidebar
        if self.sidebar is not None:
            self.sidebar.draw()
        # Handle different view modes
        if self.views and self.current_mode in self.views:
            # Use PlotContainer for all modes except SETTINGS
            if self.current_mode == ViewMode.SETTINGS:
                if self.views[ViewMode.SETTINGS] is not None:
                    self.views[ViewMode.SETTINGS].draw()
            else:
                if self.plot_container is not None:
                    # Update PlotContainer settings and window dimensions
                    if self.current_mode == ViewMode.RAW:
                        self.plot_container.set_twin_view(False)
                        self.plot_container.set_window_rect(
                            pygame.Rect(0, 0, self.width, self.height),
                            self.sidebar_width
                        )
                        self.plot_container.plots = [self.views[ViewMode.RAW]]
                    elif self.current_mode == ViewMode.PROCESSED:
                        self.plot_container.set_twin_view(False)
                        self.plot_container.set_window_rect(
                            pygame.Rect(0, 0, self.width, self.height),
                            self.sidebar_width
                        )
                        self.plot_container.plots = [self.views[ViewMode.PROCESSED]]
                    elif self.current_mode == ViewMode.TWIN:
                        self.plot_container.set_twin_view(True)
                        self.plot_container.set_window_rect(
                            pygame.Rect(0, 0, self.width, self.height),
                            self.sidebar_width
                        )
                        self.plot_container.plots = [self.views[ViewMode.TWIN]]
                    elif self.current_mode == ViewMode.STACKED:
                        self.plot_container.set_twin_view(False)
                        self.plot_container.set_window_rect(
                            pygame.Rect(0, 0, self.width, self.height),
                            self.sidebar_width
                        )
                        self.plot_container.plots = [self.views[ViewMode.STACKED]]
                    self.plot_container.update_layout()
                    self.plot_container.draw()
        else:
            # Unknown mode or missing view - fallback to raw view if available
            if self.views and ViewMode.RAW in self.views and self.views[ViewMode.RAW] is not None:
                self.views[ViewMode.RAW].draw()
        # Draw status bar with performance metrics
        if self.status_bar is not None:
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

    # --- Only keep minimal required interface for registry-driven visualization ---
    # All legacy/test/configuration/demo code has been removed. Only core methods remain.
