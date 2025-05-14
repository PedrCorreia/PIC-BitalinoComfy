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
from . import constants
from .view import RawView, ProcessedView, TwinView, StackedView, SettingsView
from .ui import Sidebar, StatusBar
from .ui.plot_container import PlotContainer
from .event_handler import EventHandler
from .performance import LatencyMonitor, FPSCounter, PlotExtensions
from .utils import convert_to_numpy
from .utils.signal_generator import generate_test_signals, update_test_signals

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
            ViewMode.STACKED: StackedView(
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
          # --- PlotContainer integration ---
        self.plot_container = PlotContainer(
            pygame.Rect(0, 0, self.width, self.height),
            self.sidebar_width,
            margin=PLOT_PADDING
        )
        
        # Initialize PlotContainer with the view for current mode
        # We'll update the plots list dynamically in the _render method
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
        based on the current mode.
        """
        # Clear the screen
        self.surface.fill(BACKGROUND_COLOR)
        
        # Draw the sidebar
        self.sidebar.draw()
        
        # Handle different view modes
        if self.current_mode == ViewMode.RAW:
            # For raw view, only show the raw view
            # Update PlotContainer settings and window dimensions
            self.plot_container.set_twin_view(False)
            self.plot_container.set_window_rect(
                pygame.Rect(0, 0, self.width, self.height),
                self.sidebar_width
            )
            # Reset the plots list and add only the raw view
            self.plot_container.plots = [self.views[ViewMode.RAW]]
            self.plot_container.update_layout()
            self.plot_container.draw()
        elif self.current_mode == ViewMode.PROCESSED:
            # For processed view, only show the processed view
            # Update PlotContainer settings and window dimensions
            self.plot_container.set_twin_view(False)
            self.plot_container.set_window_rect(
                pygame.Rect(0, 0, self.width, self.height),
                self.sidebar_width
            )
            # Reset the plots list and add only the processed view
            self.plot_container.plots = [self.views[ViewMode.PROCESSED]]
            self.plot_container.update_layout()
            self.plot_container.draw()
        elif self.current_mode == ViewMode.TWIN:
            # For twin view mode, use the PlotContainer with twin view enabled
            self.plot_container.set_twin_view(True)
            self.plot_container.set_window_rect(
                pygame.Rect(0, 0, self.width, self.height),
                self.sidebar_width
            )
            # Reset the plots list and add both raw and processed views for side-by-side comparison
            self.plot_container.plots = [self.views[ViewMode.TWIN]]
            self.plot_container.update_layout()
            self.plot_container.draw()
        elif self.current_mode == ViewMode.STACKED:
            # For stacked view mode, use the PlotContainer with stacked view enabled
            self.plot_container.set_twin_view(False)
            self.plot_container.set_window_rect(
                pygame.Rect(0, 0, self.width, self.height),
                self.sidebar_width
            )
            # Reset the plots list and add the stacked view
            self.plot_container.plots = [self.views[ViewMode.STACKED]]
            self.plot_container.update_layout()
            self.plot_container.draw()
        elif self.current_mode == ViewMode.SETTINGS:
            # For settings view, draw directly
            self.views[ViewMode.SETTINGS].draw()
        else:
            # Unknown mode - fall back to raw view
            self.views[ViewMode.RAW].draw()
        
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
    
    def load_test_signals(self, clear_existing=True):
        """
        Load test signals for demonstration.
        
        This method populates the data dictionary with test signals
        for demonstration and development purposes.
        
        Args:
            clear_existing (bool): Whether to clear existing signals before loading test signals
        """
        # Generate test signals
        test_signals = generate_test_signals()
        
        with self.data_lock:
            # Clear existing data if requested
            if clear_existing:
                self.data.clear()
                self.latency_monitor.clear()
            
            # Add raw signals
            self.data["raw_sine"] = test_signals["raw_sine"]
            self.data["raw_square"] = test_signals["raw_square"]
            
            # Add processed signals with "_processed" suffix for proper categorization
            self.data["inverted_square_processed"] = test_signals["inverted_square"]
            self.data["sawtooth_processed"] = test_signals["sawtooth"]
            self.data["triangle_processed"] = test_signals["triangle"]
            
        logger.info("Test signals loaded successfully")
        
        # If not already running, start the visualization
        if not self.running:
            self.start()
    
    def update_test_signals(self):
        """
        Update the test signals with new dynamic data.
        
        This method is useful for demos to show changing signals.
        """
        with self.data_lock:
            if not self.data:
                # If no data exists, load initial test signals
                self.load_test_signals()
                return
            
            # Update existing test signals
            update_test_signals(self.data)
            
            # Update latency timestamps
            for signal_id in self.data:
                self.latency_monitor.update_signal_time(signal_id)
    
    def configure_layout(self, grid_enabled=True, stack_mode='aligned', normalize_stacks=True):
        """
        Configure layout settings for plot visualization.
        
        This method configures how plots are displayed, particularly for stacked views.
        
        Args:
            grid_enabled (bool): Whether to enable grid lines on plots
            stack_mode (str): Mode for stacking ('aligned' or 'free')
            normalize_stacks (bool): Whether to normalize signal scales across stacked plots
        """
        if not isinstance(self.views.get(ViewMode.STACKED), StackedView):
            logger.warning("StackedView not initialized, cannot configure layout")
            return
            
        stacked_view = self.views[ViewMode.STACKED]
        stacked_view.set_grid_enabled(grid_enabled)
        stacked_view.set_stack_mode(stack_mode)
        stacked_view.set_normalize_stacks(normalize_stacks)
        
        logger.info(f"Layout configured: grid={grid_enabled}, mode={stack_mode}, normalize={normalize_stacks}")
    
    def set_signal_params(self, signal_id, color=None, label=None, stack_group=None):
        """
        Set visualization parameters for a specific signal.
        
        Args:
            signal_id (str): ID of the signal to configure
            color (tuple): RGB color tuple for the signal
            label (str): Label to display for the signal
            stack_group (str): Group for aligning signals in stacks
        """
        if not isinstance(self.views.get(ViewMode.STACKED), StackedView):
            logger.warning("StackedView not initialized, cannot set signal parameters")
            return
            
        stacked_view = self.views[ViewMode.STACKED]
        stacked_view.set_signal_params(signal_id, color, label, stack_group)
        
        logger.info(f"Signal parameters set for {signal_id}")
    
    def enable_differential_view(self, base_signal_id, compare_signal_id, label=None, stack_position=None):
        """
        Enable a differential view between two signals.
        
        Args:
            base_signal_id (str): ID of the base signal
            compare_signal_id (str): ID of the signal to compare against
            label (str): Label for the differential view
            stack_position (int): Position in the stack (0-based)
        """
        if not isinstance(self.views.get(ViewMode.STACKED), StackedView):
            logger.warning("StackedView not initialized, cannot enable differential view")
            return
            
        stacked_view = self.views[ViewMode.STACKED]
        stacked_view.enable_differential_view(
            base_signal_id, compare_signal_id, label, stack_position
        )
        
        logger.info(f"Differential view enabled: {base_signal_id} vs {compare_signal_id}")
    
    def synchronize_grid_scales(self):
        """
        Force synchronization of grid scales across stacked plots.
        """
        if not isinstance(self.views.get(ViewMode.STACKED), StackedView):
            logger.warning("StackedView not initialized, cannot synchronize grid scales")
            return
            
        stacked_view = self.views[ViewMode.STACKED]
        stacked_view.synchronize_grid_scales()
        
        logger.info("Grid scales synchronized")
