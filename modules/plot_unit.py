"""
PlotUnit main class for visualization system

This module provides the main PlotUnit class for the visualization system.
"""

import time
import pygame
import threading
import traceback

# Import local modules
from modules.constants import *
from modules.view_mode import ViewMode
from modules.ui_components import BaseView, SettingsView, Sidebar, StatusBar
from modules.event_handler import EventHandler

class PlotUnit:
    """Main class for the PlotUnit visualization system."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the PlotUnit singleton instance.
        
        Returns:
            The PlotUnit singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the PlotUnit instance."""
        # Initialize basic attributes
        self.running = False
        self.initialized = False
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Set up threading-related attributes
        self.data_lock = threading.Lock()
        self.data = {}
        
        # Set up main thread
        self.thread = None
        
        # Store last frame time for FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0
        
        print("PlotUnit initialized")
    
    def start(self):
        """Start the visualization thread."""
        if self.running:
            print("PlotUnit already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        print("PlotUnit thread started")
    
    def _run(self):
        """Main visualization loop."""
        try:
            # Initialize pygame
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("ComfyUI - PlotUnit")
            
            # Create font for rendering text
            self.font = pygame.font.SysFont("Arial", FONT_SIZE)
            
            # Initialize UI components
            self.sidebar = Sidebar(self.surface, self.font)
            self.status_bar = StatusBar(self.surface, self.font)
            
            # Create views
            self.views = {
                ViewMode.RAW: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.PROCESSED: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.TWIN: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.SETTINGS: SettingsView(self.surface, self.data_lock, self.data, self.font, self.settings)
            }
            
            # Create event handler
            self.event_handler = EventHandler(self.sidebar, self.views[ViewMode.SETTINGS])
            
            # Set as initialized
            self.initialized = True
            print("PlotUnit initialized successfully")
            
            # Main loop
            while self.running:
                # Process events
                self.running = self.event_handler.process_events()
                
                # Clear the screen
                self.surface.fill(BACKGROUND_COLOR)
                
                # Get current mode
                current_mode = self.event_handler.get_current_mode()
                
                # Draw current view
                self.views[current_mode].draw()
                
                # Draw sidebar and status bar
                self.sidebar.draw()
                self.status_bar.draw(
                    self.fps, 
                    self.settings['connected_nodes'],
                    runtime="00:04", 
                    latency="0.0 ms", 
                    signals=3, 
                    last_update="Just now"
                )
                
                # Update the display
                pygame.display.flip()
                
                # Calculate FPS
                current_time = time.time()
                delta_time = current_time - self.last_frame_time
                self.fps = 1.0 / delta_time if delta_time > 0 else 0
                self.last_frame_time = current_time
                
                # Cap FPS to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            print(f"Error in visualization thread: {str(e)}")
            traceback.print_exc()
            self.running = False
    
    def load_test_signals(self):
        """Load test signals for visualization."""
        print("Test signals loaded (placeholder)")
    
    def _set_mode(self, mode):
        """Set the current view mode.
        
        Args:
            mode: The new view mode
        """
        if self.initialized and hasattr(self, 'sidebar'):
            self.sidebar.current_mode = mode
            if hasattr(self, 'event_handler'):
                self.event_handler.current_mode = mode
            print(f"View mode set to: {mode.name}")
    
    def clear_plots(self):
        """Clear all plots."""
        print("Plots cleared")
