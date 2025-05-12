import pygame
import threading
import numpy as np
import time
import queue
from enum import Enum
import logging
import os
import atexit

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlotUnit')

class ViewMode(Enum):
    RAW = 0
    FILTERED = 1
    SETTINGS = 2

class PlotUnit:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        # Ensure this is only initialized once
        if PlotUnit._instance is not None:
            raise RuntimeError("Use PlotUnit.get_instance() to get the singleton instance")
        
        # Window properties
        self.width = 300
        self.height = 512
        self.sidebar_width = 40
        self.plot_width = self.width - self.sidebar_width
        
        # Colors
        self.background_color = (16, 16, 16)  # Dark background
        self.sidebar_color = (24, 24, 24)     # Slightly lighter
        self.accent_color = (0, 120, 215)     # Blue accent
        self.text_color = (220, 220, 220)     # Light text
        self.grid_color = (40, 40, 40)        # Dark grid
        
        # UI State
        self.current_mode = ViewMode.RAW
        self.running = False
        self.initialized = False
        
        # Thread-safe communication
        self.event_queue = queue.Queue()
        self.data_lock = threading.Lock()
        self.data = {
            'raw': np.zeros(100),
            'filtered': np.zeros(100)
        }
        
        # Settings options
        self.settings = {
            'caps_enabled': True,
            'light_mode': False,
            'performance_mode': False,
            'connected_nodes': 0,  # This will be updated dynamically
            'reset_plots': False,   # New setting for reset plots button
            'reset_registry': False  # New setting for reset registry button
        }
        
        # Setup thread and window
        self.thread = None
        self.surface = None
        self.font = None
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def start(self):
        """Start the visualization in a separate thread if not already running"""
        with self._lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_visualization, daemon=True)
                self.thread.start()
                logger.info("PlotUnit visualization started")
    
    def cleanup(self):
        """Clean up resources when application exits"""
        logger.info("Cleaning up PlotUnit resources")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        pygame.quit()
    
    def _run_visualization(self):
        """Main visualization loop running in separate thread"""
        try:
            logger.info("Starting visualization thread")
            os.environ['SDL_VIDEODRIVER'] = 'windib'  # For Windows compatibility
            pygame.init()
            pygame.display.set_caption("ComfyUI - PlotUnit")
            
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont("Arial", 14)
            self.icon_font = pygame.font.SysFont("Arial", 24)
            
            self.initialized = True
            logger.info("PlotUnit window initialized successfully")
            clock = pygame.time.Clock()
            
            while self.running:
                # Process events
                self._process_events()
                
                # Handle incoming messages
                self._process_queue()
                
                # Render the interface
                self._render()
                
                # Update the display
                pygame.display.flip()
                clock.tick(30)  # Limit to 30 FPS
                
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)  # Include traceback
        finally:
            pygame.quit()
            self.initialized = False
            self.running = False
            logger.info("Visualization thread terminated")
    
    def _process_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    x, y = event.pos
                    if x < self.sidebar_width:
                        # Clicked in sidebar area
                        button_height = 40
                        
                        # Determine which button was clicked
                        if y < button_height:
                            self.current_mode = ViewMode.RAW
                        elif y < button_height * 2:
                            self.current_mode = ViewMode.FILTERED
                        elif y < button_height * 3:
                            self.current_mode = ViewMode.SETTINGS
                    elif self.current_mode == ViewMode.SETTINGS:
                        # Check if a settings button was clicked
                        self._handle_settings_click(x, y)
    
    def _handle_settings_click(self, x, y):
        """Handle clicks on settings buttons"""
        # For each button in our settings view
        for button_rect, setting_key in getattr(self, 'settings_buttons', []):
            if button_rect.collidepoint(x, y):
                # Special handling for reset buttons
                if setting_key == 'reset_plots':
                    logger.info("Reset plots button clicked")
                    self.clear_plots()
                    return
                elif setting_key == 'reset_registry':
                    logger.info("Reset registry button clicked")
                    self._reset_signal_registry()
                    return
                
                # Normal toggle for other settings
                self.settings[setting_key] = not self.settings[setting_key]
                
                # Apply setting changes
                self._apply_setting_changes(setting_key)
                
                logger.info(f"Setting '{setting_key}' changed to {self.settings[setting_key]}")
                break
    
    def _reset_signal_registry(self):
        """Reset the signal registry"""
        try:
            # Import here to avoid circular imports
            import sys
            import importlib
            
            # Try to import the signal registry
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            try:
                from comfy.mock_signal_node import SignalRegistry
                SignalRegistry.reset()
                logger.info("Signal registry reset successfully")
            except ImportError:
                # Alternative import path
                try:
                    mock_signal = importlib.import_module('..comfy.mock_signal_node', package=__name__)
                    mock_signal.SignalRegistry.reset()
                    logger.info("Signal registry reset successfully (alternative import)")
                except Exception as e:
                    logger.error(f"Could not import SignalRegistry (alt): {str(e)}")
        except Exception as e:
            logger.error(f"Error resetting signal registry: {str(e)}")
    
    def _apply_setting_changes(self, setting_key):
        """Apply changes when settings are modified"""
        if setting_key == 'light_mode':
            if self.settings['light_mode']:
                # Light mode colors
                self.background_color = (240, 240, 240)
                self.sidebar_color = (200, 200, 200)
                self.text_color = (20, 20, 20)
                self.grid_color = (180, 180, 180)
            else:
                # Dark mode colors
                self.background_color = (16, 16, 16)
                self.sidebar_color = (24, 24, 24)
                self.text_color = (220, 220, 220)
                self.grid_color = (40, 40, 40)
        elif setting_key == 'performance_mode':
            # Adjust performance settings
            pass
    
    def _process_queue(self):
        """Process incoming messages from the main thread"""
        try:
            while True:
                message = self.event_queue.get_nowait()
                if message['type'] == 'update_data':
                    with self.data_lock:
                        data_type = message['data_type']
                        self.data[data_type] = message['data']
                        logger.info(f"Updated {data_type} data: shape={np.shape(message['data'])}")
                self.event_queue.task_done()
        except queue.Empty:
            pass
    
    def _render(self):
        """Render the visualization interface"""
        # Clear the screen
        self.surface.fill(self.background_color)
        
        # Draw sidebar
        pygame.draw.rect(self.surface, self.sidebar_color, 
                         (0, 0, self.sidebar_width, self.height))
        
        # Draw sidebar buttons
        self._draw_sidebar_buttons()
        
        # Draw visualization area based on current mode
        if self.current_mode == ViewMode.RAW:
            self._draw_raw_view()
        elif self.current_mode == ViewMode.FILTERED:
            self._draw_filtered_view()
        elif self.current_mode == ViewMode.SETTINGS:
            self._draw_settings_view()
    
    def _draw_sidebar_buttons(self):
        """Draw the sidebar navigation buttons"""
        button_height = 40
        button_icons = ["R", "P", "S"]  # Raw, Filtered, Settings
        
        for i, icon in enumerate(button_icons):
            button_y = i * button_height
            
            # Highlight active button
            if self.current_mode.value == i:
                pygame.draw.rect(self.surface, self.accent_color, 
                                 (0, button_y, self.sidebar_width, button_height))
            
            # Draw button text
            text = self.icon_font.render(icon, True, self.text_color)
            text_rect = text.get_rect(center=(self.sidebar_width // 2, button_y + button_height // 2))
            self.surface.blit(text, text_rect)
    
    def _draw_raw_view(self):
        """Draw the raw signal visualization"""
        with self.data_lock:
            data = np.copy(self.data['raw'])
        
        self._draw_plot(data, "Raw Signal", (220, 180, 0))
    
    def _draw_filtered_view(self):
        """Draw the filtered signal visualization"""
        with self.data_lock:
            data = np.copy(self.data['filtered'])
        
        self._draw_plot(data, "Filtered Signal", (0, 180, 220))
    
    def _draw_settings_view(self):
        """Draw the settings view with interactive buttons"""
        plot_area_x = self.sidebar_width
        plot_area_y = 10
        plot_area_width = self.plot_width
        
        # Title
        title = self.font.render("Settings", True, self.text_color)
        self.surface.blit(title, (plot_area_x + 10, plot_area_y))
        
        # Settings buttons and indicators
        button_height = 40
        button_width = 200
        button_x = plot_area_x + 20
        button_spacing = 15
        
        # Function to draw a toggle button
        def draw_toggle_button(y, label, state, setting_key):
            # Button background
            button_color = self.accent_color if state else (60, 60, 60)
            pygame.draw.rect(self.surface, button_color, 
                            (button_x, y, button_width, button_height), 
                            border_radius=5)
            
            # Button text
            status_text = "ON" if state else "OFF"
            text = self.font.render(f"{label}: {status_text}", True, self.text_color)
            text_rect = text.get_rect(center=(button_x + button_width // 2, y + button_height // 2))
            self.surface.blit(text, text_rect)
            
            # Store button bounds for click detection
            return pygame.Rect(button_x, y, button_width, button_height), setting_key
        
        # Function to draw an action button
        def draw_action_button(y, label, color, setting_key):
            # Button background
            pygame.draw.rect(self.surface, color, 
                            (button_x, y, button_width, button_height), 
                            border_radius=5)
            
            # Button text
            text = self.font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=(button_x + button_width // 2, y + button_height // 2))
            self.surface.blit(text, text_rect)
            
            # Store button bounds for click detection
            return pygame.Rect(button_x, y, button_width, button_height), setting_key
        
        # Function to draw an info display
        def draw_info_display(y, label, value):
            # Background
            pygame.draw.rect(self.surface, (40, 40, 40), 
                            (button_x, y, button_width, button_height), 
                            border_radius=5)
            
            # Text
            text = self.font.render(f"{label}: {value}", True, self.text_color)
            text_rect = text.get_rect(center=(button_x + button_width // 2, y + button_height // 2))
            self.surface.blit(text, text_rect)
        
        # Store button bounds for click detection
        self.settings_buttons = []
        
        # Current y position for drawing
        current_y = plot_area_y + 40
        
        # Connected nodes indicator
        draw_info_display(current_y, "Connected Nodes", self.settings['connected_nodes'])
        current_y += button_height + button_spacing
        
        # Reset plots button (RED)
        button, key = draw_action_button(current_y, "Reset All Plots", (180, 30, 30), 'reset_plots')
        self.settings_buttons.append((button, key))
        current_y += button_height + button_spacing
        
        # Reset registry button (ORANGE)
        button, key = draw_action_button(current_y, "Reset Signal Registry", (180, 100, 30), 'reset_registry')
        self.settings_buttons.append((button, key))
        current_y += button_height + button_spacing
        
        # Caps toggle
        button, key = draw_toggle_button(current_y, "Value Caps", self.settings['caps_enabled'], 'caps_enabled')
        self.settings_buttons.append((button, key))
        current_y += button_height + button_spacing
        
        # Light mode toggle
        button, key = draw_toggle_button(current_y, "Light Mode", self.settings['light_mode'], 'light_mode')
        self.settings_buttons.append((button, key))
        current_y += button_height + button_spacing
        
        # Performance mode toggle
        button, key = draw_toggle_button(current_y, "Performance Mode", self.settings['performance_mode'], 'performance_mode')
        self.settings_buttons.append((button, key))
        current_y += button_height + button_spacing
        
        # Version info
        version_text = self.font.render("PlotUnit v1.0", True, self.text_color)
        self.surface.blit(version_text, (button_x, self.height - 40))
    
    def _draw_plot(self, data, title, color):
        """Draw plot with data and grid"""
        plot_area_x = self.sidebar_width
        plot_area_y = 40
        plot_area_width = self.plot_width
        plot_area_height = self.height - 80
        
        # Draw title
        title_text = self.font.render(title, True, self.text_color)
        self.surface.blit(title_text, (plot_area_x + 10, 10))
        
        # Draw plot background
        pygame.draw.rect(self.surface, (25, 25, 25) if not self.settings['light_mode'] else (230, 230, 230), 
                        (plot_area_x, plot_area_y, plot_area_width, plot_area_height))
        
        # Draw grid
        self._draw_grid(plot_area_x, plot_area_y, plot_area_width, plot_area_height)
        
        # Draw actual data plot
        if len(data) > 1:
            # Apply caps if enabled
            if self.settings['caps_enabled']:
                # Normalize data to fit plot area
                data_min = np.min(data)
                data_max = np.max(data)
                
                # If min and max are the same, adjust to avoid division by zero
                if data_min == data_max:
                    data_min -= 0.5
                    data_max += 0.5
            else:
                # Use fixed range when caps disabled
                data_min = -1.0
                data_max = 1.0
            
            # Calculate scaling factors
            x_scale = plot_area_width / (len(data) - 1)
            y_scale = plot_area_height / (data_max - data_min)
            
            # Draw line connecting data points
            points = []
            for i, value in enumerate(data):
                x = plot_area_x + i * x_scale
                y = plot_area_y + plot_area_height - (value - data_min) * y_scale
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)
        
        # Draw min/max values at bottom
        min_text = self.font.render(f"Min: {np.min(data):.2f}", True, self.text_color)
        max_text = self.font.render(f"Max: {np.max(data):.2f}", True, self.text_color)
        avg_text = self.font.render(f"Avg: {np.mean(data):.2f}", True, self.text_color)
        
        self.surface.blit(min_text, (plot_area_x + 10, self.height - 30))
        self.surface.blit(max_text, (plot_area_x + 120, self.height - 30))
        self.surface.blit(avg_text, (plot_area_x + 220, self.height - 30))
    
    def _draw_grid(self, x, y, width, height):
        """Draw a grid for the plot background"""
        # Vertical grid lines
        for i in range(5):
            line_x = x + (width // 4) * i
            pygame.draw.line(self.surface, self.grid_color, 
                            (line_x, y), (line_x, y + height))
        
        # Horizontal grid lines
        for i in range(5):
            line_y = y + (height // 4) * i
            pygame.draw.line(self.surface, self.grid_color, 
                            (x, line_y), (x + width, line_y))
    
    def update_data(self, data, data_type='raw'):
        """Update data in a thread-safe way"""
        if not self.initialized:
            logger.warning("Cannot update data: PlotUnit not initialized")
            return
        
        # Debug log data properties
        logger.info(f"Queueing {data_type} data: shape={np.shape(data)}, min={np.min(data)}, max={np.max(data)}")
        
        # Put update message in queue
        self.event_queue.put({
            'type': 'update_data',
            'data_type': data_type,
            'data': np.array(data)
        })
    
    def add_signal_data(self, signal_data, name="signal_1", color=None):
        """
        Add a new signal to the visualization hub.
        
        Args:
            signal_data: numpy array or torch.Tensor containing signal data
            name: String name for the signal
            color: Optional RGB tuple for the signal color
        """
        if not self.initialized:
            return
            
        # Convert tensor to numpy if needed
        if isinstance(signal_data, torch.Tensor):
            # Convert to numpy and ensure it's 1D
            if len(signal_data.shape) > 1:
                # Take the first row/channel if multidimensional
                data = signal_data[0].flatten().cpu().numpy()
            else:
                data = signal_data.cpu().numpy()
            
            # Limit to 100 data points
            if len(data) > 100:
                data = data[:100]
            elif len(data) < 100:
                # Pad with zeros if too short
                data = np.pad(data, (0, 100 - len(data)), 'constant')
        elif isinstance(signal_data, (list, np.ndarray)):
            data = np.array(signal_data)
            # Limit to 100 data points
            if len(data) > 100:
                data = data[:100]
            elif len(data) < 100:
                # Pad with zeros if too short
                data = np.pad(data, (0, 100 - len(data)), 'constant')
        else:
            logger.error(f"Unsupported data type for visualization: {type(signal_data)}")
            return
            
        # If no color specified, generate one based on the signal name
        if color is None:
            # Generate a consistent color from the name
            import hashlib
            hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
            r = (hash_val & 0xFF0000) >> 16
            g = (hash_val & 0x00FF00) >> 8
            b = hash_val & 0x0000FF
            color = (min(r + 100, 255), min(g + 100, 255), min(b + 100, 255))  # Make brighter
        
        # Put update message in queue
        self.event_queue.put({
            'type': 'update_data',
            'data_type': name,
            'data': data,
            'color': color
        })
    
    def increment_connected_nodes(self):
        """Increment the count of connected nodes"""
        self.settings['connected_nodes'] += 1
        logger.info(f"Node connected. Total connected nodes: {self.settings['connected_nodes']}")
    
    def decrement_connected_nodes(self):
        """Decrement the count of connected nodes"""
        if self.settings['connected_nodes'] > 0:
            self.settings['connected_nodes'] -= 1
        logger.info(f"Node disconnected. Total connected nodes: {self.settings['connected_nodes']}")
    
    def clear_plots(self):
        """Clear all plots and reset the visualization"""
        try:
            logger.info("Clearing all plots")
            # Thread-safe access to clear plots
            self._lock.acquire()
            try:
                # Reset main visualization data
                with self.data_lock:
                    self.data = {
                        'raw': np.zeros(100),
                        'filtered': np.zeros(100)
                    }
                
                # Reset all other data structures
                if hasattr(self, 'signals'):
                    self.signals.clear()
                
                # Reset any other visualization-related properties
                if hasattr(self, 'plot_data'):
                    self.plot_data = {}
                
                logger.info("Plots cleared successfully")
            finally:
                self._lock.release()
        except Exception as e:
            logger.error(f"Error clearing plots: {str(e)}", exc_info=True)
    
    def reset_visualization(self):
        """Alternative method to reset visualization if clear_plots is not implemented"""
        print("[DEBUG-PLOT_UNIT] Reset visualization called")
        self.clear_plots()
