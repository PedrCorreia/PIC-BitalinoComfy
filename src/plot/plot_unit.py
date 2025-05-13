import pygame
import threading
import numpy as np
import torch
import queue
from enum import Enum
import logging
import os
import atexit
import time
import time
from .plot_unit_extensions import PlotUnitExtensions

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlotUnit')

class ViewMode(Enum):
    RAW = 0
    PROCESSED = 1  # Renamed from FILTERED
    TWIN = 2  # New mode for showing raw and processed side-by-side
    SETTINGS = 3

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
        self.width = 800  # Increased width for side-by-side views
        self.height = 600  # Increased height for multiple signals
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
                            self.current_mode = ViewMode.PROCESSED
                        elif y < button_height * 3:
                            self.current_mode = ViewMode.TWIN
                        elif y < button_height * 4:
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
                from comfy.Registry.mock_signal_node import SignalRegistry
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
                        
                        # If this is a new data type, make sure it's registered
                        if data_type not in ['raw', 'filtered'] and data_type not in self.data:
                            self.data[data_type] = message['data']
                            if 'color' in message:
                                # Store the color information for this signal
                                if not hasattr(self, 'signal_colors'):
                                    self.signal_colors = {}
                                self.signal_colors[data_type] = message['color']
                
                elif message['type'] == 'node_connected' or message['type'] == 'node_disconnected':
                    # These messages just serve to update the display
                    logger.info(f"Processed {message['type']} event. Current connected nodes: {self.settings['connected_nodes']}")
                    
                elif message['type'] == 'refresh':
                    logger.info("Refresh event received. Forcing visualization update.")
                    
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
        elif self.current_mode == ViewMode.PROCESSED:
            self._draw_processed_view()  # Renamed from filtered
        elif self.current_mode == ViewMode.TWIN:
            self._draw_twin_view()  # New method for side-by-side
        elif self.current_mode == ViewMode.SETTINGS:
            self._draw_settings_view()
    
    def _draw_sidebar_buttons(self):
        """Draw the sidebar navigation buttons"""
        button_height = 40
        button_icons = ["R", "P", "T", "S"]  # Raw, Processed, Twin, Settings
        
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
        """Draw the raw signal visualization with multiple signals stacked"""
        # Get available signal IDs
        signals_to_show = []
        with self.data_lock:
            # First find up to 3 signals to display
            for signal_id in self.data:
                if signal_id != 'filtered' and signal_id != 'processed':
                    signals_to_show.append(signal_id)
                if len(signals_to_show) >= 3:  # Limit to 3 signals
                    break
        
        if not signals_to_show:
            # Fall back to showing the default 'raw' signal if no other signals
            with self.data_lock:
                data = np.copy(self.data.get('raw', np.zeros(100)))
            self._draw_plot(data, "Raw Signal", (220, 180, 0))
        else:
            # Draw multiple stacked plots
            self._draw_multi_plot(signals_to_show, "Raw Signals", is_processed=False)
    
    def _draw_plot(self, data, title, color):
        """Draw a full-screen plot (legacy method for compatibility)"""
        # Plot area dimensions
        plot_area_x = self.sidebar_width
        plot_area_y = 40
        plot_area_width = self.plot_width
        plot_area_height = self.height - 80
        
        # Draw title
        title_text = self.font.render(title, True, self.text_color)
        self.surface.blit(title_text, (plot_area_x + 10, 10))
        
        # Use the signal panel method to draw the actual plot
        self._draw_signal_panel(
            data=data,
            title="",  # Title already drawn above
            color=color,
            x=plot_area_x,
            y=plot_area_y,
            width=plot_area_width,
            height=plot_area_height
        )
        
    def _draw_processed_view(self):
        """Draw the processed signal visualization (renamed from filtered)"""
        # Get available signal IDs
        signals_to_show = []
        with self.data_lock:
            # First find up to 3 signals to display
            for signal_id in self.data:
                # Check if we have a processed version of this signal
                processed_id = f"{signal_id}_processed"
                if processed_id in self.data:
                    signals_to_show.append(processed_id)
                if len(signals_to_show) >= 3:  # Limit to 3 signals
                    break
        
        if not signals_to_show:
            # Fall back to showing the default 'filtered' signal if no other signals
            with self.data_lock:
                data = np.copy(self.data.get('filtered', np.zeros(100)))
            self._draw_plot(data, "Processed Signal", (0, 180, 220))
        else:
            # Draw multiple stacked plots
            self._draw_multi_plot(signals_to_show, "Processed Signals", is_processed=True)
    
    def _draw_twin_view(self):
        """Draw raw and processed signals side by side"""
        # Get available signal IDs
        raw_signals = []
        processed_signals = []
        
        with self.data_lock:
            # Find up to 3 raw signals with corresponding processed versions
            for signal_id in self.data:
                if signal_id != 'filtered' and signal_id != 'processed' and signal_id != 'raw':
                    # Check if we also have a processed version
                    processed_id = f"{signal_id}_processed"
                    if processed_id in self.data:
                        raw_signals.append(signal_id)
                        processed_signals.append(processed_id)
                        
                    if len(raw_signals) >= 3:  # Limit to 3 signals
                        break
        
        # Fall back to default signals if no paired signals found
        if not raw_signals:
            with self.data_lock:
                raw_data = np.copy(self.data.get('raw', np.zeros(100)))
                processed_data = np.copy(self.data.get('filtered', np.zeros(100)))
            
            # Draw split screen with raw and processed signals
            self._draw_signal_panel(raw_data, "Raw Signal", (220, 180, 0), 
                                   self.sidebar_width, 0, self.plot_width // 2 - 5)
            
            self._draw_signal_panel(processed_data, "Processed Signal", (0, 180, 220),
                                  self.sidebar_width + self.plot_width // 2 + 5, 0, 
                                  self.plot_width // 2 - 5)
        else:
            # Left panel - Raw signals
            panel_width = self.plot_width // 2 - 5
            left_panel_x = self.sidebar_width
            
            # Draw raw panel with title
            title_text = self.font.render("Raw Signals", True, self.text_color)
            self.surface.blit(title_text, (left_panel_x + 10, 10))
            
            # Right panel - Processed signals
            right_panel_x = self.sidebar_width + panel_width + 10
            
            # Draw processed panel with title
            title_text = self.font.render("Processed Signals", True, self.text_color)
            self.surface.blit(title_text, (right_panel_x + 10, 10))
            
            # Calculate height for each signal
            panel_height = (self.height - 80) // min(len(raw_signals), 3)
            
            # Draw each pair of signals
            for i, (raw_id, proc_id) in enumerate(zip(raw_signals, processed_signals)):
                if i >= 3:  # Safety check
                    break
                
                # Get signal colors
                raw_color = self.signal_colors.get(raw_id, (220, 180, 0))
                proc_color = self.signal_colors.get(proc_id, (0, 180, 220))
                
                # Calculate vertical position
                y_offset = 40 + (panel_height * i)
                
                # Get signal data
                with self.data_lock:
                    raw_data = np.copy(self.data.get(raw_id, np.zeros(100)))
                    proc_data = np.copy(self.data.get(proc_id, np.zeros(100)))
                
                # Draw raw signal on left panel
                panel_title = f"{raw_id}"
                self._draw_signal_panel(raw_data, panel_title, raw_color, 
                                       left_panel_x, y_offset, panel_width, 
                                       panel_height - 10)
                                       
                # Draw processed signal on right panel
                panel_title = f"{proc_id}"
                self._draw_signal_panel(proc_data, panel_title, proc_color, 
                                       right_panel_x, y_offset, panel_width, 
                                       panel_height - 10)
    
    def _draw_settings_view(self):
        """Draw the settings panel"""
        # Settings area dimensions
        settings_x = self.sidebar_width + 20
        settings_y = 40
        button_width = 200
        button_height = 30
        button_margin = 10
        
        # Title
        title_text = self.font.render("Settings", True, self.text_color)
        self.surface.blit(title_text, (settings_x, 10))
        
        # Create rectangles for each setting (button)
        self.settings_buttons = []  # Reset buttons list
        
        # Light/Dark Mode toggle
        light_mode_text = "Light Mode: ON" if self.settings['light_mode'] else "Light Mode: OFF"
        light_mode_rect = pygame.Rect(settings_x, settings_y, button_width, button_height)
        self.settings_buttons.append((light_mode_rect, 'light_mode'))
        
        # Performance Mode toggle
        perf_y = settings_y + button_height + button_margin
        perf_mode_text = "Performance Mode: ON" if self.settings['performance_mode'] else "Performance Mode: OFF"
        perf_mode_rect = pygame.Rect(settings_x, perf_y, button_width, button_height)
        self.settings_buttons.append((perf_mode_rect, 'performance_mode'))
        
        # FPS Cap toggle
        caps_y = perf_y + button_height + button_margin
        caps_text = "Auto-scaling: ON" if self.settings['caps_enabled'] else "Auto-scaling: OFF"
        caps_rect = pygame.Rect(settings_x, caps_y, button_width, button_height)
        self.settings_buttons.append((caps_rect, 'caps_enabled'))
        
        # Reset Plots button
        reset_y = caps_y + button_height + button_margin * 2  # Extra margin
        reset_text = "Reset Plots"
        reset_rect = pygame.Rect(settings_x, reset_y, button_width, button_height)
        self.settings_buttons.append((reset_rect, 'reset_plots'))
        
        # Reset Registry button
        registry_y = reset_y + button_height + button_margin
        registry_text = "Reset Signal Registry"
        registry_rect = pygame.Rect(settings_x, registry_y, button_width, button_height)
        self.settings_buttons.append((registry_rect, 'reset_registry'))
        
        # Connected nodes info
        nodes_y = registry_y + button_height + button_margin * 2  # Extra margin
        nodes_text = f"Connected Nodes: {self.settings['connected_nodes']}"
        
        # Draw all buttons
        pygame.draw.rect(self.surface, self.accent_color if self.settings['light_mode'] else (40, 40, 40), 
                         light_mode_rect)
        self.surface.blit(self.font.render(light_mode_text, True, self.text_color), 
                          (settings_x + 10, settings_y + 5))
        
        pygame.draw.rect(self.surface, self.accent_color if self.settings['performance_mode'] else (40, 40, 40), 
                         perf_mode_rect)
        self.surface.blit(self.font.render(perf_mode_text, True, self.text_color), 
                          (settings_x + 10, perf_y + 5))
        
        pygame.draw.rect(self.surface, self.accent_color if self.settings['caps_enabled'] else (40, 40, 40), 
                         caps_rect)
        self.surface.blit(self.font.render(caps_text, True, self.text_color), 
                          (settings_x + 10, caps_y + 5))
        
        # Action buttons have different styling
        pygame.draw.rect(self.surface, (80, 0, 0), reset_rect)  # Red for reset
        self.surface.blit(self.font.render(reset_text, True, self.text_color), 
                          (settings_x + 10, reset_y + 5))
        
        pygame.draw.rect(self.surface, (80, 0, 0), registry_rect)  # Red for registry reset
        self.surface.blit(self.font.render(registry_text, True, self.text_color), 
                          (settings_x + 10, registry_y + 5))
        
        # Info text (not a button)
        self.surface.blit(self.font.render(nodes_text, True, self.text_color), 
                          (settings_x + 10, nodes_y + 5))
    
    def _draw_multi_plot(self, signal_ids, title, is_processed=False):
        """Draw multiple stacked signals in a single view"""
        if not signal_ids:
            return
            
        # Plot area dimensions
        plot_area_x = self.sidebar_width
        plot_area_y = 40
        plot_area_width = self.plot_width
        
        # Draw title
        title_text = self.font.render(title, True, self.text_color)
        self.surface.blit(title_text, (plot_area_x + 10, 10))
        
        # Calculate height for each signal panel
        panel_count = min(len(signal_ids), 3)  # Limit to max 3 panels
        panel_height = (self.height - 80) // panel_count
        
        # Draw stacked signal panels
        for i, signal_id in enumerate(signal_ids[:panel_count]):
            # Calculate vertical position
            y_offset = plot_area_y + (i * panel_height)
            
            # Get signal color
            color = (0, 180, 220) if is_processed else (220, 180, 0)  # Default colors
            if hasattr(self, 'signal_colors') and signal_id in self.signal_colors:
                color = self.signal_colors[signal_id]
            
            # Get signal data
            with self.data_lock:
                data = np.copy(self.data.get(signal_id, np.zeros(100)))
            
            # Draw panel for this signal
            self._draw_signal_panel(
                data=data,
                title=signal_id,
                color=color,
                x=plot_area_x,
                y=y_offset,
                width=plot_area_width,
                height=panel_height - 10
            )
    
    def _draw_signal_panel(self, data, title, color, x, y, width, height=None):
        """Draw a signal panel with data, grid and stats"""
        if height is None:
            height = self.height - 80
            
        # Panel title (if smaller panels, use smaller font)
        small_font = height < 150
        if small_font:
            title_font = pygame.font.SysFont("Arial", 12)
        else:
            title_font = self.font
            
        title_text = title_font.render(title, True, self.text_color)
        self.surface.blit(title_text, (x + 5, y + 5))
        
        # Draw panel background
        panel_y = y + 25  # Leave space for title
        panel_height = height - 25  # Adjust height to leave space for title
        
        pygame.draw.rect(self.surface, 
                        (25, 25, 25) if not self.settings['light_mode'] else (230, 230, 230), 
                        (x, panel_y, width, panel_height))
        
        # Draw grid
        self._draw_grid(x, panel_y, width, panel_height)
        
        # Draw actual data plot
        if len(data) > 1:
            # Apply caps if enabled
            if self.settings['caps_enabled']:
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
            x_scale = width / (len(data) - 1)
            y_scale = panel_height / (data_max - data_min)
            
            # Draw line connecting data points
            points = []
            for i, value in enumerate(data):
                point_x = x + i * x_scale
                point_y = panel_y + panel_height - (value - data_min) * y_scale
                points.append((point_x, point_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)
        
        # Draw stats if there's enough space
        if height > 120:
            stats_y = y + height - 20
            stats_font = pygame.font.SysFont("Arial", 12) if small_font else self.font
            
            # Draw min/max/avg values at bottom
            min_text = stats_font.render(f"Min: {np.min(data):.2f}", True, self.text_color)
            max_text = stats_font.render(f"Max: {np.max(data):.2f}", True, self.text_color)
            avg_text = stats_font.render(f"Avg: {np.mean(data):.2f}", True, self.text_color)
            
            self.surface.blit(min_text, (x + 5, stats_y))
            self.surface.blit(max_text, (x + 90, stats_y))
            self.surface.blit(avg_text, (x + 175, stats_y))
    
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
    
    def add_signal_data(self, signal_data, name="signal_1", color=None):
        """Add signal data to the visualization"""
        if isinstance(signal_data, (list, np.ndarray)):
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
        """
        Increment the count of connected nodes.
        Used when a new node connects to the visualization hub.
        """
        # Increment counter
        self.settings['connected_nodes'] += 1
        
        # Log and notify visualization thread
        logger.info(f"Node connected. Total connected nodes: {self.settings['connected_nodes']}")
        
        # Put message in queue for display update
        if self.initialized:
            self.event_queue.put({
                'type': 'node_connected'
            })
        
        return self.settings['connected_nodes']
    
    def decrement_connected_nodes(self):
        """
        Decrement the count of connected nodes.
        Used when a node disconnects from the visualization hub.
        """
        # Only decrement if we have connected nodes
        if self.settings['connected_nodes'] > 0:
            self.settings['connected_nodes'] -= 1
        
        # Log and notify visualization thread
        logger.info(f"Node disconnected. Total connected nodes: {self.settings['connected_nodes']}")
        
        # Put message in queue for display update
        if self.initialized:
            self.event_queue.put({
                'type': 'node_disconnected'
            })
        
    def clear_plots(self):
        """Clear all plots and reset the visualization"""
        try:
            logger.info("Clearing all plots")
            with self.data_lock:
                self.data = {
                    'raw': np.zeros(100),
                    'filtered': np.zeros(100)
                }
                if hasattr(self, 'signal_colors'):
                    self.signal_colors.clear()
            logger.info("Plots cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing plots: {str(e)}", exc_info=True)

    def update(self):
        """
        Force an update of the visualization.
        This method ensures the plot is refreshed with the latest data.
        """
        # Put a refresh message in the queue
        self.event_queue.put({
            'type': 'refresh',
            'timestamp': time.time()
        })
        
        # Log the update request
        logger.info("Plot update requested")
    
    def reset_visualization(self):
        """Alternative method to reset visualization if clear_plots is not implemented"""
        logger.info("Reset visualization called")
        self.clear_plots()
