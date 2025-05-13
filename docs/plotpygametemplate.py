import threading
import numpy as np
import time
import queue
import weakref
from collections import deque, OrderedDict
import atexit
import sys

# Try importing PyGame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available, plotting will be limited")

# Primary Plot Class - PygamePlot
class PygamePlot:
    # ==== CENTRALIZED CONFIGURATION SETTINGS ====
    # Window settings
    DEFAULT_WIDTH = 250  # Increased default window width
    DEFAULT_HEIGHT = 350
    
    # Colors configuration
    BACKGROUND_COLOR = (10, 10, 10)  # Dark background
    AXIS_COLOR = (200, 200, 200)  # Light gray for axes
    GRID_COLOR = (60, 60, 60)  # Darker gray for grid lines
    TEXT_COLOR = (220, 220, 220)  # Light text
    TIME_INFO_COLOR = (255, 255, 0)  # Yellow for time information
    
    # Signal colors - centralized here for easy modification
    SIGNAL_COLORS = {
        "EDA": (0, 255, 0),    # Green
        "ECG": (255, 0, 0),    # Red
        "RR": (255, 165, 0),   # Orange
        "DEFAULT": (0, 0, 255) # Blue default
    }
    
    # Margin settings as percentage of window dimensions
    MARGIN_LEFT_PERCENT = 0.07  # 7% of width
    MARGIN_RIGHT_PERCENT = 0.05  # 5% of width
    MARGIN_TOP_PERCENT = 0.08  # 8% of height
    MARGIN_BOTTOM_PERCENT = 0.09  # 9% of height
    
    # Performance settings
    FPS = 60
    FPS_CAP_ENABLED = True
    PERFORMANCE_MODE = False
    SMART_DOWNSAMPLE = False  # Default to OFF for downsampling
    LINE_THICKNESS = 1
    
    # Sizing for multi-signal displays
    MIN_HEIGHT_PER_SIGNAL = 180  # Minimum height per signal in pixels
    MIN_WINDOW_HEIGHT = 300      # Minimum overall window height
    # ==== END CONFIGURATION SETTINGS ====

    _window = None
    _screen = None
    _lock = threading.Lock()
    _start_time = None
    _instances = []  # Track all plot instances
    _dirty_regions = []  # Track regions that need updates
    _font_cache = {}  # Cache for frequently used fonts
    _surf_cache = {}  # Cache for reusable surfaces
    _exit_registered = False  # Flag to track if exit handler is registered

    # Register cleanup handler to ensure all pygame windows close on exit
    @classmethod
    def _register_exit_handler(cls):
        if not cls._exit_registered and PYGAME_AVAILABLE:
            atexit.register(cls._cleanup_all_windows)
            cls._exit_registered = True
            print("Registered pygame cleanup handler")

    @classmethod
    def _cleanup_all_windows(cls):
        """Clean up all pygame windows on application exit"""
        if PYGAME_AVAILABLE and pygame.get_init():
            print("Cleaning up all pygame windows")
            # Stop all plot instances
            for instance_ref in list(cls._instances):
                instance = instance_ref()
                if instance is not None:
                    try:
                        instance._stop_event.set()
                    except Exception as e:
                        print(f"Error during instance cleanup: {e}")
            
            # Clear references
            cls._instances.clear()
            cls._font_cache.clear()
            cls._surf_cache.clear()
            cls._window = None
            cls._screen = None
            
            # Quit pygame properly
            try:
                pygame.quit()
            except Exception as e:
                print(f"Error quitting pygame: {e}")

    def __init__(self, width=None, height=None, performance_mode=False):
        # Register exit handler if not already done
        PygamePlot._register_exit_handler()
        
        self._plot_thread = None
        self._latest_data = ([], [], False)
        self._new_data = threading.Event()
        self._stop_event = threading.Event()
        self._last_draw_time = 0
        self._real_time_start = time.time()  # Initialize at creation time
        self._data_hash = None  # Track data changes
        self._cached_plot = None  # Cached plot surface
        self._render_queue = queue.Queue()  # Queue for rendering tasks
        self.sampling_rate = None  # Will be set by caller
        self._closed = False  # Track window closed state
        self._last_latency = 0
        self._window_closed_externally = False  # Track if window was closed by user
        self._resize_needed = False  # Flag to track if window resize is needed
        
        # Set dimensions - use defaults if not provided
        self.width = width if width is not None else self.DEFAULT_WIDTH
        self.height = height if height is not None else self.DEFAULT_HEIGHT
        # Set performance mode
        self.PERFORMANCE_MODE = performance_mode
        # Signal type for smart downsampling
        self.signal_type = None
        # Add a new flag for downsampling
        self.enable_downsampling = False  # Default to disabled
        
        # Multi-signal support
        self.multi_signal_mode = False
        self.enabled_signals = {"EDA": False, "ECG": False, "RR": False}
        # Use centralized color definitions
        self.signal_colors = self.SIGNAL_COLORS.copy()
        self._latest_multi_data = {}
        
        # Register this instance
        with PygamePlot._lock:
            PygamePlot._instances.append(weakref.ref(self))
    
    def __del__(self):
        self._stop_event.set()
        self._close_plot_resources()
    
    def _close_plot_resources(self):
        """Clean up plot resources properly"""
        # Signal any active threads to stop
        self._stop_event.set()
        
        # Wait a moment for threads to notice stop signal
        try:
            if self._plot_thread and self._plot_thread.is_alive():
                self._plot_thread.join(timeout=0.2)
        except Exception:
            pass
        
        # Reset state
        self._closed = True
        self._plot_thread = None

    def plot(self, x, y, as_points=False, x_min=None, x_max=None, signal_type=None, enable_downsampling=False):
        """
        Plot data using pygame visualization
        
        Args:
            x: list/array of x values
            y: list/array of y values
            as_points: display as points rather than lines
            x_min: optional minimum x-value for plotting window
            x_max: optional maximum x-value for plotting window
            signal_type: type of signal for smart downsampling (EDA, ECG, RR)
            enable_downsampling: whether to enable downsampling (default: False)
        """
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot plot data")
            return
            
        # Store signal type for smart downsampling
        if signal_type:
            self.signal_type = signal_type
            
        # Set downsampling flag
        self.enable_downsampling = enable_downsampling
            
        # Reset closed state when plot is called
        self._closed = False
        
        # Compute a hash of the data to detect changes
        data_len = len(x)
        data_hash = hash((data_len, 
                         hash(x[0]) if data_len > 0 else 0, 
                         hash(x[-1]) if data_len > 0 else 0,
                         hash(y[0]) if data_len > 0 else 0,
                         hash(y[-1]) if data_len > 0 else 0,
                         as_points,
                         hash(x_min) if x_min is not None else 0,
                         hash(x_max) if x_max is not None else 0))
        
        # Only update if data actually changed
        if self._data_hash != data_hash:
            self._data_hash = data_hash
            self._latest_data = (np.array(x, dtype=np.float32), 
                                np.array(y, dtype=np.float32), 
                                as_points,
                                x_min,
                                x_max)
            self._new_data.set()
            
        if self._plot_thread is None or not self._plot_thread.is_alive():
            print("Starting new plot thread")
            self._stop_event.clear()
            self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
            self._plot_thread.start()

    def plot_multi(self, data_snapshots, x_min=None, x_max=None, enable_downsampling=False):
        """
        Plot multiple signals stacked vertically
        
        Args:
            data_snapshots: Dictionary mapping signal types to data lists
            x_min: global minimum x-value for plotting window
            x_max: global maximum x-value for plotting window
            enable_downsampling: whether to enable downsampling
        """
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot plot data")
            return
        
        # Reset closed state when plot is called
        self._closed = False
        
        # Store settings
        self.enable_downsampling = enable_downsampling
        self.multi_signal_mode = True
        
        # Count active signals (those with data)
        active_signals = [signal for signal, data in data_snapshots.items() if data]
        num_signals = len(active_signals)
        
        # Calculate window height based on number of signals
        if num_signals > 0:
            # Dynamic height calculation: minimum height per signal plus margins
            target_height = max(
                self.MIN_WINDOW_HEIGHT,
                num_signals * self.MIN_HEIGHT_PER_SIGNAL + 100  # 100px for margins and labels
            )
            
            # Only resize if needed
            if target_height != self.height:
                self.height = target_height
                # Window will be resized in _plot_multi_loop if needed
        
        # Compute a simple hash of the data to detect changes
        data_hash = hash(tuple(
            (signal_type, len(data), 
             hash(data[0][0]) if data else 0, 
             hash(data[-1][0]) if data else 0)
            for signal_type, data in data_snapshots.items()
        ))
        
        # Only update if data actually changed
        if self._data_hash != data_hash:
            self._data_hash = data_hash
            self._latest_multi_data = {
                signal_type: {
                    'x': np.array([d[0] for d in data], dtype=np.float32) if data else np.array([], dtype=np.float32),
                    'y': np.array([d[1] for d in data], dtype=np.float32) if data else np.array([], dtype=np.float32),
                }
                for signal_type, data in data_snapshots.items()
            }
            self._latest_data = (x_min, x_max)  # Store global x range
            self._new_data.set()
        
        if self._plot_thread is None or not self._plot_thread.is_alive():
            print(f"Starting new multi-signal plot thread with {num_signals} signals")
            self._stop_event.clear()
            self._plot_thread = threading.Thread(target=self._plot_multi_loop, daemon=True)
            self._plot_thread.start()

    @staticmethod
    def _get_font(size, name=None):
        """Get a cached font to avoid recreation"""
        if not PYGAME_AVAILABLE:
            return None
            
        key = (name, size)
        if key not in PygamePlot._font_cache:
            PygamePlot._font_cache[key] = pygame.font.SysFont(name, size)
        return PygamePlot._font_cache[key]

    def _safe_min_max(self, arr):
        """Calculate min and max of array, safely handling empty arrays and NaN values"""
        if arr is None or len(arr) == 0:
            return 0.0, 1.0

        # Filter out NaN values
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            return 0.0, 1.0

        return float(np.min(valid)), float(np.max(valid))

    def _plot_loop(self):
        """Main plotting loop for pygame visualization"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot create plot")
            return
            
        # Reset window if it was previously closed by user or if resize is needed
        with PygamePlot._lock:
            if self._window_closed_externally or PygamePlot._window is None or self._resize_needed:
                PygamePlot._window = None
                PygamePlot._screen = None
                self._window_closed_externally = False
                self._resize_needed = False
                print("Resetting pygame window")
            
        # Window dimensions - use instance values
        w, h = self.width, self.height
        
        # Calculate margins from percentages
        margin_left = int(w * self.MARGIN_LEFT_PERCENT)
        margin_right = int(w * self.MARGIN_RIGHT_PERCENT)
        margin_top = int(h * self.MARGIN_TOP_PERCENT)
        margin_bottom = int(h * self.MARGIN_BOTTOM_PERCENT)
        
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom
        
        # Use double buffered hardware acceleration
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
        
        # Initialize Pygame only once or if window was closed
        with PygamePlot._lock:
            if PygamePlot._window is None:
                print(f"Initializing pygame window ({w}x{h})")
                pygame.init()
                PygamePlot._screen = pygame.display.set_mode((w, h), flags, vsync=1)
                pygame.display.set_caption("Signal Plot (Real-time)")
                PygamePlot._window = True
                PygamePlot._start_time = time.time()
                # Clear the surface cache after window recreation
                PygamePlot._surf_cache = {}
            else:
                print("Using existing pygame window")
                # Check if we need to resize
                current_w, current_h = PygamePlot._screen.get_size()
                if current_w != w or current_h != h:
                    print(f"Resizing pygame window to {w}x{h}")
                    PygamePlot._screen = pygame.display.set_mode((w, h), flags, vsync=1)
        
        screen = PygamePlot._screen
        
        # Create clock for consistent framerate
        clock = pygame.time.Clock()
        
        # Create cached background surface with correct size
        bg_key = f'bg_{w}x{h}'
        if bg_key not in PygamePlot._surf_cache:
            bg_surface = pygame.Surface((w, h)).convert()
            bg_surface.fill(self.BACKGROUND_COLOR)
            line_thickness = 1 if self.PERFORMANCE_MODE else 2
            pygame.draw.line(bg_surface, self.AXIS_COLOR, (margin_left, h - margin_bottom), 
                            (w - margin_right, h - margin_bottom), line_thickness)
            pygame.draw.line(bg_surface, self.AXIS_COLOR, (margin_left, h - margin_bottom), 
                            (margin_left, margin_top), line_thickness)
            PygamePlot._surf_cache[bg_key] = bg_surface
        
        # Create cached buffer surface with correct size
        buffer_key = f'plot_buffer_{w}x{h}'
        if buffer_key not in PygamePlot._surf_cache:
            plot_buffer = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            PygamePlot._surf_cache[buffer_key] = plot_buffer
        
        bg_surface = PygamePlot._surf_cache[bg_key]
        plot_buffer = PygamePlot._surf_cache[buffer_key]
        
        # Reusable objects - adjust font size for smaller windows
        font_size_normal = max(10, int(h * 0.035))
        font_size_bold = max(12, int(h * 0.04))
        font_size_signal = max(9, int(h * 0.03))  # Smaller font for signal labels
        font_normal = self._get_font(font_size_normal)
        font_bold = self._get_font(font_size_bold)
        font_signal = self._get_font(font_size_signal)
        
        # Main loop
        running = True
        min_interval = 1.0 / self.FPS if self.FPS_CAP_ENABLED else 0
        last_x_max = 0  # Track last max x value to detect time jumps
        
        while running and not self._stop_event.is_set():
            # Process events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self._closed = True
                    self._window_closed_externally = True  # Mark as closed by user
                    with PygamePlot._lock:
                        PygamePlot._window = None  # Reset window state
                        PygamePlot._screen = None
                    self._stop_event.set()
                    print("Window closed by user")
            
            # Wait for new data with a timeout
            self._new_data.wait(timeout=0.001)
            
            # Check FPS cap
            now = time.time()
            if self.FPS_CAP_ENABLED and (now - self._last_draw_time) < min_interval:
                clock.tick(self.FPS)
                continue
            
            # Process new data if available
            if self._new_data.is_set():
                if len(self._latest_data) == 3:  # Backward compatibility
                    x_arr, y_arr, as_points = self._latest_data
                    x_min_override, x_max_override = None, None
                else:
                    x_arr, y_arr, as_points, x_min_override, x_max_override = self._latest_data
                self._new_data.clear()
                self._last_draw_time = now
                needs_redraw = True
            else:
                # Skip redrawing if no new data
                needs_redraw = False
                
            if needs_redraw:
                screen.blit(bg_surface, (0, 0))
                plot_buffer.fill((0, 0, 0, 0))  # Clear with transparency
                
                # Calculate plotting parameters with corrections for real-time display
                if len(x_arr) > 0:
                    # Use provided min/max if available, otherwise calculate from data
                    if x_min_override is not None:
                        x_min = float(x_min_override)
                    else:
                        x_min = float(x_arr.min())
                        
                    if x_max_override is not None:
                        x_max = float(x_max_override)
                    else:
                        x_max = float(x_arr.max())
                else:
                    x_min, x_max = 0, 1

                # Ensure x_min and x_max are distinct to avoid div by zero
                if x_max <= x_min:
                    x_max = x_min + 0.1
                
                x_range = x_max - x_min
                show_labels = not self.PERFORMANCE_MODE

                # Always draw axis labels including min/max values
                if show_labels:
                    # X-axis labels - explicitly show min and max values
                    x_min_label = f"{x_min:.1f}"
                    x_max_label = f"{x_max:.1f}"
                    x_label = f"Time (s)"
                    y_label = f"Signal"
                    
                    # Render min/max labels
                    x_min_surf = font_normal.render(x_min_label, True, self.TEXT_COLOR)
                    x_max_surf = font_normal.render(x_max_label, True, self.TEXT_COLOR)
                    
                    # Draw min/max labels - position at actual axis endpoints
                    screen.blit(x_min_surf, (margin_left - x_min_surf.get_width()//2, h - margin_bottom + 15))
                    screen.blit(x_max_surf, (w - margin_right - x_max_surf.get_width()//2, h - margin_bottom + 15))
                    
                    # Render axis labels if needed
                    labels_key = f"labels_{font_size_normal}"
                    if f'{labels_key}_x' not in PygamePlot._surf_cache:
                        PygamePlot._surf_cache[f'{labels_key}_x'] = font_normal.render(x_label, True, self.TEXT_COLOR)
                    if f'{labels_key}_y' not in PygamePlot._surf_cache:
                        PygamePlot._surf_cache[f'{labels_key}_y'] = font_normal.render(y_label, True, self.TEXT_COLOR)
                    
                    # Draw main axis labels
                    screen.blit(PygamePlot._surf_cache[f'{labels_key}_x'], (w//2, h - margin_bottom + 25))
                    # Draw y-label rotated
                    rot_y_label = pygame.transform.rotate(PygamePlot._surf_cache[f'{labels_key}_y'], 90)
                    screen.blit(rot_y_label, (margin_left - 40, h//2 - 40))
                
                # Plot the data only if we have points
                if len(x_arr) > 0:
                    # Apply intelligent downsampling only if enabled
                    if self.SMART_DOWNSAMPLE and self.enable_downsampling:
                        x_arr, y_arr = self._smart_downsample(x_arr, y_arr)
                    
                    # Optimized vectorized calculation
                    if x_max - x_min == 0:
                        x_norm = np.full_like(x_arr, margin_left)
                    else:
                        # Normalize x values to plot width
                        x_norm = margin_left + (x_arr - x_min) * plot_w / x_range
                    
                    y_min = float(y_arr.min()) if len(y_arr) > 0 else 0.0
                    y_max = float(y_arr.max()) if len(y_arr) > 0 else 1.0
                    
                    # Ensure y_min and y_max are distinct
                    if y_max <= y_min:
                        y_max = y_min + 0.1
                    
                    y_range = y_max - y_min
                    
                    if y_range == 0:
                        y_norm = np.full_like(y_arr, margin_top + plot_h // 2)
                    else:
                        # Normalize y values to plot height (inverted because y=0 is at top)
                        y_norm = (h - margin_bottom) - (y_arr - y_min) * plot_h / y_range
                    
                    # Add Y-axis min/max labels if enabled
                    if show_labels:
                        y_min_label = f"{y_min:.2f}"
                        y_max_label = f"{y_max:.2f}"
                        
                        y_min_surf = font_normal.render(y_min_label, True, self.TEXT_COLOR)
                        y_max_surf = font_normal.render(y_max_label, True, self.TEXT_COLOR)
                        
                        # Position at actual axis endpoints
                        screen.blit(y_min_surf, (margin_left - y_min_surf.get_width() - 5, h - margin_bottom - 10))
                        screen.blit(y_max_surf, (margin_left - y_max_surf.get_width() - 5, margin_top))
                    
                    # Set line color based on signal type (using centralized colors)
                    if self.signal_type in self.signal_colors:
                        line_color = self.signal_colors[self.signal_type]
                    else:
                        line_color = self.signal_colors["DEFAULT"]
                        
                    # Draw signal type label at top right
                    if self.signal_type:
                        signal_label = font_signal.render(self.signal_type, True, line_color)
                        screen.blit(signal_label, (w - margin_right - signal_label.get_width() - 10, margin_top))
                    
                    # Draw signal - use numpy operations as much as possible
                    if len(x_norm) > 1:
                        # Add NaN filtering before drawing lines
                        valid_indices = ~(np.isnan(x_norm) | np.isnan(y_norm))
                        if np.any(valid_indices):
                            x_filtered = x_norm[valid_indices]
                            y_filtered = y_norm[valid_indices]
                            
                            # Create list of points with integer coordinates
                            pts = [(int(x), int(y)) for x, y in zip(x_filtered, y_filtered)]
                            
                            # Draw smoother lines unless in performance mode
                            if self.PERFORMANCE_MODE:
                                # In performance mode, use regular lines (faster)
                                if pts:  # Only draw if we have valid points
                                    pygame.draw.lines(screen, line_color, False, pts, thickness)
                            else:
                                # Apply smoothing for better visual quality
                                if pts:  # Only draw if we have valid points
                                    self._draw_smooth_line(screen, line_color, pts, thickness)
                        
                    elif len(x_norm) == 1:
                        # Check if single point is valid before drawing
                        if not np.isnan(x_norm[0]) and not np.isnan(y_norm[0]):
                            pygame.draw.circle(screen, line_color, (int(x_norm[0]), int(y_norm[0])), 2)

                # Consolidated status information row at the top
                info_row_y = 10
                
                # Real time tracking
                real_elapsed = now - self._real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"RT: {real_minutes:02}:{real_seconds:02}"
                real_time_surface = font_normal.render(real_time_str, True, self.TIME_INFO_COLOR)
                
                # Latency calculation
                if len(x_arr) > 0:
                    latency = real_elapsed - x_max
                    is_stable = abs(latency - self._last_latency) < 0.5 if hasattr(self, '_last_latency') else False
                    is_acceptable = latency <= 1
                    
                    if is_stable and is_acceptable:
                        latency_color = (100, 255, 100)  # Green
                        latency_str = f"Lat: {latency:.3f}s"
                    else:
                        latency_color = (255, 165, 0)  # Orange/yellow
                        latency_str = f"Lat: {latency:.3f}s"
                        
                    self._last_latency = latency
                    latency_surface = font_normal.render(latency_str, True, latency_color)
                else:
                    latency_surface = None
                
                # Mode indicator
                mode_str = f"{'Perf' if self.PERFORMANCE_MODE else 'Quality'}"
                mode_surface = font_normal.render(mode_str, True, self.TEXT_COLOR)
                
                # Sample rate if available
                if self.sampling_rate:
                    sr_str = f"SR: {self.sampling_rate} Hz"
                    sr_surface = font_normal.render(sr_str, True, self.TEXT_COLOR)
                else:
                    sr_surface = None
                
                # Position all info elements in a single row
                info_x = margin_left
                spacing = 15
                
                # Display real time
                screen.blit(real_time_surface, (info_x, info_row_y))
                info_x += real_time_surface.get_width() + spacing
                
                # Display latency if available
                if latency_surface:
                    screen.blit(latency_surface, (info_x, info_row_y))
                    info_x += latency_surface.get_width() + spacing
                
                # Display sample rate if available
                if sr_surface:
                    screen.blit(sr_surface, (info_x, info_row_y))
                    info_x += sr_surface.get_width() + spacing
                
                # Display mode
                screen.blit(mode_surface, (info_x, info_row_y))
                info_x += mode_surface.get_width() + spacing
                
                # Optimized update - only update the screen once per frame
                pygame.display.flip()
            
            # Maintain FPS
            if self.FPS_CAP_ENABLED:
                clock.tick(self.FPS)
        
        print("Plot thread terminated")

    def _plot_multi_loop(self):
        """Main plotting loop for multi-signal visualization"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot create plot")
            return
            
        # Reset window if it was previously closed by user or if resize is needed
        with PygamePlot._lock:
            if self._window_closed_externally or PygamePlot._window is None or self._resize_needed:
                PygamePlot._window = None
                PygamePlot._screen = None
                self._window_closed_externally = False
                self._resize_needed = False
                print("Resetting pygame window for multi-signal plot")
            
        # Window dimensions - use instance values
        w, h = self.width, self.height
        
        # Use double buffered hardware acceleration
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
        
        # Initialize Pygame only once or if window was closed
        with PygamePlot._lock:
            if PygamePlot._window is None:
                print(f"Initializing pygame window ({w}x{h})")
                pygame.init()
                PygamePlot._screen = pygame.display.set_mode((w, h), flags, vsync=1)
                pygame.display.set_caption("Multi-Signal Plot (Real-time)")
                PygamePlot._window = True
                PygamePlot._start_time = time.time()
                PygamePlot._surf_cache = {}
            else:
                print("Using existing pygame window")
                current_w, current_h = PygamePlot._screen.get_size()
                if current_w != w or current_h != h:
                    print(f"Resizing pygame window to {w}x{h}")
                    PygamePlot._screen = pygame.display.set_mode((w, h), flags, vsync=1)
                    # Clear surface cache when window size changes
                    PygamePlot._surf_cache = {}
        
        screen = PygamePlot._screen
        
        # Create clock for consistent framerate
        clock = pygame.time.Clock()
        
        # Create cached background surface with correct size
        bg_key = f'bg_multi_{w}x{h}'
        if bg_key not in PygamePlot._surf_cache:
            bg_surface = pygame.Surface((w, h)).convert()
            bg_surface.fill(self.BACKGROUND_COLOR)
            PygamePlot._surf_cache[bg_key] = bg_surface
        
        # Create cached buffer surface with correct size
        buffer_key = f'plot_buffer_{w}x{h}'
        if buffer_key not in PygamePlot._surf_cache:
            plot_buffer = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            PygamePlot._surf_cache[buffer_key] = plot_buffer
        
        bg_surface = PygamePlot._surf_cache[bg_key]
        plot_buffer = PygamePlot._surf_cache[buffer_key]
        
        # Reusable objects - adjust font size for smaller windows
        font_size_normal = max(10, int(h * 0.035))
        font_size_bold = max(12, int(h * 0.04))
        font_size_signal = max(9, int(h * 0.03))  # Smaller font for signal labels
        font_normal = self._get_font(font_size_normal)
        font_bold = self._get_font(font_size_bold)
        font_signal = self._get_font(font_size_signal)
        
        # Calculate margins from percentages
        margin_left = int(w * self.MARGIN_LEFT_PERCENT)
        margin_right = int(w * self.MARGIN_RIGHT_PERCENT)
        margin_top = int(h * self.MARGIN_TOP_PERCENT)
        margin_bottom = int(h * self.MARGIN_BOTTOM_PERCENT)
        
        plot_w = w - margin_left - margin_right
        
        # Main loop
        running = True
        min_interval = 1.0 / self.FPS if self.FPS_CAP_ENABLED else 0
        
        while running and not self._stop_event.is_set():
            # Process events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self._closed = True
                    self._window_closed_externally = True  # Mark as closed by user
                    with PygamePlot._lock:
                        PygamePlot._window = None  # Reset window state
                        PygamePlot._screen = None
                    self._stop_event.set()
                    print("Window closed by user")
            
            # Wait for new data with a timeout
            self._new_data.wait(timeout=0.001)
            
            # Check FPS cap
            now = time.time()
            if self.FPS_CAP_ENABLED and (now - self._last_draw_time) < min_interval:
                clock.tick(self.FPS)
                continue
            
            # Process new data if available
            if self._new_data.is_set():
                x_min_override, x_max_override = self._latest_data
                self._new_data.clear()
                self._last_draw_time = now
                needs_redraw = True
            else:
                # Skip redrawing if no new data
                needs_redraw = False
                
            if needs_redraw:
                # Start with clean background
                screen.blit(bg_surface, (0, 0))
                plot_buffer.fill((0, 0, 0, 0))  # Clear with transparency
                
                # Get the active signals (those with data)
                active_signals = [signal for signal, data in self._latest_multi_data.items() 
                                if len(data['x']) > 0]
                
                # Skip if no active signals
                if not active_signals:
                    pygame.display.flip()
                    continue
                
                # Determine global x range
                if x_min_override is not None and x_max_override is not None:
                    x_min = float(x_min_override)
                    x_max = float(x_max_override)
                else:
                    x_min = float('inf')
                    x_max = float('-inf')
                    
                    for signal in active_signals:
                        x_arr = self._latest_multi_data[signal]['x']
                        if len(x_arr) > 0:
                            # Use safe min/max calculation
                            x_min_signal, x_max_signal = self._safe_min_max(x_arr)
                            x_min = min(x_min, x_min_signal)
                            x_max = max(x_max, x_max_signal)
                    
                    if x_min == float('inf'):
                        x_min = 0
                    if x_max == float('-inf'):
                        x_max = 1

                # Ensure x_min and x_max are distinct to avoid div by zero
                if x_max <= x_min:
                    x_max = x_min + 0.1
                
                x_range = x_max - x_min
                
                # Calculate the height available for each plot
                # Reserve space between plots (3% of total height per divider)
                divider_height = int(h * 0.03) if len(active_signals) > 1 else 0
                total_divider_height = divider_height * (len(active_signals) - 1)
                available_height = h - margin_top - margin_bottom - total_divider_height
                
                # Height per plot
                plot_h = available_height // len(active_signals)
                
                # Draw global x-axis labels
                x_min_label = f"{x_min:.1f}"
                x_max_label = f"{x_max:.1f}"
                x_label = f"Time (s)"
                
                # Draw global x axis
                line_thickness = 1 if self.PERFORMANCE_MODE else 2
                line_color = self.AXIS_COLOR
                
                # Draw horizontal x-axis at the bottom
                pygame.draw.line(screen, line_color, 
                                (margin_left, h - margin_bottom), 
                                (w - margin_right, h - margin_bottom), 
                                line_thickness)
                
                # Draw x-axis labels
                if not self.PERFORMANCE_MODE:
                    # X-axis min/max labels
                    x_min_surf = font_normal.render(x_min_label, True, self.TEXT_COLOR)
                    x_max_surf = font_normal.render(x_max_label, True, self.TEXT_COLOR)
                    
                    # Position at the left and right of the axis
                    screen.blit(x_min_surf, (margin_left - x_min_surf.get_width()//2, h - margin_bottom + 15))
                    screen.blit(x_max_surf, (w - margin_right - x_max_surf.get_width()//2, h - margin_bottom + 15))
                    
                    # X-axis label centered below
                    x_axis_label = font_normal.render(x_label, True, self.TEXT_COLOR)
                    screen.blit(x_axis_label, (w//2 - x_axis_label.get_width()//2, h - margin_bottom + 25))
                
                # Consolidated status information row at the top
                info_row_y = 10
                
                # Real time tracking
                real_elapsed = now - self._real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"RT: {real_minutes:02}:{real_seconds:02}"
                real_time_surface = font_normal.render(real_time_str, True, self.TIME_INFO_COLOR)
                
                # Latency calculation
                latency = real_elapsed - x_max
                is_stable = abs(latency - self._last_latency) < 0.5 if hasattr(self, '_last_latency') else False
                is_acceptable = latency <= 1
                
                if is_stable and is_acceptable:
                    latency_color = (100, 255, 100)  # Green
                    latency_str = f"Lat: {latency:.3f}s"
                else:
                    latency_color = (255, 165, 0)  # Orange/yellow
                    latency_str = f"Lat: {latency:.3f}s"
                    
                self._last_latency = latency
                latency_surface = font_normal.render(latency_str, True, latency_color)
                
                # Mode indicator
                mode_str = f"{'Perf' if self.PERFORMANCE_MODE else 'Quality'}"
                mode_surface = font_normal.render(mode_str, True, self.TEXT_COLOR)
                
                # Sample rate if available
                if self.sampling_rate:
                    sr_str = f"SR: {self.sampling_rate} Hz"
                    sr_surface = font_normal.render(sr_str, True, self.TEXT_COLOR)
                else:
                    sr_surface = None
                
                # Position all info elements in a single row
                info_x = margin_left
                spacing = 15
                
                # Display real time
                screen.blit(real_time_surface, (info_x, info_row_y))
                info_x += real_time_surface.get_width() + spacing
                
                # Display latency
                screen.blit(latency_surface, (info_x, info_row_y))
                info_x += latency_surface.get_width() + spacing
                
                # Display sample rate if available
                if sr_surface:
                    screen.blit(sr_surface, (info_x, info_row_y))
                    info_x += sr_surface.get_width() + spacing
                
                # Display mode
                screen.blit(mode_surface, (info_x, info_row_y))
                
                # Draw each signal in its own subplot
                for i, signal_type in enumerate(active_signals):
                    # Get signal data
                    x_arr = self._latest_multi_data[signal_type]['x']
                    y_arr = self._latest_multi_data[signal_type]['y']
                    
                    # Skip if no data
                    if len(x_arr) == 0:
                        continue
                    
                    # Calculate vertical position for this subplot
                    y_offset = margin_top + i * (plot_h + divider_height)
                    
                    # Draw signal label on top right instead of left
                    signal_label = font_signal.render(signal_type, True, self.signal_colors[signal_type])
                    screen.blit(signal_label, (w - margin_right - signal_label.get_width() - 10, y_offset + 5))
                    
                    # Normalize x values to plot width (using global x_min/x_max)
                    if x_range == 0:
                        x_norm = np.full_like(x_arr, margin_left)
                    else:
                        x_norm = margin_left + (x_arr - x_min) * plot_w / x_range
                    
                    # Y axis scaling (specific to this signal)
                    y_min = float(np.min(y_arr))
                    y_max = float(np.max(y_arr))
                    
                    # Ensure y_min and y_max are distinct
                    if y_max <= y_min:
                        y_max = y_min + 0.1
                    
                    y_range = y_max - y_min
                    
                    # Draw y axis for this subplot
                    pygame.draw.line(screen, line_color, 
                                    (margin_left, y_offset), 
                                    (margin_left, y_offset + plot_h), 
                                    line_thickness)
                    
                    # Draw y-axis labels for this subplot
                    if not self.PERFORMANCE_MODE:
                        y_min_label = f"{y_min:.2f}"
                        y_max_label = f"{y_max:.2f}"
                        
                        y_min_surf = font_normal.render(y_min_label, True, self.TEXT_COLOR)
                        y_max_surf = font_normal.render(y_max_label, True, self.TEXT_COLOR)
                        
                        # Position at top and bottom of the y-axis for this subplot
                        screen.blit(y_min_surf, (margin_left - y_min_surf.get_width() - 5, 
                                               y_offset + plot_h - 10))
                        screen.blit(y_max_surf, (margin_left - y_max_surf.get_width() - 5, 
                                               y_offset + 5))
                    
                    # Draw horizontal gridlines
                    if not self.PERFORMANCE_MODE:
                        grid_color = self.GRID_COLOR
                        num_grids = 4
                        for j in range(1, num_grids):
                            y_pos = y_offset + plot_h - (j * plot_h / num_grids)
                            pygame.draw.line(screen, grid_color, 
                                           (margin_left + 1, y_pos), 
                                           (w - margin_right, y_pos), 
                                           1)
                    
                    # Normalize y values specific to this subplot
                    if y_range == 0:
                        y_norm = np.full_like(y_arr, y_offset + plot_h/2)
                    else:
                        # Scale to this subplot's height and position
                        y_norm = y_offset + plot_h - (y_arr - y_min) * plot_h / y_range
                    
                    # Get color for this signal
                    line_color = self.signal_colors.get(signal_type, self.signal_colors["DEFAULT"])
                    
                    # Apply intelligent downsampling
                    if self.SMART_DOWNSAMPLE and self.enable_downsampling:
                        self.signal_type = signal_type  # Set for downsampling algorithm
                        x_arr, y_arr = self._smart_downsample(x_arr, y_arr)
                        # Recompute normalized coordinates after downsampling
                        if x_range == 0:
                            x_norm = np.full_like(x_arr, margin_left)
                        else:
                            x_norm = margin_left + (x_arr - x_min) * plot_w / x_range
                        if y_range == 0:
                            y_norm = np.full_like(y_arr, y_offset + plot_h/2)
                        else:
                            y_norm = y_offset + plot_h - (y_arr - y_min) * plot_h / y_range
                    
                    # Draw signal
                    if len(x_norm) > 1:
                        # Filter out any NaN values before creating points
                        valid_indices = ~(np.isnan(x_norm) | np.isnan(y_norm))
                        if np.any(valid_indices):
                            x_filtered = x_norm[valid_indices]
                            y_filtered = y_norm[valid_indices]
                            
                            # Only create points from valid coordinates
                            pts = [(int(x), int(y)) for x, y in zip(x_filtered, y_filtered)]
                            
                            thickness = 1 if self.PERFORMANCE_MODE else self.LINE_THICKNESS
                            if pts:  # Only draw if we have valid points
                                if self.PERFORMANCE_MODE:
                                    pygame.draw.lines(screen, line_color, False, pts, thickness)
                                else:
                                    self._draw_smooth_line(screen, line_color, pts, thickness)
                    
                    elif len(x_norm) == 1:
                        # Check if single point is valid before drawing
                        if not np.isnan(x_norm[0]) and not np.isnan(y_norm[0]):
                            pygame.draw.circle(screen, line_color, (int(x_norm[0]), int(y_norm[0])), 2)
                
                # Update display
                pygame.display.flip()
            
            # Maintain FPS
            if self.FPS_CAP_ENABLED:
                clock.tick(self.FPS)
        
        print("Multi-signal plot thread terminated")

    def _draw_smooth_line(self, surface, color, points, thickness=1):
        """Draw a smoother line by using anti-aliasing and optional interpolation"""
        if len(points) < 2:
            return
            
        # For thick lines, use aalines with blend
        if thickness <= 1:
            pygame.draw.aalines(surface, color, False, points)
        else:
            # For thicker lines, first draw a regular line
            pygame.draw.lines(surface, color, False, points, thickness)
            # Then overlay an anti-aliased line for smoothing the edges
            pygame.draw.aalines(surface, color, False, points)

    def resize_window(self, width, height):
        """Request window resize on next render cycle"""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self._resize_needed = True
            print(f"Window resize requested to {width}x{height}")

# Node implementation of PygamePlot for ComfyUI
class PygamePlotNode:
    """ComfyUI node for PygamePlot"""
    
    def __init__(self):
        self.plot = PygamePlot()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("LIST", {}),
                "y": ("LIST", {}),
                "as_points": ("BOOLEAN", {"default": False}),
            }
        }
    CATEGORY = "Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True

    def plot(self, x, y, as_points):
        self.plot.plot(x, y, as_points)
        return ()