import threading
import numpy as np
import time
import queue
import weakref
from collections import deque

# Try importing PyGame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available, plotting will be limited")

# Primary Plot Class - PygamePlot
class PygamePlot:
    _window = None
    _screen = None
    _lock = threading.Lock()
    _start_time = None
    _instances = []  # Track all plot instances
    _dirty_regions = []  # Track regions that need updates
    _font_cache = {}  # Cache for frequently used fonts
    _surf_cache = {}  # Cache for reusable surfaces

    FPS = 60
    FPS_CAP_ENABLED = True
    # Default window dimensions
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    # New optimization flags
    PERFORMANCE_MODE = False
    SMART_DOWNSAMPLE = False  # Default to OFF for downsampling
    LINE_THICKNESS = 1

    def __init__(self, width=None, height=None, performance_mode=False):
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
        # Set dimensions - use defaults if not provided
        self.width = width if width is not None else self.DEFAULT_WIDTH
        self.height = height if height is not None else self.DEFAULT_HEIGHT
        # Set performance mode
        self.PERFORMANCE_MODE = performance_mode
        # Signal type for smart downsampling
        self.signal_type = None
        # Add a new flag for downsampling
        self.enable_downsampling = False  # Default to disabled
        
        # Register this instance
        with PygamePlot._lock:
            PygamePlot._instances.append(weakref.ref(self))
    
    def __del__(self):
        self._stop_event.set()

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

    @staticmethod
    def _get_font(size, name=None):
        """Get a cached font to avoid recreation"""
        if not PYGAME_AVAILABLE:
            return None
            
        key = (name, size)
        if key not in PygamePlot._font_cache:
            PygamePlot._font_cache[key] = pygame.font.SysFont(name, size)
        return PygamePlot._font_cache[key]

    def _smart_downsample(self, x_arr, y_arr):
        """
        Apply signal-specific downsampling optimized for low-noise signals.
        Only applies if enable_downsampling is True.
        """
        # Skip downsampling if it's disabled or if data is small enough
        if not self.enable_downsampling or len(x_arr) <= 1000:
            return x_arr, y_arr
        
        # Get time duration of the signal
        time_span = x_arr[-1] - x_arr[0]
        if time_span <= 0:
            return x_arr, y_arr  # Avoid division by zero
        
        # Calculate current effective sampling rate
        current_sampling_rate = len(x_arr) / time_span
        
        # For low-noise signals, we need to be more careful with downsampling
        # Calculate the signal variance to estimate noise level
        y_variance = np.var(y_arr)
        
        # Define minimum required sampling rates based on signal type
        # and adjust for low-noise conditions
        if self.signal_type == "ECG":
            # ECG with QRS complexes needs higher resolution
            # For low-noise ECG, we can use lower rates but still preserve peaks
            base_rate = 100.0  # Base rate for ECG
            
            # For very clean signals, we can be more aggressive
            if y_variance < 0.01:  # Low variance suggests clean signal
                return self._peak_preserving_downsample(x_arr, y_arr, base_rate)
            else:
                # Higher variance needs more careful sampling
                return self._peak_preserving_downsample(x_arr, y_arr, base_rate * 2)
                
        elif self.signal_type == "EDA":
            # EDA is slow-changing, can use lower rates
            if current_sampling_rate > 16.0:  # Allow more downsampling for clean EDA
                downsample_factor = int(current_sampling_rate / 8.0)
                return x_arr[::downsample_factor], y_arr[::downsample_factor]
            return x_arr, y_arr
            
        elif self.signal_type == "RR":
            # Respiratory rate is very slow
            if current_sampling_rate > 10.0:
                downsample_factor = int(current_sampling_rate / 5.0)
                return x_arr[::downsample_factor], y_arr[::downsample_factor]
            return x_arr, y_arr
            
        else:
            # Default case - simple downsampling based on current rate
            if current_sampling_rate > 50.0:
                downsample_factor = max(1, int(current_sampling_rate / 50.0))
                return x_arr[::downsample_factor], y_arr[::downsample_factor]
            return x_arr, y_arr

    def _peak_preserving_downsample(self, x_arr, y_arr, target_rate):
        """
        Downsample while preserving important peaks in the signal.
        Especially useful for ECG where QRS complexes need to be preserved.
        """
        if len(x_arr) <= 2:
            return x_arr, y_arr
            
        time_span = x_arr[-1] - x_arr[0]
        if time_span <= 0:
            return x_arr, y_arr
            
        current_rate = len(x_arr) / time_span
        if current_rate <= target_rate * 1.5:
            return x_arr, y_arr  # Already close to target rate
            
        # Calculate basic downsampling factor
        basic_factor = int(current_rate / target_rate)
        if basic_factor <= 1:
            return x_arr, y_arr
            
        # Find peaks in the signal
        # Simple peak detection using local maxima/minima
        peaks = []
        for i in range(1, len(y_arr)-1):
            # Look for local maxima and minima
            if (y_arr[i] > y_arr[i-1] and y_arr[i] > y_arr[i+1]) or \
               (y_arr[i] < y_arr[i-1] and y_arr[i] < y_arr[i+1]):
                peaks.append(i)
                
        # If no peaks found, use regular downsampling
        if not peaks:
            return x_arr[::basic_factor], y_arr[::basic_factor]
            
        # Create mask for points to keep
        mask = np.zeros(len(x_arr), dtype=bool)
        
        # Keep every nth point as base sampling
        mask[::basic_factor] = True
        
        # Always keep the first and last points
        mask[0] = mask[-1] = True
        
        # Keep points around peaks
        for peak in peaks:
            # Keep the peak and some points around it
            start_idx = max(0, peak - 2)
            end_idx = min(len(mask), peak + 3)
            mask[start_idx:end_idx] = True
            
        # Apply mask to get downsampled arrays
        return x_arr[mask], y_arr[mask]

    def _plot_loop(self):
        """Main plotting loop for pygame visualization"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot create plot")
            return
            
        # Reset window if it was previously closed by user
        with PygamePlot._lock:
            if self._window_closed_externally:
                PygamePlot._window = None
                PygamePlot._screen = None
                self._window_closed_externally = False
                print("Resetting pygame window after previous close")
            
        # Window dimensions - use instance values
        w, h = self.width, self.height
        
        # Adjust margins for smaller windows
        margin_left = int(w * 0.09)  # 9% of width
        margin_right = int(w * 0.05)  # 5% of width
        margin_top = int(h * 0.08)    # 8% of height
        margin_bottom = int(h * 0.12)  # 12% of height
        
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
            
            # Performance mode uses simpler colors
            if self.PERFORMANCE_MODE:
                bg_color = (0, 0, 0)  # Black background for better performance
                line_color = (100, 100, 100)  # Gray lines
            else:
                bg_color = (30, 30, 30)  # Dark gray
                line_color = (255, 255, 255)  # White
                
            bg_surface.fill(bg_color)
            line_thickness = 1 if self.PERFORMANCE_MODE else 2
            pygame.draw.line(bg_surface, line_color, (margin_left, h - margin_bottom), 
                            (w - margin_right, h - margin_bottom), line_thickness)
            pygame.draw.line(bg_surface, line_color, (margin_left, h - margin_bottom), 
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
        font_size_normal = max(14, int(h * 0.045))
        font_size_bold = max(16, int(h * 0.05))
        font_normal = self._get_font(font_size_normal)
        font_bold = self._get_font(font_size_bold)
        
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
                    x_min_surf = font_normal.render(x_min_label, True, (200, 200, 200))
                    x_max_surf = font_normal.render(x_max_label, True, (200, 200, 200))
                    
                    # Draw min/max labels - position at actual axis endpoints
                    screen.blit(x_min_surf, (margin_left - x_min_surf.get_width()//2, h - margin_bottom + 15))
                    screen.blit(x_max_surf, (w - margin_right - x_max_surf.get_width()//2, h - margin_bottom + 15))
                    
                    # Render axis labels if needed
                    labels_key = f"labels_{font_size_normal}"
                    if f'{labels_key}_x' not in PygamePlot._surf_cache:
                        PygamePlot._surf_cache[f'{labels_key}_x'] = font_normal.render(x_label, True, (200, 200, 200))
                    if f'{labels_key}_y' not in PygamePlot._surf_cache:
                        PygamePlot._surf_cache[f'{labels_key}_y'] = font_normal.render(y_label, True, (200, 200, 200))
                    
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
                        
                        y_min_surf = font_normal.render(y_min_label, True, (200, 200, 200))
                        y_max_surf = font_normal.render(y_max_label, True, (200, 200, 200))
                        
                        # Position at actual axis endpoints
                        screen.blit(y_min_surf, (margin_left - y_min_surf.get_width() - 5, h - margin_bottom - 10))
                        screen.blit(y_max_surf, (margin_left - y_max_surf.get_width() - 5, margin_top))
                    
                    # Set line color based on signal type or performance mode
                    if self.PERFORMANCE_MODE:
                        # Fast, bright colors
                        line_color = (0, 255, 0)  # Green for performance
                    else:
                        if self.signal_type == "ECG":
                            line_color = (0, 0, 255)  # Blue for ECG
                        elif self.signal_type == "EDA":
                            line_color = (0, 255, 0)  # Green for EDA
                        elif self.signal_type == "RR":
                            line_color = (255, 165, 0)  # Orange for RR
                        else:
                            line_color = (0, 0, 255)  # Default blue
                    
                    # Draw signal - use numpy operations as much as possible
                    if as_points:
                        # Batch the points for drawing - use smaller circles in performance mode
                        radius = 1 if self.PERFORMANCE_MODE else 2
                        point_coords = list(zip(x_norm, y_norm))
                        for pt in point_coords:
                            pygame.draw.circle(screen, line_color, (int(pt[0]), int(pt[1])), radius)
                    else:
                        # Draw lines more efficiently
                        if len(x_norm) > 1:
                            # Always use anti-aliased lines for smoother appearance
                            thickness = 1 if self.PERFORMANCE_MODE else self.LINE_THICKNESS
                            
                            # Create list of points with integer coordinates
                            pts = [(int(x), int(y)) for x, y in zip(x_norm, y_norm)]
                            
                            # Draw smoother lines unless in performance mode
                            if self.PERFORMANCE_MODE:
                                # In performance mode, use regular lines (faster)
                                pygame.draw.lines(screen, line_color, False, pts, thickness)
                            else:
                                # Apply smoothing for better visual quality
                                self._draw_smooth_line(screen, line_color, pts, thickness)
                                
                        elif len(x_norm) == 1:
                            pygame.draw.circle(screen, line_color, (int(x_norm[0]), int(y_norm[0])), 2)

                # RESTORED: Always draw timing info and latency info
                # Node time information
                node_time_str = time.strftime("Node Time: %Y-%m-%d %H:%M:%S", time.localtime(PygamePlot._start_time))
                node_time_key = f"node_time_{int(now) % 10}_{font_size_bold}"  # Update every 10 seconds
                if node_time_key not in PygamePlot._surf_cache:
                    PygamePlot._surf_cache[node_time_key] = font_bold.render(node_time_str, True, (255, 255, 0))
                screen.blit(PygamePlot._surf_cache[node_time_key], (10, 10))

                # Real time tracking - always base on clock time
                real_elapsed = now - self._real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"Real Time: {real_minutes:02}:{real_seconds:02}"
                real_time_surface = font_bold.render(real_time_str, True, (255, 255, 0))
                screen.blit(real_time_surface, (10, 10 + font_size_bold + 5))
                
                # Display sampling rate information
                if self.sampling_rate:
                    sr_str = f"Sample Rate: {self.sampling_rate} Hz"
                    sr_surface = font_normal.render(sr_str, True, (200, 200, 200))
                    screen.blit(sr_surface, (10, 10 + (font_size_bold + 5) * 3))
                
                # RESTORED: Latency calculation - critical for signal analysis
                if len(x_arr) > 0:
                    # Latency is how far behind real-time the latest data point is
                    latency = real_elapsed - x_max
                    
                    # Only consider latency problematic if >1s or very unstable
                    is_stable = abs(latency - self._last_latency) < 0.5 if hasattr(self, '_last_latency') else False
                    is_acceptable = latency <= 1
                    
                    if is_stable and is_acceptable:
                        # Green for stable, acceptable latency
                        latency_color = (100, 255, 100)
                        latency_str = f"Latency: {latency:.3f}s (stable)"
                    elif is_stable and not is_acceptable:
                        # Orange for stable but high latency
                        latency_color = (255, 165, 0)
                        latency_str = f"Latency: {latency:.3f}s (high)"
                    elif not is_stable and is_acceptable:
                        # Yellow for changing but acceptable latency
                        latency_color = (255, 255, 100)
                        latency_str = f"Latency: {latency:.3f}s"
                    else:
                        # Red for unstable and high latency
                        latency_color = (255, 100, 100)
                        latency_str = f"Latency: {latency:.3f}s (!)"

                    self._last_latency = latency  # Store for next comparison
                    
                    latency_surface = font_bold.render(latency_str, True, latency_color)
                    screen.blit(latency_surface, (10, 10 + (font_size_bold + 5) * 2))
                    
                    # Add data points counter
                    points_str = f"Points: {len(x_arr)} ({'performance' if self.PERFORMANCE_MODE else 'quality'} mode)"
                    points_surface = font_normal.render(points_str, True, (200, 200, 200))
                    screen.blit(points_surface, (10, 10 + (font_size_bold + 5) * 4))

                # Optimized update - only update the screen once per frame
                pygame.display.flip()
            
            # Maintain FPS
            if self.FPS_CAP_ENABLED:
                clock.tick(self.FPS)
        
        print("Plot thread terminated")

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