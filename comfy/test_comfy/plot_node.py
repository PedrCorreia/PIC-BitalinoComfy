import threading
import numpy as np
import time
import queue
import weakref

class NodeKO:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default":"Hey Hey!"}),
                "text2": ("STRING", {"forceInput": True}),				
            },
        }

    RETURN_TYPES = ("STRING","INT", "FLOAT", 'LATENT', "CONDITIONING", "IMAGE", "MODEL")
    RETURN_NAMES = ("TxtO", "IntO", "FloatO", "Latent output. Really cool, huh?", "A condition" , "Our image." , "Mo mo modell!!!")

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "nodeKO"

    def test(self):
        return ()

class PygamePlotNode:
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

    def __init__(self):
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
        
        # Register this instance
        with PygamePlotNode._lock:
            PygamePlotNode._instances.append(weakref.ref(self))
    
    def __del__(self):
        self._stop_event.set()

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
        # Reset closed state when plot is called
        self._closed = False
        
        # Compute a hash of the data to detect changes
        data_len = len(x)
        data_hash = hash((data_len, 
                         hash(x[0]) if data_len > 0 else 0, 
                         hash(x[-1]) if data_len > 0 else 0,
                         hash(y[0]) if data_len > 0 else 0,
                         hash(y[-1]) if data_len > 0 else 0,
                         as_points))
        
        # Only update if data actually changed
        if self._data_hash != data_hash:
            self._data_hash = data_hash
            self._latest_data = (np.array(x, dtype=np.float32), 
                                np.array(y, dtype=np.float32), 
                                as_points)
            self._new_data.set()
            
        if self._plot_thread is None or not self._plot_thread.is_alive():
            print("Starting new plot thread")
            self._stop_event.clear()
            self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
            self._plot_thread.start()
        return ()
    
    @staticmethod
    def _get_font(size, name=None):
        """Get a cached font to avoid recreation"""
        key = (name, size)
        if key not in PygamePlotNode._font_cache:
            import pygame
            PygamePlotNode._font_cache[key] = pygame.font.SysFont(name, size)
        return PygamePlotNode._font_cache[key]

    def _plot_loop(self):
        import pygame
        import time as _time
        
        # Window dimensions
        w, h = 640, 480
        margin_left, margin_right, margin_top, margin_bottom = 60, 30, 40, 60
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom
        
        # Use double buffered hardware acceleration
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
        
        # Initialize Pygame only once
        with PygamePlotNode._lock:
            if PygamePlotNode._window is None:
                print("Initializing pygame window")
                pygame.init()
                PygamePlotNode._screen = pygame.display.set_mode((w, h), flags, vsync=1)
                pygame.display.set_caption("Pygame Plot (Optimized)")
                PygamePlotNode._window = True
                PygamePlotNode._start_time = _time.time()
            else:
                print("Using existing pygame window")
        
        screen = PygamePlotNode._screen
        
        # Create clock for consistent framerate
        clock = pygame.time.Clock()
        
        # Create cached background surface only once
        if 'bg' not in PygamePlotNode._surf_cache:
            bg_surface = pygame.Surface((w, h)).convert()
            bg_surface.fill((30, 30, 30))
            pygame.draw.line(bg_surface, (255,255,255), (margin_left, h - margin_bottom), (w - margin_right, h - margin_bottom), 2)
            pygame.draw.line(bg_surface, (255,255,255), (margin_left, h - margin_bottom), (margin_left, margin_top), 2)
            PygamePlotNode._surf_cache['bg'] = bg_surface
        
        # Create cached buffer surfaces for drawing
        if 'plot_buffer' not in PygamePlotNode._surf_cache:
            plot_buffer = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            PygamePlotNode._surf_cache['plot_buffer'] = plot_buffer
        
        bg_surface = PygamePlotNode._surf_cache['bg']
        plot_buffer = PygamePlotNode._surf_cache['plot_buffer']
        
        # Reusable objects
        font_normal = self._get_font(22)
        font_bold = self._get_font(24)
        update_rect = pygame.Rect(0, 0, w, h)
        
        # Main loop
        running = True
        min_interval = 1.0 / self.FPS if self.FPS_CAP_ENABLED else 0
        last_x_max = 0  # Track last max x value to detect time jumps
        
        while running and not self._stop_event.is_set():
            # Wait for new data with a timeout
            self._new_data.wait(timeout=0.001)
            
            # Check FPS cap
            now = _time.time()
            if self.FPS_CAP_ENABLED and (now - self._last_draw_time) < min_interval:
                clock.tick(self.FPS)
                continue
            
            # Process new data if available
            if self._new_data.is_set():
                x_arr, y_arr, as_points = self._latest_data
                self._new_data.clear()
                self._last_draw_time = now
                needs_redraw = True
            else:
                # Skip redrawing if no new data
                needs_redraw = False
                
            if needs_redraw:
                # Start with a clean slate by copying the background
                screen.blit(bg_surface, (0, 0))
                plot_buffer.fill((0, 0, 0, 0))  # Clear with transparency
                
                # Calculate plotting parameters with corrections for real-time display
                if len(x_arr) > 0:
                    x_min = float(x_arr.min())
                    x_max = float(x_arr.max())
                    
                    # Check for time jumps (could indicate reset or missed data)
                    if x_max < last_x_max:
                        # Time has reset, update real_time_start to resync
                        self._real_time_start = now - x_max
                        print(f"Time reset detected: adjusting real_time")
                    last_x_max = x_max
                else:
                    x_min, x_max = 0.0, 1.0

                x_min_rounded = round(x_min)
                x_max_rounded = round(x_max)
                show_labels = x_max >= 1.0

                # Axis labels (reuse surfaces with caching)
                if show_labels:
                    # X axis labels
                    label_key = f"x_min_{x_min_rounded}"
                    if label_key not in PygamePlotNode._surf_cache:
                        PygamePlotNode._surf_cache[label_key] = font_normal.render(
                            f"{x_min_rounded:.0f}s", True, (200, 200, 200))
                    screen.blit(PygamePlotNode._surf_cache[label_key], (margin_left, h - margin_bottom + 8))
                    
                    label_key = f"x_max_{x_max_rounded}"
                    if label_key not in PygamePlotNode._surf_cache:
                        PygamePlotNode._surf_cache[label_key] = font_normal.render(
                            f"{x_max_rounded:.0f}s", True, (200, 200, 200))
                    x_label_max = PygamePlotNode._surf_cache[label_key]
                    screen.blit(x_label_max, (w - margin_right - x_label_max.get_width(), h - margin_bottom + 8))
                    
                    # Y axis labels - more dynamic, less cache opportunity
                    if len(y_arr) > 0:
                        y_min = float(y_arr.min())
                        y_max = float(y_arr.max())
                        y_label_min = font_normal.render(f"{y_min:.2f}", True, (200, 200, 200))
                        y_label_max = font_normal.render(f"{y_max:.2f}", True, (200, 200, 200))
                        screen.blit(y_label_min, (margin_left - y_label_min.get_width() - 8, h - margin_bottom - y_label_min.get_height()//2))
                        screen.blit(y_label_max, (margin_left - y_label_max.get_width() - 8, margin_top - y_label_max.get_height()//2))
                
                # Plot the data only if we have points
                if len(x_arr) > 0:
                    # Optimized vectorized calculation
                    if x_max - x_min == 0:
                        x_norm = np.full_like(x_arr, margin_left + plot_w // 2)
                    else:
                        x_norm = ((x_arr - x_min) / (x_max - x_min) * plot_w + margin_left).astype(np.int32)
                    
                    y_min = float(y_arr.min()) if len(y_arr) > 0 else 0.0
                    y_max = float(y_arr.max()) if len(y_arr) > 0 else 1.0
                    
                    if y_max - y_min == 0:
                        y_norm = np.full_like(y_arr, margin_top + plot_h // 2)
                    else:
                        y_norm = (h - margin_bottom - ((y_arr - y_min) / (y_max - y_min) * plot_h)).astype(np.int32)
                    
                    # Draw signal - use numpy operations as much as possible
                    if as_points:
                        # Batch the points for drawing
                        point_coords = list(zip(x_norm, y_norm))
                        for pt in point_coords:
                            pygame.draw.circle(screen, (0,0,255), pt, 2)
                    else:
                        # Draw lines more efficiently
                        if len(x_norm) > 1:
                            # Use pygame.draw.aalines for smoother lines when not too many points
                            if len(x_norm) <= 10000:  # Arbitrary threshold
                                pts = list(zip(x_norm, y_norm))
                                pygame.draw.aalines(screen, (0,0,255), False, pts, 2)
                            else:
                                # For large datasets, downsample
                                step = max(1, len(x_norm) // 5000)
                                pts = list(zip(x_norm[::step], y_norm[::step]))
                                pygame.draw.aalines(screen, (0,0,255), False, pts, 2)
                        elif len(x_norm) == 1:
                            pygame.draw.circle(screen, (0,0,255), (x_norm[0], y_norm[0]), 2)

                # Draw timing info
                node_time_str = time.strftime("Node Time: %Y-%m-%d %H:%M:%S", time.localtime(PygamePlotNode._start_time))
                node_time_key = f"node_time_{int(now) % 10}"  # Update every 10 seconds
                if node_time_key not in PygamePlotNode._surf_cache:
                    PygamePlotNode._surf_cache[node_time_key] = font_bold.render(node_time_str, True, (255, 255, 0))
                screen.blit(PygamePlotNode._surf_cache[node_time_key], (10, 10))

                # Real time tracking - always base on clock time
                real_elapsed = now - self._real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"Real Time: {real_minutes:02}:{real_seconds:02}"
                real_time_surface = font_bold.render(real_time_str, True, (255, 255, 0))
                screen.blit(real_time_surface, (10, 35))
                
                # Display sampling rate information
                if self.sampling_rate:
                    sr_str = f"Sample Rate: {self.sampling_rate} Hz"
                    sr_surface = font_normal.render(sr_str, True, (200, 200, 200))
                    screen.blit(sr_surface, (10, 110))
                
                # Latency calculation - use real elapsed time for accurate calculation
                if len(x_arr) > 0:
                    # Latency is how far behind real-time the latest data point is
                    latency = real_elapsed - x_max
                    
                    # Only consider latency problematic if >1s or very unstable
                    is_stable = abs(latency - self._last_latency) < 0.1 if hasattr(self, '_last_latency') else False
                    is_acceptable = latency <= 0.5
                    
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
                    screen.blit(latency_surface, (10, 60))
                    
                    # Add data points counter
                    points_str = f"Points: {len(x_arr)}"
                    points_surface = font_normal.render(points_str, True, (200, 200, 200))
                    screen.blit(points_surface, (10, 85))

                # Optimized update - only update the screen once per frame
                pygame.display.flip()

            # Handle events efficiently
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self._closed = True  # Mark as closed
                    with PygamePlotNode._lock:
                        # Only clear global state if this is the last window
                        if len([p for p in PygamePlotNode._instances if p() and not getattr(p(), '_closed', False)]) == 0:
                            print("Closing last pygame window - clearing global state")
                            PygamePlotNode._window = None
                            PygamePlotNode._screen = None
                            pygame.quit()
                        else:
                            print("Closing window but others remain")
                    break
            
            # Maintain framerate
            clock.tick(self.FPS)

class PyQtGraphPlotNode:
    _win = None
    _app = None
    _plot_item = None
    _lock = threading.Lock()

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
        def plot_with_pyqtgraph(x, y, as_points):
            import sys
            from PyQt6.QtWidgets import QApplication
            import pyqtgraph as pg
            with PyQtGraphPlotNode._lock:
                if PyQtGraphPlotNode._app is None:
                    PyQtGraphPlotNode._app = QApplication.instance()
                    if PyQtGraphPlotNode._app is None:
                        PyQtGraphPlotNode._app = QApplication(sys.argv)
                if PyQtGraphPlotNode._win is None:
                    PyQtGraphPlotNode._win = pg.plot(title="PyQtGraph Plot")
                    PyQtGraphPlotNode._plot_item = PyQtGraphPlotNode._win.plotItem
                else:
                    PyQtGraphPlotNode._plot_item.clear()
                if as_points:
                    PyQtGraphPlotNode._plot_item.plot(x, y, pen=None, symbol='o', symbolBrush='b')
                else:
                    PyQtGraphPlotNode._plot_item.plot(x, y, pen=pg.mkPen('b', width=2))
                PyQtGraphPlotNode._win.show()
            if hasattr(PyQtGraphPlotNode._app, 'exec'):
                try:
                    PyQtGraphPlotNode._app.exec()
                except Exception:
                    pass
        threading.Thread(target=plot_with_pyqtgraph, args=(x, y, as_points), daemon=True).start()
        return ()

class OpenCVPlotNode:
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
        def plot_with_opencv(x, y, as_points):
            import cv2
            w, h = 640, 480
            img = np.zeros((h, w, 3), dtype=np.uint8)
            x = np.array(x)
            y = np.array(y)
            x_norm = ((x - x.min()) / (x.max() - x.min()) * (w - 40) + 20).astype(int)
            y_norm = (h - ((y - y.min()) / (y.max() - y.min()) * (h - 40) + 20)).astype(int)
            cv2.line(img, (20, h-20), (w-20, h-20), (255,255,255), 2)
            cv2.line(img, (20, h-20), (20, 20), (255,255,255), 2)
            if as_points:
                for i in range(len(x)):
                    cv2.circle(img, (x_norm[i], y_norm[i]), 2, (255,0,0), -1)
            else:
                pts = np.column_stack((x_norm, y_norm))
                cv2.polylines(img, [pts], isClosed=False, color=(255,0,0), thickness=2)
            cv2.imshow("OpenCV Plot", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        threading.Thread(target=plot_with_opencv, args=(x, y, as_points), daemon=True).start()
        return ()

class VispyPlotNode:
    _canvas = None
    _view = None
    _lock = threading.Lock()
    _start_time = None
    _bg_drawn = False
    _timer = None
    _data_queue = queue.Queue()
    _real_time_start = None
    FPS = 60

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
        # Put new data in the queue for the main thread to process
        VispyPlotNode._data_queue.put((list(x), list(y), as_points))
        # If not started, start the vispy window and timer in the main thread
        if VispyPlotNode._canvas is None:
            self._start_vispy_mainloop()
        return ()

    def _start_vispy_mainloop(self):
        from vispy import app, scene
        import numpy as np
        import time as _time

        w, h = 640, 480
        margin_left, margin_right, margin_top, margin_bottom = 60, 30, 40, 60
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        VispyPlotNode._canvas = scene.SceneCanvas(keys='interactive', show=True, title="Vispy Plot", size=(w, h))
        VispyPlotNode._view = VispyPlotNode._canvas.central_widget.add_view()
        VispyPlotNode._canvas.events.close.connect(self._on_close)
        VispyPlotNode._start_time = _time.time()
        VispyPlotNode._bg_drawn = False
        VispyPlotNode._real_time_start = None

        view = VispyPlotNode._view
        canvas = VispyPlotNode._canvas

        # Draw axes/background only once
        if not VispyPlotNode._bg_drawn:
            from vispy.color import Color
            # X axis
            scene.visuals.Line(np.array([
                [margin_left, h - margin_bottom],
                [w - margin_right, h - margin_bottom]
            ]), color=Color('white'), width=2, parent=view, name='axes')
            # Y axis
            scene.visuals.Line(np.array([
                [margin_left, h - margin_bottom],
                [margin_left, margin_top]
            ]), color=Color('white'), width=2, parent=view, name='axes')
            VispyPlotNode._bg_drawn = True

        # Timer callback to update plot from queue
        def update_plot(event):
            import numpy as np
            import time as _time
            # Remove previous plot visuals except axes
            for v in list(view.children):
                if hasattr(v, 'name') and v.name == 'axes':
                    continue
                v.parent = None

            try:
                x, y, as_points = VispyPlotNode._data_queue.get_nowait()
            except queue.Empty:
                return  # No new data

            x_arr = np.array(x)
            y_arr = np.array(y)
            if len(x_arr) > 0:
                x_min = float(x_arr[0])
                x_max = float(x_arr[-1])
            else:
                x_min, x_max = 0.0, 1.0
            x_min_rounded = round(x_min)
            x_max_rounded = round(x_max)
            show_labels = x_max >= 1.0

            # Axis labels
            if show_labels:
                y_min = float(np.min(y_arr)) if len(y_arr) > 0 else 0.0
                y_max = float(np.max(y_arr)) if len(y_arr) > 0 else 1.0
                scene.visuals.Text(
                    f"{x_min_rounded:.0f}s", color='white', font_size=14,
                    pos=(margin_left, h - margin_bottom + 18), parent=view)
                scene.visuals.Text(
                    f"{x_max_rounded:.0f}s", color='white', font_size=14,
                    pos=(w - margin_right - 20, h - margin_bottom + 18), parent=view)
                scene.visuals.Text(
                    f"{y_min:.2f}", color='white', font_size=14,
                    pos=(margin_left - 35, h - margin_bottom), parent=view)
                scene.visuals.Text(
                    f"{y_max:.2f}", color='white', font_size=14,
                    pos=(margin_left - 35, margin_top), parent=view)

            # Vectorized normalization
            if x_max - x_min == 0:
                x_norm = np.full_like(x_arr, margin_left + plot_w // 2)
            else:
                x_norm = ((x_arr - x_min) / (x_max - x_min) * plot_w + margin_left)
            if len(y_arr) > 0:
                y_min = float(np.min(y_arr))
                y_max = float(np.max(y_arr))
            else:
                y_min, y_max = 0.0, 1.0
            if y_max - y_min == 0:
                y_norm = np.full_like(y_arr, margin_top + plot_h // 2)
            else:
                y_norm = (h - margin_bottom - ((y_arr - y_min) / (y_max - y_min) * plot_h))

            # Draw signal
            if as_points:
                scene.visuals.Markers(pos=np.column_stack((x_norm, y_norm)), face_color='blue', size=6, parent=view)
            else:
                if len(x_norm) > 1:
                    pts = np.column_stack((x_norm, y_norm))
                    scene.visuals.Line(pts, color='blue', width=2, parent=view)
                elif len(x_norm) == 1:
                    scene.visuals.Markers(pos=np.column_stack((x_norm, y_norm)), face_color='blue', size=6, parent=view)

            # Node Time
            node_time_str = time.strftime("Node Time: %Y-%m-%d %H:%M:%S", time.localtime(VispyPlotNode._start_time))
            scene.visuals.Text(
                node_time_str, color='yellow', font_size=16, pos=(10, 20), parent=view)

            # Real Time: start after first frame with signal (x_arr not empty)
            if VispyPlotNode._real_time_start is None and len(x_arr) > 0:
                VispyPlotNode._real_time_start = _time.time()
            if VispyPlotNode._real_time_start is not None:
                real_elapsed = _time.time() - VispyPlotNode._real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"Real Time: {real_minutes:02}:{real_seconds:02}"
                scene.visuals.Text(
                    real_time_str, color='yellow', font_size=16, pos=(10, 45), parent=view)
            else:
                scene.visuals.Text(
                    "Real Time: --:--", color='yellow', font_size=16, pos=(10, 45), parent=view)

            # Latency: difference between real time and last x
            if VispyPlotNode._real_time_start is not None and len(x_arr) > 0:
                latency = real_elapsed - x_max
                latency_str = f"Latency: {latency:.2f}s"
                scene.visuals.Text(
                    latency_str, color='#64ff64', font_size=16, pos=(10, 70), parent=view)

            canvas.update()

        # Start timer to update plot at FPS
        from vispy import app
        VispyPlotNode._timer = app.Timer(interval=1.0/self.FPS, connect=update_plot, start=True)
        app.run()

    def _on_close(self, event):
        with VispyPlotNode._lock:
            VispyPlotNode._canvas = None
            VispyPlotNode._view = None
            VispyPlotNode._start_time = None
            VispyPlotNode._bg_drawn = False
            VispyPlotNode._timer = None
            VispyPlotNode._real_time_start = None

NODE_CLASS_MAPPINGS = {
    "My KO Node": NodeKO,
    "PygamePlotNode": PygamePlotNode,
    "PyQtGraphPlotNode": PyQtGraphPlotNode,
    "OpenCVPlotNode": OpenCVPlotNode,
    "VispyPlotNode": VispyPlotNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstNode": "My First Node",
    "PygamePlotNode": "Pygame Plot Node",
    "PyQtGraphPlotNode": "PyQtGraph Plot Node",
    "OpenCVPlotNode": "OpenCV Plot Node",
    "VispyPlotNode": "Vispy Plot Node",
}