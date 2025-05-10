import threading
import numpy as np
import time

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

    FPS = 60
    FPS_CAP_ENABLED = True

    def __init__(self):
        self._plot_thread = None
        self._latest_data = ([], [], False)
        self._new_data = threading.Event()
        self._stop_event = threading.Event()
        self._last_draw_time = 0
        self._bg_surface = None
        self._bg_rect = None
        self._real_time_start = None

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
        self._latest_data = (list(x), list(y), as_points)
        self._new_data.set()
        if self._plot_thread is None or not self._plot_thread.is_alive():
            self._stop_event.clear()
            self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
            self._plot_thread.start()
        return ()

    def _plot_loop(self):
        import pygame
        import time as _time
        w, h = 640, 480
        margin_left, margin_right, margin_top, margin_bottom = 60, 30, 40, 60
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        with PygamePlotNode._lock:
        with PygamePlotNode._lock:
            if PygamePlotNode._window is None:
                pygame.init()
                PygamePlotNode._screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Pygame Plot")
                PygamePlotNode._window = True
                PygamePlotNode._start_time = _time.time()
        screen = PygamePlotNode._screen

        running = True
        last_data = ([], [], False)
        last_draw_time = 0
        min_interval = 1.0 / self.FPS if self.FPS_CAP_ENABLED else 0

        real_time_start = None  # Real time starts after first frame with signal

        while running and not self._stop_event.is_set():
            self._new_data.wait(timeout=0.001)
            now = _time.time()
            if self._new_data.is_set():
                last_data = self._latest_data
                self._new_data.clear()
            if self.FPS_CAP_ENABLED and (now - last_draw_time) < min_interval:
                continue
            last_draw_time = now

            x, y, as_points = last_data
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

            screen.fill((30, 30, 30))
            pygame.draw.line(screen, (255,255,255), (margin_left, h - margin_bottom), (w - margin_right, h - margin_bottom), 2)
            pygame.draw.line(screen, (255,255,255), (margin_left, h - margin_bottom), (margin_left, margin_top), 2)
            font = pygame.font.SysFont(None, 22)
            if show_labels:
                x_label_min = font.render(f"{x_min_rounded:.0f}s", True, (200, 200, 200))
                x_label_max = font.render(f"{x_max_rounded:.0f}s", True, (200, 200, 200))
                screen.blit(x_label_min, (margin_left, h - margin_bottom + 8))
                screen.blit(x_label_max, (w - margin_right - x_label_max.get_width(), h - margin_bottom + 8))
                y_min = float(np.min(y_arr)) if len(y_arr) > 0 else 0.0
                y_max = float(np.max(y_arr)) if len(y_arr) > 0 else 1.0
                y_label_min = font.render(f"{y_min:.2f}", True, (200, 200, 200))
                y_label_max = font.render(f"{y_max:.2f}", True, (200, 200, 200))
                screen.blit(y_label_min, (margin_left - y_label_min.get_width() - 8, h - margin_bottom - y_label_min.get_height()//2))
                screen.blit(y_label_max, (margin_left - y_label_max.get_width() - 8, margin_top - y_label_max.get_height()//2))

            if x_max - x_min == 0:
                x_norm = np.full_like(x_arr, margin_left + plot_w // 2)
            else:
                x_norm = ((x_arr - x_min) / (x_max - x_min) * plot_w + margin_left).astype(int)
            if len(y_arr) > 0:
                y_min = float(np.min(y_arr))
                y_max = float(np.max(y_arr))
            else:
                y_min, y_max = 0.0, 1.0
            if y_max - y_min == 0:
                y_norm = np.full_like(y_arr, margin_top + plot_h // 2)
            else:
                y_norm = (h - margin_bottom - ((y_arr - y_min) / (y_max - y_min) * plot_h)).astype(int)
            if PygamePlotNode._start_time is None:
                PygamePlotNode._start_time = _time.time()
            if as_points:
                for i in range(len(x_arr)):
                    pygame.draw.circle(screen, (0,0,255), (x_norm[i], y_norm[i]), 2)
            else:
                pts = list(zip(x_norm, y_norm))
                if len(pts) > 1:
                    pygame.draw.lines(screen, (0,0,255), False, pts, 2)

            font2 = pygame.font.SysFont(None, 24)
            # Node Time: start time of window (absolute, as string)
            node_time_str = time.strftime("Node Time: %Y-%m-%d %H:%M:%S", time.localtime(PygamePlotNode._start_time))
            node_time_surface = font2.render(node_time_str, True, (255, 255, 0))
            node_time_rect = node_time_surface.get_rect(topleft=(10, 10))
            screen.blit(node_time_surface, node_time_rect)

            # Real Time: start after first frame with signal (x_arr not empty)
            if real_time_start is None and len(x_arr) > 0:
                real_time_start = now
            if real_time_start is not None:
                real_elapsed = now - real_time_start
                real_minutes = int(real_elapsed) // 60
                real_seconds = int(real_elapsed) % 60
                real_time_str = f"Real Time: {real_minutes:02}:{real_seconds:02}"
                real_time_surface = font2.render(real_time_str, True, (255, 255, 0))
                real_time_rect = real_time_surface.get_rect(topleft=(10, 35))
                screen.blit(real_time_surface, real_time_rect)
            else:
                real_time_surface = font2.render("Real Time: --:--", True, (255, 255, 0))
                real_time_rect = real_time_surface.get_rect(topleft=(10, 35))
                screen.blit(real_time_surface, real_time_rect)

            # Latency: difference between real time and last x
            if real_time_start is not None and len(x_arr) > 0:
                latency = real_elapsed - x_max
                latency_str = f"Latency: {latency:.2f}s"
                latency_surface = font2.render(latency_str, True, (100, 255, 100))
                latency_rect = latency_surface.get_rect(topleft=(10, 60))
                screen.blit(latency_surface, latency_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    with PygamePlotNode._lock:
                        PygamePlotNode._window = None
                        PygamePlotNode._screen = None
                        PygamePlotNode._start_time = None
                    pygame.quit()

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

NODE_CLASS_MAPPINGS = {
    "My KO Node": NodeKO,
    "PygamePlotNode": PygamePlotNode,
    "PyQtGraphPlotNode": PyQtGraphPlotNode,
    "OpenCVPlotNode": OpenCVPlotNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstNode": "My First Node",
    "PygamePlotNode": "Pygame Plot Node",
    "PyQtGraphPlotNode": "PyQtGraph Plot Node",
    "OpenCVPlotNode": "OpenCV Plot Node",
}