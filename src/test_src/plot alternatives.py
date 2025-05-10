import numpy as np
import threading
import time

class PlotBenchmarker:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

    def plot_with_opencv(self, x, y, window_name="OpenCV Plot", as_points=False):
        import cv2
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        x = np.array(x)
        y = np.array(y)
        x_norm = ((x - x.min()) / (x.max() - x.min()) * (self.w - 40) + 20).astype(int)
        y_norm = (self.h - ((y - y.min()) / (y.max() - y.min()) * (self.h - 40) + 20)).astype(int)
        cv2.line(img, (20, self.h-20), (self.w-20, self.h-20), (255,255,255), 2)
        cv2.line(img, (20, self.h-20), (20, 20), (255,255,255), 2)
        if as_points:
            for i in range(len(x)):
                cv2.circle(img, (x_norm[i], y_norm[i]), 2, (255,0,0), -1)
        else:
            pts = np.column_stack((x_norm, y_norm))
            cv2.polylines(img, [pts], isClosed=False, color=(255,0,0), thickness=2)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_with_pygame(self, x, y, window_name="Pygame Plot", as_points=False):
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(window_name)
        screen.fill((0,0,0))
        x = np.array(x)
        y = np.array(y)
        x_norm = ((x - x.min()) / (x.max() - x.min()) * (self.w - 40) + 20).astype(int)
        y_norm = (self.h - ((y - y.min()) / (y.max() - y.min()) * (self.h - 40) + 20)).astype(int)
        pygame.draw.line(screen, (255,255,255), (20, self.h-20), (self.w-20, self.h-20), 2)
        pygame.draw.line(screen, (255,255,255), (20, self.h-20), (20, 20), 2)
        if as_points:
            for i in range(len(x)):
                pygame.draw.circle(screen, (0,0,255), (x_norm[i], y_norm[i]), 2)
        else:
            pts = list(zip(x_norm, y_norm))
            if len(pts) > 1:
                pygame.draw.lines(screen, (0,0,255), False, pts, 2)
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

    def plot_with_pyqtgraph(self, x, y, window_name="PyQtGraph Plot", as_points=False):
        import sys
        from PyQt6.QtWidgets import QApplication
        import pyqtgraph as pg

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            created_app = True
        else:
            created_app = False

        win = pg.plot(title=window_name)
        if as_points:
            win.plot(x, y, pen=None, symbol='o', symbolBrush='b')
        else:
            win.plot(x, y, pen=pg.mkPen('b', width=2))
        win.show()

        if created_app:
            app.exec()

    def run_plot_in_thread(self, plot_func, x, y, window_name, as_points=False):
        t = threading.Thread(target=plot_func, args=(x, y, window_name, as_points), daemon=True)
        t.start()
        return t

    def benchmark(self, x, y, as_points):
        print(f"\nBenchmarking OpenCV ({'points' if as_points else 'lines'})...")
        t0 = time.perf_counter()
        self.run_plot_in_thread(self.plot_with_opencv, x, y, f"OpenCV Plot ({'points' if as_points else 'lines'})", as_points=as_points)
        print(f"OpenCV ({'points' if as_points else 'lines'}) launched in {time.perf_counter() - t0:.6f} seconds.")

        print(f"Benchmarking Pygame ({'points' if as_points else 'lines'})...")
        t0 = time.perf_counter()
        self.run_plot_in_thread(self.plot_with_pygame, x, y, f"Pygame Plot ({'points' if as_points else 'lines'})", as_points=as_points)
        print(f"Pygame ({'points' if as_points else 'lines'}) launched in {time.perf_counter() - t0:.6f} seconds.")

        print(f"Benchmarking PyQtGraph ({'points' if as_points else 'lines'})...")
        t0 = time.perf_counter()
        self.run_plot_in_thread(self.plot_with_pyqtgraph, x, y, f"PyQtGraph Plot ({'points' if as_points else 'lines'})", as_points=as_points)
        print(f"PyQtGraph ({'points' if as_points else 'lines'}) launched in {time.perf_counter() - t0:.6f} seconds.\n")

if __name__ == "__main__":
    fs = 1000
    duration = 5
    x = np.linspace(0, duration, fs * duration)
    y = np.sin(x) + np.random.normal(0, 0.1, x.shape) + np.cos(x) * 0.5 + np.sin(x * 2) * 0.3

    benchmarker = PlotBenchmarker()

    print("Type 'run' to benchmark all plots, or 'exit' to quit.")

    while True:
        try:
            cmd = input(">>> ").strip().lower()
            if cmd == "exit":
                print("Exiting.")
                break
            elif cmd == "run":
                style = ""
                while style not in ("lines", "points"):
                    style = input("Plot style ('lines' or 'points'): ").strip().lower()
                as_points = (style == "points")
                benchmarker.benchmark(x, y, as_points)
            else:
                print("Unknown command. Type 'run' to benchmark or 'exit' to quit.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
