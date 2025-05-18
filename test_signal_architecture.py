"""
Minimal test for signal architecture:
- Real-time signal generator with ID
- Adapter to register signal in a plot registry
- Minimal pygame plot with rolling window
"""
import threading
import time
import numpy as np
import pygame
from collections import deque

# --- Minimal Signal Generator ---
class TestSignalGenerator:
    def __init__(self, signal_id, freq=1.0, noise=0.05):
        self.signal_id = signal_id
        self.freq = freq
        self.noise = noise
        self.metadata = {'name': f'Test Signal {signal_id}', 'color': (0, 200, 100)}
    def generate(self, t):
        return np.sin(2 * np.pi * self.freq * t) + np.random.normal(0, self.noise)

# --- Minimal Plot Registry ---
class MinimalPlotRegistry:
    def __init__(self):
        self.signals = {}
        self.metadata = {}
    def register_signal(self, signal_id, data, meta=None):
        self.signals[signal_id] = data
        if meta:
            self.metadata[signal_id] = meta

# --- Adapter function ---
def register_signal_in_registry(generator, registry, window_seconds=5, update_interval=0.05):
    t_deque = deque(maxlen=int(window_seconds / update_interval))
    data_deque = deque(maxlen=int(window_seconds / update_interval))
    def updater():
        while running[0]:
            now = time.time()
            t_deque.append(now)
            data_deque.append(generator.generate(now))
            registry.register_signal(generator.signal_id, (list(t_deque), list(data_deque)), generator.metadata)
            time.sleep(update_interval)
    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    return thread

# --- Minimal Pygame Plot ---
def run_minimal_plot():
    pygame.init()
    W, H = 600, 300
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Minimal Signal Plot Test")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    
    # Setup signal and registry
    registry = MinimalPlotRegistry()
    generator = TestSignalGenerator("TEST_SIGNAL", freq=0.7)
    global running
    running = [True]
    updater_thread = register_signal_in_registry(generator, registry, window_seconds=5, update_interval=0.05)
    
    try:
        start_time = time.time()
        buffer_filled = [False]
        maxlen = int(60 / 0.05)  # 1 minute buffer
        plot_window = 10.0  # seconds
        t_zero = [None]  # store initial time for axis zeroing
        while running[0]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running[0] = False
            screen.fill((30, 30, 30))
            # Draw plot
            if generator.signal_id in registry.signals:
                t, data = registry.signals[generator.signal_id]
                if len(t) > 1:
                    t = np.array(t)
                    data = np.array(data)
                    # Set t_zero at first data
                    if t_zero[0] is None:
                        t_zero[0] = t[0]
                    t = t - t_zero[0]
                    # Only plot the last plot_window seconds
                    t_now = t[-1]
                    mask = t >= (t_now - plot_window)
                    t_plot = t[mask]
                    data_plot = data[mask]
                    if len(t_plot) < 2:
                        continue
                    tmin, tmax = np.min(t_plot), np.max(t_plot)
                    dmin, dmax = np.min(data_plot), np.max(data_plot)
                    if dmax == dmin:
                        dmax = dmin + 1
                    points = []
                    for i in range(len(t_plot)):
                        px = int(40 + (t_plot[i] - tmin) / (tmax - tmin + 1e-6) * (W - 80))
                        py = int(H - 40 - (data_plot[i] - dmin) / (dmax - dmin + 1e-6) * (H - 80))
                        points.append((px, py))
                    if len(points) > 1:
                        pygame.draw.lines(screen, (0, 200, 100), False, points, 2)
                    # Draw a proper time axis with ticks and labels
                    axis_y = H - 35
                    pygame.draw.line(screen, (120,120,120), (40, axis_y), (W-40, axis_y), 2)
                    n_ticks = 6
                    for i in range(n_ticks):
                        frac = i/(n_ticks-1)
                        tx = 40 + frac*(W-80)
                        tval = tmin + frac*(tmax-tmin)
                        pygame.draw.line(screen, (180,180,180), (tx, axis_y-5), (tx, axis_y+5), 2)
                        tick_label = font.render(f"{tval:.2f}s", True, (180,180,180))
                        label_rect = tick_label.get_rect(center=(tx, axis_y+18))
                        screen.blit(tick_label, label_rect)
            # Show runtime
            runtime = time.time() - start_time
            runtime_label = font.render(f"Runtime: {runtime:.2f}s", True, (255, 200, 100))
            screen.blit(runtime_label, (W-200, 10))
            label = font.render("Minimal Real-Time Signal Plot", True, (220, 220, 220))
            screen.blit(label, (20, 10))
            pygame.display.flip()
            clock.tick(60)
    finally:
        running[0] = False
        updater_thread.join(timeout=1)
        pygame.quit()

if __name__ == "__main__":
    run_minimal_plot()
