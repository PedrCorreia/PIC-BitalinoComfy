import pygame
import numpy as np
import re

class MetricsView:
    """
    Modular metrics dashboard: plots time series and a color-changing bar for each metric.
    Each metric can be configured with label, signal id, min/max, and color stops.
    """
    def __init__(self, font, plot_registry, metric_configs=None):
        self.font = font
        self.plot_registry = plot_registry
        # Each config: (label, signal_id, min, max, color_stops)
        self.metric_configs = metric_configs or [
            ("HR", "HR_METRIC", 40, 120, [
                (40, (0, 120, 255)),   # blue
                (60, (0, 200, 80)),    # green
                (80, (255, 220, 0)),   # yellow
                (100, (255, 140, 0)),  # orange
                (120, (255, 40, 40)),  # red
            ]),
            ("SCL", "SCL_METRIC", 0, 10, [
                (0, (0, 120, 255)),
                (2, (0, 200, 80)),
                (5, (255, 220, 0)),
                (8, (255, 140, 0)),
                (10, (255, 40, 40)),
            ]),
            ("SCK", "SCK_METRIC", 0, 10, [
                (0, (0, 120, 255)),
                (2, (0, 200, 80)),
                (5, (255, 220, 0)),
                (8, (255, 140, 0)),
                (10, (255, 40, 40)),
            ]),
            ("RR", "RR_METRIC", 8, 30, [  # RR = respiration rate (breaths/min)
                (8, (0, 120, 255)),
                (12, (0, 200, 80)),
                (18, (255, 220, 0)),
                (24, (255, 140, 0)),
                (30, (255, 40, 40)),
            ]),
        ]

    def _find_matching_signal(self, pattern):
        # Match any signal id in the registry that contains the pattern (case-insensitive, allows suffixes/prefixes)
        all_ids = self.plot_registry.get_all_signal_ids() if hasattr(self.plot_registry, 'get_all_signal_ids') else []
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        for sid in all_ids:
            if regex.search(str(sid)):
                return self.plot_registry.get_signal(sid)
        return None

    def draw(self, screen, x, y, width, height, window_sec=30):
        print("[MetricsView] draw called")
        n_metrics = len(self.metric_configs)
        min_plot_h = 30
        plot_h = max((height - n_metrics * 30) // n_metrics, min_plot_h)
        for i, (label, sig_pattern, min_v, max_v, color_stops) in enumerate(self.metric_configs):
            sig = self._find_matching_signal(sig_pattern)
            plot_y = y + i * (plot_h + 30)
            # Debug: print type and value if not dict
            if sig is not None and not isinstance(sig, dict):
                print(f"[MetricsView][WARN] Signal for '{label}' matched pattern '{sig_pattern}' but is not a dict: type={type(sig)}, value={repr(sig)[:120]}")
            # Draw time-series plot
            if (
                sig and isinstance(sig, dict)
                and "t" in sig and "v" in sig
                and isinstance(sig["t"], (list, np.ndarray))
                and isinstance(sig["v"], (list, np.ndarray))
                and len(sig["t"]) > 1 and len(sig["v"]) > 1
                and len(sig["t"]) == len(sig["v"])
            ):
                try:
                    t = np.array(sig["t"], dtype=float)
                    v = np.array(sig["v"], dtype=float)
                    now = t[-1]
                    mask = (t >= now - window_sec)
                    t = t[mask]
                    v = v[mask]
                    if len(t) > 1:
                        t_norm = (t - t[0]) / (t[-1] - t[0]) if t[-1] != t[0] else np.zeros_like(t)
                        v_min, v_max = np.min(v), np.max(v)
                        v_norm = (v - v_min) / (v_max - v_min) if v_max != v_min else np.zeros_like(v)
                        pts = [
                            (int(x + tx * width), int(plot_y + plot_h - vy * plot_h))
                            for tx, vy in zip(t_norm, v_norm)
                        ]
                        if len(pts) > 1:
                            pygame.draw.lines(screen, (180, 220, 255), False, pts, 2)
                except Exception as e:
                    print(f"[MetricsView] Exception in time-series plot for {label}: {e}")
            # Draw label
            label_surface = self.font.render(label, True, (255, 255, 255))
            screen.blit(label_surface, (x + 10, plot_y + 5))
            # Draw average bar for the same window_sec
            bar_y = plot_y + plot_h + 5
            bar_h = 20
            bar_w = width - 120
            bar_x = x + 100
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
            # --- Color-changing, size-varying bar based on 'last' value ---
            last_val = None
            try:
                if sig and isinstance(sig, dict):
                    if "last" in sig and isinstance(sig["last"], (float, int, np.floating, np.integer)):
                        last_val = float(sig["last"])
                    elif "last" in sig and isinstance(sig["last"], (list, np.ndarray)) and len(sig["last"]):
                        last_val = float(sig["last"][-1])
                    elif "v" in sig and isinstance(sig["v"], (list, np.ndarray)) and len(sig["v"]):
                        last_val = float(sig["v"][-1])
            except Exception as e:
                print(f"[MetricsView] Exception extracting last_val for {label}: {e}")
            try:
                sig_keys = list(sig.keys()) if isinstance(sig, dict) else None
            except Exception:
                sig_keys = None
            print(f"[MetricsView] {label}: last={last_val}, sig_id={sig_pattern}, sig_keys={sig_keys}")
            # Clamp and normalize last_val for bar
            if last_val is not None and np.isfinite(last_val):
                norm = (last_val - min_v) / (max_v - min_v) if (max_v - min_v) != 0 else 0.0
                norm = max(0.0, min(1.0, norm))
                def lerp(a, b, t):
                    return a + (b - a) * t
                def color_gradient(val):
                    for j in range(len(color_stops) - 1):
                        v0, c0 = color_stops[j]
                        v1, c1 = color_stops[j+1]
                        if v1 == v0:
                            t = 0.0
                        else:
                            t = (val - v0) / (v1 - v0)
                        if val <= v1:
                            return tuple(int(lerp(c0[k], c1[k], max(0.0, min(1.0, t)))) for k in range(3))
                    return color_stops[-1][1]
                bar_color = color_gradient(last_val)
                fill_w = int(bar_w * norm)
                pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
                val_surface = self.font.render(f"Last: {last_val:.2f}", True, (255, 255, 255))
                screen.blit(val_surface, (bar_x + 5, bar_y + 2))
            else:
                # If no signal or value, show a gray bar and 'N/A' text
                pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, 2, bar_h))
                na_surface = self.font.render("N/A", True, (180, 180, 180))
                screen.blit(na_surface, (bar_x + 5, bar_y + 2))
                if sig is not None and not isinstance(sig, dict):
                    na_type_surface = self.font.render(f"Type: {type(sig).__name__}", True, (180, 100, 100))
                    screen.blit(na_type_surface, (bar_x + 60, bar_y + 2))
