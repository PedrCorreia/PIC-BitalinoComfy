import pygame
import numpy as np
import re
from ..constants import SECTION_MARGIN  # Add this import

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
            ("HR", "HR_METRIC", 35, 125, [
                (35, (0, 120, 255)),   # blue
                (60, (0, 200, 80)),    # green
                (80, (255, 220, 0)),   # yellow
                (100, (255, 140, 0)),  # orange
                (120, (255, 140, 0)) # red
            ]),
            # SCL = Skin Conductance Level (tonic, slow-changing baseline, in uS)
            ("SCL", "SCL_METRIC", 0, 40, [
                (0, (0, 120, 255)),     # sleep (blue)
                (2, (0, 200, 80)),      # relaxed (green)
                (10, (255, 220, 0)),     # normal (yellow)
                (15, (255, 140, 0)),    # aroused (orange)
                (20, (255, 40, 40)),    # stressed (red)
            ]),
            # SCR = Skin Conductance Response (phasic, rapid changes, in uS)
            ("SCR", "SCK_METRIC", 0, 20, [
                (0, (0, 120, 255)),     # sleep (blue)
                (0.1, (0, 200, 80)),      # relaxed (green)
                (0.5, (255, 220, 0)),     # normal (yellow)
                (0.8, (255, 140, 0)),    # aroused (orange)
                (1, (255, 40, 40)),    # stressed (red)
            ]),
            ("SCR Freq", "scr_frequency", 0, 30, [
                (0, (0, 120, 255)),
                (1, (0, 200, 80)),
                (1.5, (255, 220, 0)),
                (2, (255, 140, 0)),
                (2.5, (255, 40, 40)),
            ]),
            ("RR", "RR_METRIC", 8, 40, [  # RR = respiration rate (breaths/min)
                (5, (0, 120, 255)),
                (10, (0, 200, 80)),
                (20, (255, 220, 0)),
                (25, (255, 140, 0)),
                (40, (255, 40, 40)),
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
        n_metrics = len(self.metric_configs)
        n_cols = 2
        n_rows = 2
        margin = SECTION_MARGIN  # Use SECTION_MARGIN for consistency
        total_cells = n_cols * n_rows
        # Center the grid in the available area
        cell_size = min((width - (n_cols + 1) * margin) // n_cols, (height - (n_rows + 1) * margin) // n_rows)
        cell_size = cell_size * 2  # Double the cell height (and width, since grid is square)
        grid_w = n_cols * cell_size + (n_cols + 1) * margin
        grid_h = n_rows * cell_size + (n_rows + 1) * margin
        grid_x = x + (width - grid_w) // 2
        grid_y = y + (height - grid_h) // 2
        plot_h = int(cell_size * 0.7)
        bar_h = int(cell_size * 0.22)
        plot_w = cell_size
        font = self.font
        try:
            small_font = pygame.font.Font(font.get_font() if hasattr(font, 'get_font') else None, max(12, font.get_height() - 4))
        except Exception:
            small_font = font
        # Pad metric configs to fill grid
        padded_configs = list(self.metric_configs) + [None] * (total_cells - n_metrics)
        for idx in range(total_cells):
            row = idx // n_cols
            col = idx % n_cols
            px = grid_x + margin + col * (cell_size + margin)
            py = grid_y + margin + row * (cell_size + margin)
            config = padded_configs[idx]
            if config is not None:
                label, sig_pattern, min_v, max_v, color_stops = config
                sig = self._find_matching_signal(sig_pattern)
            else:
                label, sig, min_v, max_v, color_stops = "", None, 0, 1, [(0, (80, 80, 80)), (1, (80, 80, 80))]
            # --- Draw plot rectangle (top, square) ---
            plot_rect = pygame.Rect(px, py, plot_w, plot_h)
            pygame.draw.rect(screen, (30, 30, 30), plot_rect, border_radius=8)
            # --- Draw time-series plot inside plot_rect ---
            plot_drawn = False
            try:
                if (
                    sig and isinstance(sig, dict)
                    and "t" in sig and "v" in sig
                    and isinstance(sig["t"], (list, np.ndarray))
                    and isinstance(sig["v"], (list, np.ndarray))
                    and len(sig["t"]) > 1 and len(sig["v"]) > 1
                    and len(sig["t"]) == len(sig["v"])
                ):
                    t = np.array(sig["t"], dtype=float)
                    v = np.array(sig["v"], dtype=float)
                    if t.size > 1 and v.size > 1 and np.all(np.isfinite(t)) and np.all(np.isfinite(v)):
                        now = t[-1]
                        mask = (t >= now - window_sec)
                        t = t[mask]
                        v = v[mask]
                        if t.size > 1 and v.size > 1:
                            t0, t1 = t[0], t[-1]
                            t_norm = (t - t0) / (t1 - t0) if t1 != t0 else np.zeros_like(t)
                            v_min, v_max = np.min(v), np.max(v)
                            v_norm = (v - v_min) / (v_max - v_min) if v_max != v_min else np.zeros_like(v)
                            pts = [
                                (int(plot_rect.left + 40 + tx * (plot_rect.width - 80)),
                                 int(plot_rect.top + 10 + (1 - vy) * (plot_rect.height - 40)))
                                for tx, vy in zip(t_norm, v_norm)
                            ]
                            if len(pts) > 1:
                                pygame.draw.lines(screen, (180, 220, 255), False, pts, 2)
                            # --- Draw time at left/right below plot ---
                            def fmt_time(val):
                                try:
                                    val = float(val)
                                    if val > 3600:
                                        h = int(val // 3600)
                                        m = int((val % 3600) // 60)
                                        s = int(val % 60)
                                        return f"{h:02}:{m:02}:{s:02}"
                                    else:
                                        return f"{val:.1f}s"
                                except Exception:
                                    return str(val)
                            t0_surface = font.render(fmt_time(t0), True, (120, 120, 120))
                            t1_surface = font.render(fmt_time(t1), True, (120, 120, 120))
                            screen.blit(t0_surface, (plot_rect.left + 40, plot_rect.bottom - 18))
                            screen.blit(t1_surface, (plot_rect.right - 40 - t1_surface.get_width(), plot_rect.bottom - 18))
                            # --- Draw min/max value at left/right of plot ---
                            vmin_surface = font.render(f"{v_min:.1f}", True, (120, 120, 120))
                            vmax_surface = font.render(f"{v_max:.1f}", True, (120, 120, 120))
                            screen.blit(vmin_surface, (plot_rect.left + 8, plot_rect.top + plot_rect.height // 2 - vmin_surface.get_height() // 2))
                            screen.blit(vmax_surface, (plot_rect.right - 8 - vmax_surface.get_width(), plot_rect.top + plot_rect.height // 2 - vmax_surface.get_height() // 2))
                            plot_drawn = True
            except Exception:
                pass
            if not plot_drawn:
                # Draw placeholder axes and label if no data
                try:
                    pygame.draw.line(screen, (80, 80, 80), (plot_rect.left + 40, plot_rect.bottom - 20), (plot_rect.right - 40, plot_rect.bottom - 20), 1)
                    pygame.draw.line(screen, (80, 80, 80), (plot_rect.left + 40, plot_rect.top + 10), (plot_rect.left + 40, plot_rect.bottom - 20), 1)
                    placeholder_surface = small_font.render("No data" if config is not None else "", True, (100, 100, 100))
                    screen.blit(placeholder_surface, (plot_rect.centerx - placeholder_surface.get_width() // 2, plot_rect.centery - placeholder_surface.get_height() // 2))
                except Exception:
                    pass
            # --- Draw label centered at top ---
            label_surface = font.render(label, True, (255, 255, 255)) if label else None
            if label_surface:
                screen.blit(label_surface, (plot_rect.centerx - label_surface.get_width() // 2, plot_rect.top + 4))
            # --- Draw bar (bottom, under plot, full width of cell) ---
            bar_y = plot_rect.bottom + 4
            bar_x = plot_rect.left
            bar_w = plot_rect.width
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
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
            except Exception:
                pass
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
                pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=6)
                # Draw min/max value labels at bar ends
                min_surface = font.render(f"{min_v:.1f}", True, (180, 180, 180))
                max_surface = font.render(f"{max_v:.1f}", True, (180, 180, 180))
                screen.blit(min_surface, (bar_x - min_surface.get_width() - 5, bar_y + bar_h//2 - min_surface.get_height()//2))
                screen.blit(max_surface, (bar_x + bar_w + 5, bar_y + bar_h//2 - max_surface.get_height()//2))
                # Draw last value label (smaller, semi-transparent) centered over the bar
                val_surface = small_font.render(f"{last_val:.2f}", True, (220, 220, 220, 160))
                val_x = bar_x + (bar_w - val_surface.get_width()) // 2
                val_y = bar_y + (bar_h - val_surface.get_height()) // 2
                try:
                    val_surface.set_alpha(160)
                except Exception:
                    pass  # set_alpha may not be supported on all surfaces
                screen.blit(val_surface, (val_x, val_y))
            else:
                # If no signal or value, show a gray bar and 'N/A' text
                pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, 2, bar_h), border_radius=6)
                if config is not None:
                    na_surface = small_font.render("N/A", True, (180, 180, 180))
                    screen.blit(na_surface, (bar_x + 5, bar_y + 2))
                if sig is not None and not isinstance(sig, dict):
                    na_type_surface = small_font.render(f"Type: {type(sig).__name__}", True, (180, 100, 100))
                    screen.blit(na_type_surface, (bar_x + 60, bar_y + 2))
            # Defensive: always continue to next metric, never crash
            continue
