import pygame
import numpy as np
import re
from ..constants import SECTION_MARGIN  # Add this import

class MetricsView:
    """
    Modular metrics dashboard: plots time series and a color-changing bar for each metric.
    Each metric can be configured with label, signal id, min/max, and color stops.
    """
    def __init__(self, font, signal_registry, metric_configs=None): # Renamed plot_registry to signal_registry
        self.font = font
        self.signal_registry = signal_registry # Use signal_registry
        # Each config: (label, signal_id, min, max, color_stops)
        self.metric_configs = metric_configs or [
            ("HR", "HR_METRIC", 35, 125, [
                (35, (0, 120, 255)),   # blue
                (60, (0, 200, 80)),    # green
                (80, (255, 220, 0)),   # yellow
                (100, (255, 140, 0)),  # orange
                (120, (255, 40, 40))   # red
            ]),
            ("RR", "RR_METRIC", 0, 30, [  # RR = respiration rate (breaths/min), 0 = no peaks detected
                (0, (0, 120, 255)),    # No breathing detected (blue - sleep/apnea)
                (8, (0, 200, 80)),     # Very slow breathing (green - relaxed)
                (12, (255, 220, 0)),   # Normal breathing (yellow - normal)
                (18, (255, 140, 0)),   # Fast breathing (orange - aroused)
                (25, (255, 40, 40)),   # Very fast breathing (red - stressed)
                (30, (255, 40, 40)),   # Max arousal (red)
            ]),
            # SCL = Skin Conductance Level (tonic, slow-changing baseline, in uS)
            ("SCL", "SCL_METRIC", 0, 40, [
                (0, (0, 120, 255)),     # sleep (blue)
                (2, (0, 200, 80)),      # relaxed (green)
                (5, (255, 220, 0)),     # normal (yellow)
                (12, (255, 140, 0)),    # aroused (orange)
                (20, (255, 40, 40)),    # stressed (red)
            ]),
            ("SCR Freq", "scr_frequency", 0, 6, [
                (0, (0, 120, 255)),
                (2, (0, 200, 80)),
                (5, (255, 220, 0)),
                (7, (255, 140, 0)),
                (10, (255, 40, 40)),
            ]),
            
            # AROUSAL METRICS - All use 0.0-1.0 range with consistent color mapping
            ("RR Arousal", "RR_AROUSAL_METRIC", 0.0, 1.0, [
                (0.0, (0, 120, 255)),    # Sleep (blue)
                (0.2, (0, 200, 80)),     # Relaxed (green)  
                (0.4, (255, 220, 0)),    # Normal (yellow)
                (0.6, (255, 140, 0)),    # Aroused (orange)
                (0.8, (255, 40, 40)),    # Stressed (red)
                (1.0, (255, 40, 40)),    # Max stressed (red)
            ]),
            ("ECG Arousal", "ECG_AROUSAL_METRIC", 0.0, 1.0, [
                (0.0, (0, 120, 255)),
                (0.2, (0, 200, 80)),
                (0.4, (255, 220, 0)),
                (0.6, (255, 140, 0)),
                (0.8, (255, 40, 40)),
                (1.0, (255, 40, 40)),
            ]),
            ("EDA Arousal", "EDA_AROUSAL_METRIC", 0.0, 1.0, [
                (0.0, (0, 120, 255)),
                (0.2, (0, 200, 80)),
                (0.4, (255, 220, 0)),
                (0.6, (255, 140, 0)),
                (0.8, (255, 40, 40)),
                (1.0, (255, 40, 40)),
            ]),
            ("Overall Arousal", "OVERALL_AROUSAL_METRIC", 0.0, 1.0, [
                (0.0, (0, 120, 255)),
                (0.2, (0, 200, 80)),
                (0.4, (255, 220, 0)),
                (0.6, (255, 140, 0)),
                (0.8, (255, 40, 40)),
                (1.0, (255, 40, 40)),
            ]),
            
        ]
        
        # Remove separate arousal_metrics - now included in main metric_configs
        self.arousal_metrics = [
            ("RR Arousal", "RR_AROUSAL_METRIC"), 
            ("ECG Arousal", "ECG_AROUSAL_METRIC"), # Assuming HR and ECG arousal are related or HR_AROUSAL_METRIC might be intended
            ("EDA Arousal", "EDA_AROUSAL_METRIC"),
            ("Overall", "OVERALL_AROUSAL_METRIC")
        ]
        self.legend_colors = [
            ((0, 120, 255), "Sleep (0.0-0.2)"),
            ((0, 200, 80), "Relaxed (0.2-0.4)"),
            ((255, 220, 0), "Normal (0.4-0.6)"),
            ((255, 140, 0), "Aroused (0.6-0.8)"),
            ((255, 40, 40), "Stressed (0.8-1.0)")
        ]

    def _find_matching_signal(self, pattern):
        # Match any signal id in the registry that contains the pattern (case-insensitive, allows suffixes/prefixes)
        all_ids = self.signal_registry.get_all_signal_ids() if hasattr(self.signal_registry, 'get_all_signal_ids') else [] # Use signal_registry
        #print(f"[MetricsView DEBUG] _find_matching_signal: Searching for pattern '{pattern}' among {len(all_ids)} signals: {all_ids}") # DEBUG PRINT
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        for sid in all_ids:
            if regex.search(str(sid)):
                #print(f"[MetricsView DEBUG] _find_matching_signal: Found match for '{pattern}': {sid}") # DEBUG PRINT
                return self.signal_registry.get_signal(sid) # Use signal_registry
        #print(f"[MetricsView DEBUG] _find_matching_signal: No matching signal found for '{pattern}'") # DEBUG PRINT
        return None

    def draw(self, screen, x, y, width, height, window_sec=30):
        font = self.font
        try:
            small_font = pygame.font.Font(font.get_font() if hasattr(font, 'get_font') else None, max(12, font.get_height() - 4))
        except Exception:
            small_font = font

        arousal_bar_height = 40  # Height for the top arousal bar
        color_key_bar_height = 40 # Height for the bottom color key bar
        margin = SECTION_MARGIN

        # Draw Top Arousal Bar
        self._draw_top_arousal_bar(screen, x, y, width, arousal_bar_height, font, small_font)

        # Calculate area for metric plots (between arousal bar and color key bar)
        metric_plots_y = y + arousal_bar_height + margin
        metric_plots_height = height - arousal_bar_height - color_key_bar_height - (2 * margin)
        
        n_metrics = len(self.metric_configs)
        n_cols = 2
        n_rows = 2
        total_cells = n_cols * n_rows
        
        # Grid calculations (now uses full width)
        cell_size = min((width - (n_cols + 1) * margin) // n_cols, (metric_plots_height - (n_rows + 1) * margin) // n_rows)
        grid_w = n_cols * cell_size + (n_cols + 1) * margin
        grid_h = n_rows * cell_size + (n_rows + 1) * margin
        grid_x = x + (width - grid_w) // 2
        grid_y = metric_plots_y + (metric_plots_height - grid_h) // 2 # Centered in the available vertical space
        
        plot_h = int(cell_size * 0.7)
        bar_h = int(cell_size * 0.22)
        plot_w = cell_size

        # Pad metric configs to fill grid
        padded_configs = list(self.metric_configs) + [None] * (total_cells - n_metrics)
        for idx in range(total_cells):
            row_idx = idx // n_cols # Renamed from row to row_idx to avoid conflict
            col_idx = idx % n_cols # Renamed from col to col_idx
            px = grid_x + margin + col_idx * (cell_size + margin)
            py = grid_y + margin + row_idx * (cell_size + margin)
            config = padded_configs[idx]
            if config is not None:
                label, sig_pattern, min_v, max_v, color_stops = config
                #print(f"[MetricsView DEBUG] Processing metric: Label='{label}', SignalPattern='{sig_pattern}'") 
                sig = self._find_matching_signal(sig_pattern)
                #print(f"[MetricsView DEBUG] Metric '{label}': Retrieved signal data: {'Exists' if sig else 'None'}") 
                if sig and isinstance(sig, dict):
                    pass
                    #print(f"[MetricsView DEBUG] Metric '{label}': Signal keys: {list(sig.keys())}, t_len: {len(sig.get('t', []))}, v_len: {len(sig.get('v', []))}, last: {sig.get('last', 'N/A')}")
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
                    #print(f"[MetricsView DEBUG] Metric '{label}': Data is valid for plotting.") 
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
                            
                            # Use STATIC Y-axis ranges from config (physiological expected ranges)
                            v_min_expected, v_max_expected = min_v, max_v  # From metric config
                            v_norm = (v - v_min_expected) / (v_max_expected - v_min_expected) if v_max_expected != v_min_expected else np.zeros_like(v)
                            # Clamp normalized values to [0,1] range to handle out-of-range data gracefully
                            v_norm = np.clip(v_norm, 0.0, 1.0)
                            
                            pts = [
                                (int(plot_rect.left + 40 + tx * (plot_rect.width - 80)),
                                 int(plot_rect.top + 10 + (1 - vy) * (plot_rect.height - 40)))
                                for tx, vy in zip(t_norm, v_norm)
                            ]
                            if len(pts) > 1:
                                pygame.draw.lines(screen, (180, 220, 255), False, pts, 2)
                            # --- Draw time at left/right below plot ---
                            def fmt_time(val_time): # Renamed val to val_time
                                try:
                                    val_time = float(val_time)
                                    if val_time > 3600:
                                        h = int(val_time // 3600)
                                        m = int((val_time % 3600) // 60)
                                        s = int(val_time % 60)
                                        return f"{h:02}:{m:02}:{s:02}"
                                    else:
                                        return f"{val_time:.1f}s"
                                except Exception:
                                    return str(val_time)
                            t0_surface = font.render(fmt_time(t0), True, (120, 120, 120))
                            t1_surface = font.render(fmt_time(t1), True, (120, 120, 120))
                            screen.blit(t0_surface, (plot_rect.left + 40, plot_rect.bottom - 18))
                            screen.blit(t1_surface, (plot_rect.right - 40 - t1_surface.get_width(), plot_rect.bottom - 18))
                            # --- Draw min/max expected values at left/right of plot ---
                            vmin_surface = font.render(f"{min_v:.1f}", True, (120, 120, 120))
                            vmax_surface = font.render(f"{max_v:.1f}", True, (120, 120, 120))
                            screen.blit(vmin_surface, (plot_rect.left + 8, plot_rect.bottom - 20 - vmin_surface.get_height()))
                            screen.blit(vmax_surface, (plot_rect.left + 8, plot_rect.top + 10))
                            plot_drawn = True
            except Exception as e:
                print(f"[MetricsView DEBUG] Metric '{label}': Exception during plot drawing: {e}") 
                pass # Keep existing pass
            if not plot_drawn:
                #print(f"[MetricsView DEBUG] Metric '{label}': Plot not drawn (no data or error).") 
                try:
                    pygame.draw.line(screen, (80, 80, 80), (plot_rect.left + 40, plot_rect.bottom - 20), (plot_rect.right - 40, plot_rect.bottom - 20), 1)
                    pygame.draw.line(screen, (80, 80, 80), (plot_rect.left + 40, plot_rect.top + 10), (plot_rect.left + 40, plot_rect.bottom - 20), 1)
                    placeholder_surface = small_font.render("No data" if config is not None else "", True, (100, 100, 100))
                    screen.blit(placeholder_surface, (plot_rect.centerx - placeholder_surface.get_width() // 2, plot_rect.centery - placeholder_surface.get_height() // 2))
                except Exception:
                    pass # Keep existing pass
            # --- Draw label centered at top ---
            label_surface = font.render(label, True, (255, 255, 255)) if label else None
            if label_surface:
                screen.blit(label_surface, (plot_rect.centerx - label_surface.get_width() // 2, plot_rect.top + 4))
            # --- Draw bar (bottom, under plot, full width of cell) ---
            bar_y_pos = plot_rect.bottom + 4 # Renamed bar_y to bar_y_pos
            bar_x_pos = plot_rect.left # Renamed bar_x to bar_x_pos
            bar_w_val = plot_rect.width # Renamed bar_w to bar_w_val
            pygame.draw.rect(screen, (60, 60, 60), (bar_x_pos, bar_y_pos, bar_w_val, bar_h), border_radius=6)
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
                pass  # Keep existing pass
            
            if last_val is not None and np.isfinite(last_val):
                norm = (last_val - min_v) / (max_v - min_v) if (max_v - min_v) != 0 else 0.0
                norm = max(0.0, min(1.0, norm))
                def lerp(a_val, b_val, t_val): # Renamed to avoid conflict
                    return a_val + (b_val - a_val) * t_val
                def color_gradient(val_grad): # Renamed val to val_grad
                    # For regular metrics (HR, RR, SCL, SCR Freq), the color_stops contain actual metric values
                    # For arousal metrics (0.0-1.0), the color_stops contain arousal levels
                    # We use the actual metric value to find the appropriate color
                    for j_idx in range(len(color_stops) - 1): # Renamed j to j_idx
                        v0, c0 = color_stops[j_idx]
                        v1, c1 = color_stops[j_idx+1]
                        if val_grad <= v1:
                            if v1 == v0: t_calc = 0.0 # Renamed t to t_calc
                            else: t_calc = (val_grad - v0) / (v1 - v0)
                            t_calc = max(0.0, min(1.0, t_calc))
                            return tuple(int(lerp(c0[k_idx], c1[k_idx], t_calc)) for k_idx in range(3)) # Renamed k to k_idx
                    return color_stops[-1][1]
                bar_color = color_gradient(last_val)
                fill_w = int(bar_w_val * norm)
                pygame.draw.rect(screen, bar_color, (bar_x_pos, bar_y_pos, fill_w, bar_h), border_radius=6)
                min_surface = font.render(f"{min_v:.1f}", True, (180, 180, 180))
                max_surface = font.render(f"{max_v:.1f}", True, (180, 180, 180))
                screen.blit(min_surface, (bar_x_pos - min_surface.get_width() - 5, bar_y_pos + bar_h//2 - min_surface.get_height()//2))
                screen.blit(max_surface, (bar_x_pos + bar_w_val + 5, bar_y_pos + bar_h//2 - max_surface.get_height()//2))
                val_surface = small_font.render(f"{last_val:.2f}", True, (220, 220, 220, 160))
                val_x_pos = bar_x_pos + (bar_w_val - val_surface.get_width()) // 2 # Renamed val_x to val_x_pos
                val_y_pos = bar_y_pos + (bar_h - val_surface.get_height()) // 2 # Renamed val_y to val_y_pos
                try: val_surface.set_alpha(160)
                except Exception: pass # Keep existing pass
                screen.blit(val_surface, (val_x_pos, val_y_pos))
            else:
                pygame.draw.rect(screen, (100, 100, 100), (bar_x_pos, bar_y_pos, 2, bar_h), border_radius=6)
                if config is not None:
                    na_surface = small_font.render("N/A", True, (180, 180, 180))
                    screen.blit(na_surface, (bar_x_pos + 5, bar_y_pos + 2))
                if sig is not None and not isinstance(sig, dict):
                    na_type_surface = small_font.render(f"Type: {type(sig).__name__}", True, (180, 100, 100))
                    screen.blit(na_type_surface, (bar_x_pos + 60, bar_y_pos + 2))
            continue
        
        # Draw Bottom Color Key Bar
        color_key_bar_y = y + height - color_key_bar_height # Positioned at the very bottom
        self._draw_bottom_color_key(screen, x, color_key_bar_y, width, color_key_bar_height, font, small_font)

    def _draw_top_arousal_bar(self, screen, x, y, width, bar_height, font, small_font):
        pygame.draw.rect(screen, (25, 25, 25), (x, y, width, bar_height)) # Bar background
        current_x = x + 10 # Start drawing text from left + margin
        
        for label, signal_id_pattern in self.arousal_metrics:
            arousal_value = None
            sig = self._find_matching_signal(signal_id_pattern)
            
            if sig and isinstance(sig, dict):
                if "last" in sig and sig["last"] is not None: arousal_value = sig["last"]
                elif "v" in sig and len(sig["v"]) > 0 and sig["v"][-1] is not None: arousal_value = sig["v"][-1]
                elif "meta" in sig and isinstance(sig["meta"], dict) and "arousal_value" in sig["meta"]:
                    arousal_value = sig["meta"]["arousal_value"]
                
                if arousal_value is None and hasattr(self.signal_registry, 'get_metadata'):
                    actual_signal_id_in_registry = None
                    all_ids = self.signal_registry.get_all_signal_ids() if hasattr(self.signal_registry, 'get_all_signal_ids') else []
                    regex = re.compile(re.escape(signal_id_pattern), re.IGNORECASE)
                    for sid_in_reg in all_ids:
                        if regex.search(str(sid_in_reg)):
                            actual_signal_id_in_registry = sid_in_reg
                            break
                    if actual_signal_id_in_registry:
                        meta = self.signal_registry.get_metadata(actual_signal_id_in_registry)
                        if meta and "arousal_value" in meta and meta["arousal_value"] is not None:
                            arousal_value = meta["arousal_value"]
            
            value_str = f"{float(arousal_value):.2f}" if arousal_value is not None and isinstance(arousal_value, (float, int, np.floating, np.integer)) and np.isfinite(arousal_value) else "N/A"
            
            text = f"{label}: {value_str}"
            text_surface = small_font.render(text, True, (220, 220, 220))
            
            if current_x + text_surface.get_width() < x + width -10: # Check if fits
                screen.blit(text_surface, (current_x, y + (bar_height - text_surface.get_height()) // 2))
                current_x += text_surface.get_width() + 15 # Add padding for next item (" | ")
                
                # Draw separator if not the last item and there's space
                if label != self.arousal_metrics[-1][0] and current_x + small_font.render(" | ", True, (150,150,150)).get_width() < x + width -10 :
                    sep_surface = small_font.render(" | ", True, (150,150,150))
                    screen.blit(sep_surface, (current_x, y + (bar_height - sep_surface.get_height()) // 2))
                    current_x += sep_surface.get_width() + 15


    def _draw_bottom_color_key(self, screen, x, y, width, bar_height, font, small_font):
        pygame.draw.rect(screen, (25, 25, 25), (x, y, width, bar_height)) # Bar background
        current_x = x + 10
        item_spacing = 10 # Spacing between color key items
        rect_size = 10 # Size of the color square

        for color_val, label_text in self.legend_colors:
            # Draw color square
            pygame.draw.rect(screen, color_val, (current_x, y + (bar_height - rect_size) // 2, rect_size, rect_size))
            current_x += rect_size + 5 # Space after color square
            
            # Draw label text
            label_surface = small_font.render(label_text, True, (180, 180, 180))
            if current_x + label_surface.get_width() < x + width -10: # Check if fits
                screen.blit(label_surface, (current_x, y + (bar_height - label_surface.get_height()) // 2))
                current_x += label_surface.get_width() + item_spacing
            else:
                break # Stop if no more space
