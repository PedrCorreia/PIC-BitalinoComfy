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
                (120, (255, 140, 0)) # red
            ]),
            ("RR", "RR_METRIC", 8, 40, [  # RR = respiration rate (breaths/min)
                (5, (0, 120, 255)),
                (10, (0, 200, 80)),
                (20, (255, 220, 0)),
                (25, (255, 140, 0)),
                (40, (255, 40, 40)),
            ]),
            # SCL = Skin Conductance Level (tonic, slow-changing baseline, in uS)
            ("SCL", "SCL_METRIC", 0, 40, [
                (0, (0, 120, 255)),     # sleep (blue)
                (2, (0, 200, 80)),      # relaxed (green)
                (10, (255, 220, 0)),     # normal (yellow)
                (15, (255, 140, 0)),    # aroused (orange)
                (20, (255, 40, 40)),    # stressed (red)
            ]),
            ("SCR Freq", "scr_frequency", 0, 2.5, [
                (0, (0, 120, 255)),
                (1, (0, 200, 80)),
                (1.5, (255, 220, 0)),
                (2, (255, 140, 0)),
                (2.5, (255, 40, 40)),
            ]),
            
        ]
        
        # Arousal metrics configurations
        self.arousal_metrics = [
            ("RR Arousal ", "RR_AROUSAL_METRIC"),
            ("ECG Arousal", "ECG_AROUSAL_METRIC"),
            ("EDA Arousal", "EDA_AROUSAL_METRIC"),
            ("Overall", "OVERALL_AROUSAL_METRIC")
        ]

    def _find_matching_signal(self, pattern):
        # Match any signal id in the registry that contains the pattern (case-insensitive, allows suffixes/prefixes)
        all_ids = self.signal_registry.get_all_signal_ids() if hasattr(self.signal_registry, 'get_all_signal_ids') else [] # Use signal_registry
        print(f"[MetricsView DEBUG] _find_matching_signal: Searching for pattern '{pattern}' among {len(all_ids)} signals: {all_ids}") # DEBUG PRINT
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        for sid in all_ids:
            if regex.search(str(sid)):
                print(f"[MetricsView DEBUG] _find_matching_signal: Found match for '{pattern}': {sid}") # DEBUG PRINT
                return self.signal_registry.get_signal(sid) # Use signal_registry
        print(f"[MetricsView DEBUG] _find_matching_signal: No matching signal found for '{pattern}'") # DEBUG PRINT
        return None

    def draw(self, screen, x, y, width, height, window_sec=30):
        n_metrics = len(self.metric_configs)
        n_cols = 2
        n_rows = 2
        margin = SECTION_MARGIN  # Use SECTION_MARGIN for consistency
        total_cells = n_cols * n_rows
        # Allocate space for the arousal table on the right (25% of width)
        table_width = width * 0.25
        main_width = width - table_width
        
        # Center the grid in the available area (75% of width)
        cell_size = min((main_width - (n_cols + 1) * margin) // n_cols, (height - (n_rows + 1) * margin) // n_rows)
        cell_size = cell_size    # Double the cell height (and width, since grid is square)
        grid_w = n_cols * cell_size + (n_cols + 1) * margin
        grid_h = n_rows * cell_size + (n_rows + 1) * margin
        grid_x = x + (main_width - grid_w) // 2
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
                print(f"[MetricsView DEBUG] Processing metric: Label='{label}', SignalPattern='{sig_pattern}'") # DEBUG PRINT
                sig = self._find_matching_signal(sig_pattern)
                print(f"[MetricsView DEBUG] Metric '{label}': Retrieved signal data: {'Exists' if sig else 'None'}") # DEBUG PRINT
                if sig and isinstance(sig, dict):
                    print(f"[MetricsView DEBUG] Metric '{label}': Signal keys: {list(sig.keys())}, t_len: {len(sig.get('t', []))}, v_len: {len(sig.get('v', []))}, last: {sig.get('last', 'N/A')}")
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
                    print(f"[MetricsView DEBUG] Metric '{label}': Data is valid for plotting.") # DEBUG PRINT
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
            except Exception as e:
                print(f"[MetricsView DEBUG] Metric '{label}': Exception during plot drawing: {e}") # DEBUG PRINT
                pass
            if not plot_drawn:
                print(f"[MetricsView DEBUG] Metric '{label}': Plot not drawn (no data or error).") # DEBUG PRINT
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
                    #print(f"[HR BAR DEBUG] sig dict for {label}: keys={list(sig.keys())}")
                    if "last" in sig and isinstance(sig["last"], (float, int, np.floating, np.integer)):
                        last_val = float(sig["last"])
                        print(f"[MetricsView DEBUG] Metric '{label}': last_val from sig['last']: {last_val}") # DEBUG PRINT
                    elif "last" in sig and isinstance(sig["last"], (list, np.ndarray)) and len(sig["last"]):
                        last_val = float(sig["last"][-1])
                        print(f"[MetricsView DEBUG] Metric '{label}': last_val from sig['last'][-1]: {last_val}") # DEBUG PRINT
                    elif "v" in sig and isinstance(sig["v"], (list, np.ndarray)) and len(sig["v"]):
                        last_val = float(sig["v"][-1])
                        print(f"[MetricsView DEBUG] Metric '{label}': last_val from sig['v'][-1]: {last_val}") # DEBUG PRINT
                    else:
                        print(f"[MetricsView DEBUG] Metric '{label}': 'last_val' could not be extracted from signal data. Keys: {list(sig.keys()) if sig else 'None'}") # DEBUG PRINT
                else:
                    print(f"[MetricsView DEBUG] Metric '{label}': Signal data is None or not a dict, cannot extract 'last_val'.") # DEBUG PRINT
            except Exception as e:
                print(f"[MetricsView DEBUG] Metric '{label}': Exception extracting last_val: {e}") # DEBUG PRINT
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
        
        # Draw the arousal metrics table on the right side
        self._draw_arousal_table(screen, x + main_width, y, table_width, height, font, small_font)

    def _draw_arousal_table(self, screen, x, y, width, height, font, small_font):
        """
        Draw a table of current arousal values from different biosignals
        """
        # Draw table header
        header_height = 40
        row_height = 32
        table_margin = 10
        
        # Draw table background
        table_rect = pygame.Rect(x + table_margin, y + table_margin, 
                               width - (2 * table_margin), height - (2 * table_margin))
        pygame.draw.rect(screen, (30, 30, 30), table_rect, border_radius=8)
        
        # Draw table title
        title = "Arousal"
        title_surface = font.render(title, True, (255, 255, 255))
        screen.blit(title_surface, (table_rect.centerx +10- title_surface.get_width()//2, 
                       table_rect.y + 10))

        # Draw subtitle below the title
        subtitle = "Values"
        subtitle_surface = small_font.render(subtitle, True, (200, 200, 200))
        screen.blit(subtitle_surface, (table_rect.centerx - subtitle_surface.get_width()//2,
                           table_rect.y + 10 + title_surface.get_height() + 2))
        # Draw header line
        pygame.draw.line(screen, (100, 100, 100), 
                       (table_rect.x + 10, table_rect.y + header_height - 5),
                       (table_rect.right - 10, table_rect.y + header_height - 5), 2)
        
        # Draw column headers with better spacing
        source_header = small_font.render("Source", True, (200, 200, 200))
        value_header = small_font.render("Value", True, (200, 200, 200))
        col1_x = table_rect.x + 20
        col2_x = table_rect.right - 120  # Move column further from right edge for more spacing
        header_y = table_rect.y + header_height + 5
        screen.blit(source_header, (col1_x, header_y))
        screen.blit(value_header, (col2_x, header_y))
        
        # Draw legend for color mapping
        legend_y = table_rect.bottom - 100
        legend_spacing = 15
        legend_text = small_font.render("Color Key:", True, (200, 200, 200))
        screen.blit(legend_text, (col1_x, legend_y))
        
        # Draw color legend squares with labels
        legend_colors = [
            ((0, 120, 255), "Sleep (0.0-0.2)"),
            ((0, 200, 80), "Relaxed (0.2-0.4)"),
            ((255, 220, 0), "Normal (0.4-0.6)"),
            ((255, 140, 0), "Aroused (0.6-0.8)"),
            ((255, 40, 40), "Stressed (0.8-1.0)")
        ]
        
        for i, (color, label) in enumerate(legend_colors):
            y_pos = legend_y + 20 + i * legend_spacing
            pygame.draw.rect(screen, color, (col1_x, y_pos, 10, 10))
            label_surf = small_font.render(label, True, (180, 180, 180))
            screen.blit(label_surf, (col1_x + 15, y_pos - 2))

        # Draw each arousal row
        for i, (label, signal_id) in enumerate(self.arousal_metrics):
            row_y = header_y + ((i + 1) * row_height)
            
            print(f"[MetricsView DEBUG] Arousal Table: Processing '{label}', looking for SignalID '{signal_id}'") # DEBUG PRINT
            # Draw the row label
            label_surface = small_font.render(label, True, (180, 180, 180))
            screen.blit(label_surface, (col1_x, row_y))
            
            # Get the arousal value if available
            arousal_value = None
            sig = self._find_matching_signal(signal_id)
            print(f"[MetricsView DEBUG] Arousal '{label}': Retrieved signal for '{signal_id}': {'Exists' if sig else 'None'}") # DEBUG PRINT
            
            # Try to get the arousal value from signal or its metadata
            if sig and isinstance(sig, dict):
                print(f"[MetricsView DEBUG] Arousal '{label}': Signal keys for '{signal_id}': {list(sig.keys())}") # DEBUG PRINT
                # First check if it's directly in the signal
                if "last" in sig and sig["last"] is not None:
                    arousal_value = sig["last"]
                    print(f"[MetricsView DEBUG] Arousal '{label}': Using 'last' value: {arousal_value}") # DEBUG PRINT
                elif "v" in sig and len(sig["v"]) > 0 and sig["v"][-1] is not None:
                    arousal_value = sig["v"][-1]
                    print(f"[MetricsView DEBUG] Arousal '{label}': Using 'v[-1]' value: {arousal_value}") # DEBUG PRINT
                # If not found, try to get it from metadata
                elif "meta" in sig and isinstance(sig["meta"], dict) and "arousal_value" in sig["meta"] and sig["meta"]["arousal_value"] is not None:
                    arousal_value = sig["meta"]["arousal_value"]
                    print(f"[MetricsView DEBUG] Arousal '{label}': Using metadata 'arousal_value': {arousal_value}") # DEBUG PRINT
                else:
                    print(f"[MetricsView DEBUG] Arousal '{label}': Arousal value not found directly in signal keys or was None.") # DEBUG PRINT
                    
                # Try to get metadata directly from registry if available
                if arousal_value is None and hasattr(self.signal_registry, 'get_signal_metadata'): 
                    try:
                        meta = self.signal_registry.get_signal_metadata(signal_id) 
                        print(f"[MetricsView DEBUG] Arousal '{label}': Fetched registry metadata for '{signal_id}': {meta is not None}") # DEBUG PRINT
                        if meta and "arousal_value" in meta and meta["arousal_value"] is not None:
                            arousal_value = meta["arousal_value"]
                            print(f"[MetricsView DEBUG] Arousal '{label}': Using registry metadata 'arousal_value': {arousal_value}") # DEBUG PRINT
                        else:
                            print(f"[MetricsView DEBUG] Arousal '{label}': 'arousal_value' not in registry metadata, or metadata is None, or arousal_value in meta is None.") # DEBUG PRINT
                    except Exception as e:
                        print(f"[MetricsView DEBUG] Arousal '{label}': Exception getting metadata for {signal_id}: {e}") # DEBUG PRINT
                elif arousal_value is None:
                    print(f"[MetricsView DEBUG] Arousal '{label}': signal_registry does not have get_signal_metadata or arousal_value was already found (or still None).") # DEBUG PRINT
            else:
                print(f"[MetricsView DEBUG] Arousal '{label}': Signal data for '{signal_id}' is None or not a dict.") # DEBUG PRINT
            
            # Draw the arousal value
            if arousal_value is not None and isinstance(arousal_value, (float, int)) and np.isfinite(arousal_value):
                print(f"[MetricsView DEBUG] Arousal '{label}': Drawing value {float(arousal_value):.2f}") # DEBUG PRINT
                # Color based on arousal level (use consistent colors from metric_configs)
                level = float(arousal_value)
                # Map the arousal value (0-1) to the color scheme used in metrics
                if level < 0.2:
                    color = (0, 120, 255)    # Blue - sleep
                elif level < 0.4:
                    color = (0, 200, 80)     # Green - relaxed
                elif level < 0.6:
                    color = (255, 220, 0)    # Yellow - normal
                elif level < 0.8:
                    color = (255, 140, 0)    # Orange - aroused
                else:
                    color = (255, 40, 40)    # Red - stressed
                    
                # Format value to always show 2 decimal places for consistency
                value_text = f"{level:.2f}"
                value_surface = small_font.render(value_text, True, color)
                # Center the value in its column
                text_x = col2_x + (80 - value_surface.get_width()) // 2
                screen.blit(value_surface, (text_x, row_y))
                
                # Draw a colored indicator dot
                dot_radius = 5
                dot_x = col2_x - 20  # Move dot further away from text for better spacing
                dot_y = row_y + small_font.get_height() // 2
                pygame.draw.circle(screen, color, (dot_x, dot_y), dot_radius)
            else:
                # N/A if no value
                na_text = "N/A"
                na_surface = small_font.render(na_text, True, (150, 150, 150))
                # Center the text
                text_x = col2_x + (205 - na_surface.get_width()) // 2  # Updated to match value text positioning (80)
                screen.blit(na_surface, (text_x, row_y))
