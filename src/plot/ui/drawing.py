import pygame
import numpy as np
from src.plot.constants import TEXT_COLOR

def draw_signal_plot(screen, font, signal, x, y, w, h, show_time_markers=False, window_sec=None,mode='default'):

    if mode == 'default':
        #print("[DRAW_SIGNAL_PLOT_DEBUG] Default mode entered", flush=True)
        t, v, meta = signal['t'], signal['v'], signal.get('meta', {})
        t = np.array(t)
        v = np.array(v)
        if meta is None:
            meta = {}
        if len(t) < 2 or len(v) < 2:
            pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 2)
            label = meta.get('name', signal.get('id', ''))
            label_surface = font.render(label, True, TEXT_COLOR)
            no_data_surface = font.render("No data", True, (180, 80, 80))
            screen.blit(label_surface, (x + 10, y + 10))
            screen.blit(no_data_surface, (x + w//2 - no_data_surface.get_width()//2, y + h//2 - no_data_surface.get_height()//2))
            return
        vmin, vmax = np.min(v), np.max(v)
        if vmax == vmin:
            vmax = vmin + 1
        # Always use the original timestamps for plotting and peak placement
        t_plot = t
        v_plot = v
        window_min = t_plot[0] if len(t_plot) > 0 else 0
        window_max = t_plot[-1] if len(t_plot) > 0 else 0
        if window_sec is not None and len(t_plot) > 1:
            window_max = t_plot[-1]
            window_min = window_max - window_sec
            indices = np.where((t_plot >= window_min) & (t_plot <= window_max))[0]
            if len(indices) < 2:
                t_plot = t_plot[-2:]
                v_plot = v_plot[-2:]
                window_min = t_plot[0] if len(t_plot) > 0 else 0
                window_max = t_plot[-1] if len(t_plot) > 0 else 0
            else:
                t_plot = t_plot[indices]
                v_plot = v_plot[indices]
        points = [(x + int((t_plot[j] - window_min) / (window_max - window_min) * w), y + h - int((v_plot[j]-vmin)/(vmax-vmin)*h)) for j in range(len(t_plot))] if len(t_plot) > 1 and window_max > window_min else []
        # Color handling: support string color names as well as RGB tuples
        color = meta.get('color', (255,255,255))
        if isinstance(color, str):
            try:
                if color.startswith('#') and len(color) == 7:
                    color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                else:
                    color_map = {
                        'red': (255, 51, 51),
                        'orange': (255, 153, 51),
                        'yellow': (255, 255, 51),
                        'blue': (51, 153, 255),
                    }
                    color = color_map.get(color.lower(), (255,255,255))
            except Exception:
                color = (255, 0, 255)  # fallback magenta for error
        if not (isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
            color = (255, 255, 255)  # fallback to white for error
        # Draw the signal lines if we have at least 2 points
        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, 2)
            # Draw peak markers if available in metadata
            if meta.get('show_peaks', False) and 'peak_timestamps' in meta:
                peak_marker = meta.get('peak_marker', 'x')  # Default to 'x' if not specified
                peak_marker_size = 8  # Size of the peak marker
                peak_marker_color = (255, 255, 255)  # White color for peak markers
                peak_times = np.array(meta['peak_timestamps'])
                # Only plot peaks that are within the visible window
                if len(peak_times) > 0 and window_max > window_min:
                    in_window = (peak_times >= window_min) & (peak_times <= window_max)
                    peak_times = peak_times[in_window]
                    for pt in peak_times:
                        # Interpolate y value for the peak timestamp
                        peak_x = x + int((pt - window_min) / (window_max - window_min) * w)
                        peak_y_val = np.interp(pt, t_plot, v_plot)
                        peak_y = y + h - int((peak_y_val-vmin)/(vmax-vmin)*h)
                        if peak_marker == 'x':
                            pygame.draw.line(screen, peak_marker_color, 
                                            (peak_x - peak_marker_size//2, peak_y - peak_marker_size//2),
                                            (peak_x + peak_marker_size//2, peak_y + peak_marker_size//2), 2)
                            pygame.draw.line(screen, peak_marker_color, 
                                            (peak_x - peak_marker_size//2, peak_y + peak_marker_size//2),
                                            (peak_x + peak_marker_size//2, peak_y - peak_marker_size//2), 2)
                        elif peak_marker == 'o':
                            pygame.draw.circle(screen, peak_marker_color, (peak_x, peak_y), peak_marker_size//2, 2)
                        elif peak_marker == '+':
                            pygame.draw.line(screen, peak_marker_color, 
                                            (peak_x, peak_y - peak_marker_size//2),
                                            (peak_x, peak_y + peak_marker_size//2), 2)
                            pygame.draw.line(screen, peak_marker_color, 
                                            (peak_x - peak_marker_size//2, peak_y),
                                            (peak_x + peak_marker_size//2, peak_y), 2)
                        else:
                            pygame.draw.rect(screen, peak_marker_color, 
                                            (peak_x - peak_marker_size//2, peak_y - peak_marker_size//2, 
                                            peak_marker_size, peak_marker_size), 2)
        label = meta.get('name', signal.get('id', ''))
        label_surface = font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (x + 10, y + 10))
        if show_time_markers and len(t_plot) > 1:
            pygame.draw.line(screen, (200, 200, 80), (x, y), (x, y + h), 2)
            pygame.draw.line(screen, (80, 200, 200), (x + w - 1, y), (x + w - 1, y + h), 2)
            min_time = t_plot[0] if len(t_plot) > 0 else 0
            max_time = t_plot[-1] if len(t_plot) > 0 else 0
            min_label = font.render(f"{min_time:.1f}s", True, (200, 200, 80))
            max_label = font.render(f"{max_time:.1f}s", True, (80, 200, 200))
            screen.blit(min_label, (x + 2, y + h - 22))
            screen.blit(max_label, (x + w - max_label.get_width() - 2, y + h - 22))
    if mode == 'eda':
        #print("[DRAW_SIGNAL_PLOT_DEBUG] EDA mode entered", flush=True) # DEBUG PRINT
        
        #print(f"[EDA_DRAW_DEBUG] Initial signal object (type: {type(signal)}): {signal}", flush=True) # DEBUG PRINT

        # --- Retrieve data with robust fallbacks and type checking ---
        # General metadata (like name, show_peaks) comes from the main meta object from the signal
        meta = signal.get('meta', {}) if isinstance(signal, dict) else {}
        if meta is None: meta = {} # Ensure meta is a dict

        original_data_dict = meta.get('_original_data_dict', {})
        if original_data_dict is None: original_data_dict = {} # Ensure it's a dict

        # 't' comes from the main signal dict, as prepared by PlotRegistry
        raw_t_data = signal.get('t') if isinstance(signal, dict) else None

        # Phasic, tonic, and peak_indices come from the _original_data_dict
        raw_phasic_data = original_data_dict.get('phasic_norm')
        raw_tonic_data = original_data_dict.get('tonic_norm')
        raw_peak_indices_data = original_data_dict.get('peak_indices')
        
        #print(f"[EDA_DRAW_DEBUG] Raw data retrieved: t is None: {raw_t_data is None}, p is None: {raw_phasic_data is None}, to is None: {raw_tonic_data is None}, pi is None: {raw_peak_indices_data is None}", flush=True)

        # 'meta' is already defined and ensured to be a dict from signal.get('meta',...)
        show_peaks_meta = meta.get('show_peaks', True) # Uses the correct 'meta' for plot attributes

        # Helper to ensure data is a list, or an empty list if None or not a list/ndarray
        def ensure_list(data_item):
            if data_item is None:
                return []
            if isinstance(data_item, (list, np.ndarray)):
                return list(data_item) # Convert ndarray to list here
            return [] # Fallback for other unexpected types

        t_data = ensure_list(raw_t_data)
        phasic_data = ensure_list(raw_phasic_data)
        tonic_data = ensure_list(raw_tonic_data)
        peak_indices_data = ensure_list(raw_peak_indices_data)
        
        #print(f"[EDA_DRAW_DEBUG] Raw data after ensure_list: t={type(t_data)}, p={type(phasic_data)}, to={type(tonic_data)}, pi={type(peak_indices_data)}", flush=True) # DEBUG PRINT
        #print(f"[EDA_DRAW_DEBUG] Raw data lengths before np.asarray: t_len={len(t_data) if t_data is not None else 'None'}, p_len={len(phasic_data) if phasic_data is not None else 'None'}, to_len={len(tonic_data) if tonic_data is not None else 'None'}, pi_len={len(peak_indices_data) if peak_indices_data is not None else 'None'}", flush=True)
        
        # --- Convert to NumPy arrays for processing (after ensuring they are lists) ---
        # This is where we ensure they are numpy arrays for consistent processing.
        # If ensure_list returned [], np.asarray([]) is fine.
        t_data_np = np.asarray(t_data, dtype=float) # Assuming time data should be float
        phasic_data_np = np.asarray(phasic_data, dtype=float) # Assuming numeric data
        tonic_data_np = np.asarray(tonic_data, dtype=float)   # Assuming numeric data
        peak_indices_data_np = np.asarray(peak_indices_data, dtype=int) # Peak indices should be int

        #print(f"[EDA_DRAW_DEBUG] NumPy data types: t={t_data_np.dtype}, p={phasic_data_np.dtype}, to={tonic_data_np.dtype}, pi={peak_indices_data_np.dtype}", flush=True) # DEBUG PRINT
        #print(f"[EDA_DRAW_DEBUG] NumPy data shapes: t={t_data_np.shape}, p={phasic_data_np.shape}, to={tonic_data_np.shape}, pi={peak_indices_data_np.shape}", flush=True) # DEBUG PRINT
        #print(f"[EDA_DRAW_DEBUG] NumPy data content (first 5 if available): t={t_data_np[:5]}, p={phasic_data_np[:5]}, to={tonic_data_np[:5]}, pi={peak_indices_data_np[:5]}", flush=True)

        # --- Basic Data Validation (using NumPy arrays) ---
        display_error_message = None
        if not (t_data_np.ndim == 1 and phasic_data_np.ndim == 1 and tonic_data_np.ndim == 1 and peak_indices_data_np.ndim == 1):
            display_error_message = "EDA data not 1D"
        elif len(t_data_np) < 2 or len(phasic_data_np) < 2 or len(tonic_data_np) < 2:
            display_error_message = "Insufficient EDA data points"
        elif not (len(t_data_np) == len(phasic_data_np) == len(tonic_data_np)):
            display_error_message = "Mismatched EDA data lengths"
        
        if display_error_message:
            pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 2)
            label = meta.get('name', signal.get('id', 'EDA Plot'))
            label_surface = font.render(label, True, TEXT_COLOR)
            error_surface = font.render(display_error_message, True, (255, 100, 100))
            screen.blit(label_surface, (x + 10, y + 10))
            screen.blit(error_surface, (x + w//2 - error_surface.get_width()//2, y + h//2 - error_surface.get_height()//2))
            return
            
        # Windowing (data from eda.py is already windowed, but this allows further zoom if window_sec is passed)
        t_plot = t_data_np
        phasic_plot = phasic_data_np
        tonic_plot = tonic_data_np
        
        current_window_min_time = t_plot[0] if len(t_plot) > 0 else 0
        current_window_max_time = t_plot[-1] if len(t_plot) > 0 else 0

        if window_sec is not None and len(t_plot) > 1:
            # Apply a potentially smaller window than what eda.py provided
            target_max_time = t_plot[-1]
            target_min_time = max(target_max_time - window_sec, t_plot[0])
            
            indices = np.where((t_plot >= target_min_time) & (t_plot <= target_max_time))[0]
            
            if len(indices) < 2: 
                num_fallback_points = min(len(t_plot), 5) 
                t_plot = t_plot[-num_fallback_points:]
                phasic_plot = phasic_plot[-num_fallback_points:]
                tonic_plot = tonic_plot[-num_fallback_points:]
            else:
                t_plot = t_plot[indices]
                phasic_plot = phasic_plot[indices]
                tonic_plot = tonic_plot[indices]
            
            current_window_min_time = t_plot[0] if len(t_plot) > 0 else 0
            current_window_max_time = t_plot[-1] if len(t_plot) > 0 else 0

        # --- Dynamic Normalization of Phasic and Tonic components for plotting ---
        min_phasic_val, max_phasic_val, delta_phasic = 0.0, 1.0, 1.0
        if len(phasic_plot) >= 2:
            min_phasic_val = np.min(phasic_plot)
            max_phasic_val = np.max(phasic_plot)
            delta_phasic = max_phasic_val - min_phasic_val
        if delta_phasic < 1e-6: # Avoid division by zero or extreme scaling for flat lines
            delta_phasic = 1.0 # Effectively makes norm_val 0 if min/max are same, or use min_val
            # For a truly flat line (min_phasic_val == max_phasic_val), norm_val will be 0.
            # To center a flat line, norm_val should be 0.5. Let's adjust for that.

        min_tonic_val, max_tonic_val, delta_tonic = 0.0, 1.0, 1.0
        if len(tonic_plot) >= 2:
            min_tonic_val = np.min(tonic_plot)
            max_tonic_val = np.max(tonic_plot)
            delta_tonic = max_tonic_val - min_tonic_val
        if delta_tonic < 1e-6:
            delta_tonic = 1.0
            
        # pygame.draw.rect(screen, (40, 40, 40), (x, y, w, h)) # Background - REMOVED to match default plot background
        # pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 2) # Border - REMOVED to avoid "box outliner"
            
        # Generate points for Phasic (orange)
        phasic_color = (255, 170, 0) # Orange
        phasic_points = []
        if len(t_plot) > 1 and current_window_max_time > current_window_min_time:
            for j in range(len(t_plot)):
                time_val = t_plot[j]
                current_phasic_val = phasic_plot[j]
                
                norm_val = 0.5 # Default for flat line in middle
                if delta_phasic > 1e-6:
                    norm_val = (current_phasic_val - min_phasic_val) / delta_phasic
                elif min_phasic_val == max_phasic_val : # Truly flat, check if it's not at 0 to avoid norm_val = 0
                     # if min_phasic_val != 0, it's a flat line not at 0.
                     # if min_phasic_val == 0, it's flat at 0.
                     # The previous logic (delta_phasic=1) would make (val - min_val)/1 = 0.
                     # So, if min_phasic_val == max_phasic_val, norm_val = 0.5 to center it.
                     pass # norm_val is already 0.5

                norm_val = np.clip(norm_val, 0.0, 1.0) # Ensure it's within [0,1]

                px = x + int(((time_val - current_window_min_time) / (current_window_max_time - current_window_min_time)) * w)
                py = y + h - int(norm_val * h) 
                phasic_points.append((px, py))

        # Generate points for Tonic (green)
        tonic_color = (0, 220, 0) # Green
        tonic_points = []
        if len(t_plot) > 1 and current_window_max_time > current_window_min_time:
            for j in range(len(t_plot)):
                time_val = t_plot[j]
                current_tonic_val = tonic_plot[j]

                norm_val = 0.5 # Default for flat line
                if delta_tonic > 1e-6:
                    norm_val = (current_tonic_val - min_tonic_val) / delta_tonic
                elif min_tonic_val == max_tonic_val:
                    pass # norm_val is already 0.5
                
                norm_val = np.clip(norm_val, 0.0, 1.0)

                px = x + int(((time_val - current_window_min_time) / (current_window_max_time - current_window_min_time)) * w)
                py = y + h - int(norm_val * h)
                tonic_points.append((px, py))
        
        if len(phasic_points) >= 2:
            pygame.draw.lines(screen, phasic_color, False, phasic_points, 2)
            
        if len(tonic_points) >= 2:
            pygame.draw.lines(screen, tonic_color, False, tonic_points, 2)
            
        # Draw peak markers on phasic_plot using peak_indices_data
        if show_peaks_meta and len(peak_indices_data_np) > 0 and len(t_plot) > 1 and current_window_max_time > current_window_min_time:
            peak_marker_size = 8
            peak_marker_color = (255, 255, 255) # White X

            valid_original_indices = peak_indices_data_np[(peak_indices_data_np >= 0) & (peak_indices_data_np < len(t_data_np))]
            
            if len(valid_original_indices) > 0:
                original_peak_times = t_data_np[valid_original_indices]
                original_peak_phasic_values = phasic_data_np[valid_original_indices] 

                peaks_in_current_view_mask = (original_peak_times >= current_window_min_time) & (original_peak_times <= current_window_max_time)
                
                visible_peak_times = original_peak_times[peaks_in_current_view_mask]
                visible_peak_phasic_values = original_peak_phasic_values[peaks_in_current_view_mask]

                for i in range(len(visible_peak_times)):
                    peak_time_val = visible_peak_times[i]
                    peak_phasic_val = visible_peak_phasic_values[i]

                    px_peak = x + int(((peak_time_val - current_window_min_time) / (current_window_max_time - current_window_min_time)) * w)
                    
                    norm_peak_val = 0.5 # Default for flat line
                    if delta_phasic > 1e-6:
                        norm_peak_val = (peak_phasic_val - min_phasic_val) / delta_phasic
                    elif min_phasic_val == max_phasic_val: # If original data for peak was on a flat line
                         pass # norm_peak_val is already 0.5

                    norm_peak_val = np.clip(norm_peak_val, 0.0, 1.0)
                    py_peak = y + h - int(norm_peak_val * h)
                    
                    # Draw X marker
                    pygame.draw.line(screen, peak_marker_color, (px_peak - peak_marker_size // 2, py_peak - peak_marker_size // 2), (px_peak + peak_marker_size // 2, py_peak + peak_marker_size // 2), 2)
                    pygame.draw.line(screen, peak_marker_color, (px_peak + peak_marker_size // 2, py_peak - peak_marker_size // 2), (px_peak - peak_marker_size // 2, py_peak + peak_marker_size // 2), 2)
        
        # Add labels and legend (label first, then time markers, then legend)
        label = meta.get('name', signal.get('id', 'EDA Plot'))
        label_surface = font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (x + 10, y + 10))

        # --- Draw Time Markers (like in default mode) ---
        if show_time_markers and len(t_plot) > 1:
            # Use actual min/max times from the currently plotted window
            min_time_val = t_plot[0]
            max_time_val = t_plot[-1]

            # Left vertical bar (yellowish)
            pygame.draw.line(screen, (200, 200, 80), (x, y), (x, y + h), 2)
            # Right vertical bar (cyanish)
            pygame.draw.line(screen, (80, 200, 200), (x + w - 1, y), (x + w - 1, y + h), 2) # -1 to be fully within w

            min_label_text = f"{min_time_val:.1f}s"
            max_label_text = f"{max_time_val:.1f}s"
            
            min_time_label_surface = font.render(min_label_text, True, (200, 200, 80))
            max_time_label_surface = font.render(max_label_text, True, (80, 200, 200))
            
            screen.blit(min_time_label_surface, (x + 2, y + h - min_time_label_surface.get_height() - 2))
            screen.blit(max_time_label_surface, (x + w - max_time_label_surface.get_width() - 2, y + h - max_time_label_surface.get_height() - 2))
        
        legend_font = font 
        legend_y_start = y + 10 + label_surface.get_height() + 5

        phasic_legend_surface = legend_font.render("Phasic", True, phasic_color)
        screen.blit(phasic_legend_surface, (x + 10, legend_y_start))
        
        tonic_legend_surface = legend_font.render("Tonic", True, tonic_color)
        # Adjusted y position for tonic legend to be on the same line if space, or next line.
        screen.blit(tonic_legend_surface, (x + 10 + phasic_legend_surface.get_width() + 15, legend_y_start))
