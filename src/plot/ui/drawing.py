import pygame
import numpy as np
from src.plot.constants import TEXT_COLOR

def draw_signal_plot(screen, font, signal, x, y, w, h, show_time_markers=False, window_sec=None,mode='default'):
    if mode == 'default':
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
        # EDA mode plots tonic (SCL) and phasic (SCR) components
        t = signal.get('t', [])
        scl = signal.get('scl', [])  # Tonic component
        scr = signal.get('scr', [])  # Phasic component
        peaks = signal.get('peaks', [])  # SCR peak timestamps
        meta = signal.get('meta', {})
        
        if len(t) < 2 or len(scl) < 2 or len(scr) < 2:
            pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 2)
            label = meta.get('name', signal.get('id', ''))
            label_surface = font.render(label, True, TEXT_COLOR)
            no_data_surface = font.render("No EDA data", True, (180, 80, 80))
            screen.blit(label_surface, (x + 10, y + 10))
            screen.blit(no_data_surface, (x + w//2 - no_data_surface.get_width()//2, y + h//2 - no_data_surface.get_height()//2))
            return
            
        # Convert to numpy arrays for easier processing
        t = np.array(t)
        scl = np.array(scl)
        scr = np.array(scr)
        
        # Determine window boundaries
        t_plot = t
        window_min = t_plot[0] if len(t_plot) > 0 else 0
        window_max = t_plot[-1] if len(t_plot) > 0 else 0
        
        if window_sec is not None and len(t_plot) > 1:
            window_max = t_plot[-1]
            window_min = window_max - window_sec
            indices = np.where((t_plot >= window_min) & (t_plot <= window_max))[0]
            if len(indices) < 2:
                t_plot = t_plot[-2:]
                scl_plot = scl[-2:]
                scr_plot = scr[-2:]
                window_min = t_plot[0] if len(t_plot) > 0 else 0
                window_max = t_plot[-1] if len(t_plot) > 0 else 0
            else:
                t_plot = t_plot[indices]
                scl_plot = scl[indices]
                scr_plot = scr[indices]
        else:
            scl_plot = scl
            scr_plot = scr
            
        # Calculate value ranges for both components
        all_values = np.concatenate([scl_plot, scr_plot])
        vmin, vmax = np.min(all_values), np.max(all_values)
        if vmax == vmin:
            vmax = vmin + 1
            
        # Draw background and frame
        pygame.draw.rect(screen, (40, 40, 40), (x, y, w, h))
        pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 2)
            
        # Generate points for SCL (tonic) - blue color
        scl_color = (51, 153, 255)  # Blue for SCL
        scl_points = [(x + int((t_plot[j] - window_min) / (window_max - window_min) * w), 
                      y + h - int((scl_plot[j]-vmin)/(vmax-vmin)*h)) 
                     for j in range(len(t_plot))] if len(t_plot) > 1 and window_max > window_min else []
                     
        # Generate points for SCR (phasic) - orange color
        scr_color = (255, 153, 51)  # Orange for SCR
        scr_points = [(x + int((t_plot[j] - window_min) / (window_max - window_min) * w), 
                      y + h - int((scr_plot[j]-vmin)/(vmax-vmin)*h)) 
                     for j in range(len(t_plot))] if len(t_plot) > 1 and window_max > window_min else []
        
        # Draw the SCL line
        if len(scl_points) >= 2:
            pygame.draw.lines(screen, scl_color, False, scl_points, 2)
            
        # Draw the SCR line
        if len(scr_points) >= 2:
            pygame.draw.lines(screen, scr_color, False, scr_points, 2)
            
        # Draw SCR peak markers if available
        if peaks and len(peaks) > 0:
            peak_marker_size = 8
            peak_marker_color = (255, 255, 255)  # White color for peak markers
            peak_times = np.array(peaks)
            
            # Only plot peaks that are within the visible window
            if len(peak_times) > 0 and window_max > window_min:
                in_window = (peak_times >= window_min) & (peak_times <= window_max)
                peak_times = peak_times[in_window]
                
                for pt in peak_times:
                    # Interpolate y value for the peak timestamp in SCR component
                    peak_x = x + int((pt - window_min) / (window_max - window_min) * w)
                    try:
                        # Find SCR value at peak time
                        peak_y_val = np.interp(pt, t_plot, scr_plot)
                        peak_y = y + h - int((peak_y_val-vmin)/(vmax-vmin)*h)
                        
                        # Draw an X marker for peaks
                        pygame.draw.line(screen, peak_marker_color, 
                                        (peak_x - peak_marker_size//2, peak_y - peak_marker_size//2),
                                        (peak_x + peak_marker_size//2, peak_y + peak_marker_size//2), 2)
                        pygame.draw.line(screen, peak_marker_color, 
                                        (peak_x - peak_marker_size//2, peak_y + peak_marker_size//2),
                                        (peak_x + peak_marker_size//2, peak_y - peak_marker_size//2), 2)
                    except:
                        pass  # Skip if interpolation fails
        
        # Add labels and legend
        label = meta.get('name', signal.get('id', ''))
        label_surface = font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (x + 10, y + 10))
        
        # Add legend for SCL and SCR
        scl_label = font.render("SCL", True, scl_color)
        scr_label = font.render("SCR", True, scr_color)
        screen.blit(scl_label, (x + 10, y + 30))
        screen.blit(scr_label, (x + 10 + scl_label.get_width() + 20, y + 30))
        
        # Add time markers if requested
        if show_time_markers and len(t_plot) > 1:
            pygame.draw.line(screen, (200, 200, 80), (x, y), (x, y + h), 2)
            pygame.draw.line(screen, (80, 200, 200), (x + w - 1, y), (x + w - 1, y + h), 2)
            min_time = t_plot[0] if len(t_plot) > 0 else 0
            max_time = t_plot[-1] if len(t_plot) > 0 else 0
            min_label = font.render(f"{min_time:.1f}s", True, (200, 200, 80))
            max_label = font.render(f"{max_time:.1f}s", True, (80, 200, 200))
            screen.blit(min_label, (x + 2, y + h - 22))
            screen.blit(max_label, (x + w - max_label.get_width() - 2, y + h - 22))
