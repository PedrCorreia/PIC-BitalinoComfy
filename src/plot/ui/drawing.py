import pygame
import numpy as np
from src.plot.constants import TEXT_COLOR

def draw_signal_plot(screen, font, signal, x, y, w, h, show_time_markers=False, window_sec=None):
    t, v, meta = signal['t'], signal['v'], signal.get('meta', {})
    t = np.array(t)
    v = np.array(v)
    if meta is None:
        meta = {}
    # Special handling for signals with overlay flag or EDA signals:
# - Checks for phasic_norm and tonic_norm components
# - Displays both components as overlays when 'over' flag is True or signal ID starts with 'EDA'
# - This enables proper visualization of EDA components without duplicate displays    # Check signal ID from both metadata and signal data
    signal_id = signal.get('id', meta.get('id', 'unknown'))
    
    # Check for components in both metadata and signal data
    phasic_in_meta = 'phasic_norm' in meta and len(meta['phasic_norm']) > 0
    tonic_in_meta = 'tonic_norm' in meta and len(meta['tonic_norm']) > 0
    phasic_in_signal = 'phasic_norm' in signal and len(signal['phasic_norm']) > 0
    tonic_in_signal = 'tonic_norm' in signal and len(signal['tonic_norm']) > 0
    
    has_components = (phasic_in_meta and tonic_in_meta) or (phasic_in_signal and tonic_in_signal)
    
    # If components are in signal data but not metadata, use signal data
    use_signal_components = phasic_in_signal and tonic_in_signal
    
    # Check length match using appropriate component source
    if use_signal_components:
        components_match_size = len(t) == len(signal['phasic_norm']) == len(signal['tonic_norm'])
    else:
        components_match_size = has_components and len(t) == len(meta['phasic_norm']) == len(meta['tonic_norm'])
        
    # Check overlay mode from metadata or signal type
    is_overlay_mode = meta.get('over', False) or signal_id.upper().startswith('EDA')
    
    # Remove debug prints for production
    # print(f"[DEBUG] Drawing: id={signal_id}, has_components={has_components}, components_match={components_match_size}, overlay_mode={is_overlay_mode}")
    # print(f"[DEBUG] Meta keys: {list(meta.keys())}")
    # print(f"[DEBUG] Signal keys: {list(signal.keys())}")
    # print(f"[DEBUG] Components: meta=({phasic_in_meta}, {tonic_in_meta}), signal=({phasic_in_signal}, {tonic_in_signal})")
    
    if has_components:
        if use_signal_components:
            # print(f"[DEBUG] Component lengths (from signal): t={len(t)}, phasic={len(signal['phasic_norm'])}, tonic={len(signal['tonic_norm'])}")
            pass
        else:
            # print(f"[DEBUG] Component lengths (from meta): t={len(t)}, phasic={len(meta['phasic_norm'])}, tonic={len(meta['tonic_norm'])}")
            pass    # Use components for overlay if we have them and overlay mode is active
    if (has_components and components_match_size and is_overlay_mode):
        # Overlay both on same axes, using normalized values
        # Choose component source based on availability and extract components
        if use_signal_components:
            phasic = np.array(signal['phasic_norm'])
            tonic = np.array(signal['tonic_norm'])
        else:
            phasic = np.array(meta['phasic_norm'])
            tonic = np.array(meta['tonic_norm'])
        
        window_min = t[0] if len(t) > 0 else 0
        window_max = t[-1] if len(t) > 0 else 0
        if window_sec is not None and len(t) > 1:
            window_max = t[-1]
            window_min = window_max - window_sec
            indices = np.where((t >= window_min) & (t <= window_max))[0]
            t_overlay = t[indices]
            phasic = phasic[indices]
            tonic = tonic[indices]
        else:
            t_overlay = t
        def norm_to_y(val):
            return y + h - int(val * h)
        phasic_points = [(x + int((t_overlay[j] - window_min) / (window_max - window_min) * w), norm_to_y(phasic[j])) for j in range(len(t_overlay))] if len(t_overlay) > 1 and window_max > window_min else []
        tonic_points = [(x + int((t_overlay[j] - window_min) / (window_max - window_min) * w), norm_to_y(tonic[j])) for j in range(len(t_overlay))] if len(t_overlay) > 1 and window_max > window_min else []
        # Draw overlays: phasic (orange), tonic (green)
        if len(phasic_points) >= 2:
            pygame.draw.lines(screen, (255, 170, 0), False, phasic_points, 2)  # Orange
        if len(tonic_points) >= 2:
            pygame.draw.lines(screen, (0, 220, 0), False, tonic_points, 2)    # Green        # Draw peak markers on phasic only if available and enabled
        if (meta.get('show_peaks', False) and 'scr_peak_indices' in meta and 
            len(meta['scr_peak_indices']) > 0):
            peak_indices = np.array(meta['scr_peak_indices'])
            # Only show peaks that are in the current window
            for idx in peak_indices:
                if idx < 0 or idx >= len(t_overlay):
                    continue
                px = x + int((t_overlay[idx] - window_min) / (window_max - window_min) * w)
                py = norm_to_y(phasic[idx])
                pygame.draw.line(screen, (255, 255, 255), (px-6, py-6), (px+6, py+6), 2)
                pygame.draw.line(screen, (255, 255, 255), (px-6, py+6), (px+6, py-6), 2)
        # Draw label (signal id) at top left
        label = meta.get('name', signal.get('id', ''))
        label_surface = font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (x + 10, y + 10))        # Draw legend for overlays, with more vertical space below label
        legend_font = font
        legend_y = y + 10 + label_surface.get_height() + 18  # Add extra space
        legend_x = x + w - 120
        phasic_legend = legend_font.render("Phasic", True, (255, 170, 0))
        tonic_legend = legend_font.render("Tonic", True, (0, 220, 0))
        overlay_legend = legend_font.render("(Overlay)", True, (180, 180, 200))
        screen.blit(phasic_legend, (legend_x, legend_y))
        screen.blit(tonic_legend, (legend_x, legend_y + 18))
        # Add overlay indicator
        screen.blit(overlay_legend, (legend_x, legend_y + 36))
        return
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
