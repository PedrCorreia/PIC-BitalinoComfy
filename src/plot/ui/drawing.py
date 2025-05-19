import pygame
import numpy as np
from src.plot.constants import TEXT_COLOR

def draw_signal_plot(screen, font, signal, x, y, w, h, show_time_markers=False, window_sec=None):
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
    if window_sec is not None and len(t) > 1:
        window_max = t[-1]
        window_min = window_max - window_sec
        indices = np.where((t >= window_min) & (t <= window_max))[0]
        if len(indices) < 2:
            t_plot = t[-2:]
            v_plot = v[-2:]
            window_min = t_plot[0] if len(t_plot) > 0 else 0
            window_max = t_plot[-1] if len(t_plot) > 0 else 0
        else:
            t_plot = t[indices]
            v_plot = v[indices]
    else:
        window_min = t[0] if len(t) > 0 else 0
        window_max = t[-1] if len(t) > 0 else 0
        t_plot = t
        v_plot = v
    points = [(x + int((t_plot[j] - window_min) / (window_max - window_min) * w), y + h - int((v_plot[j]-vmin)/(vmax-vmin)*h)) for j in range(len(t_plot))] if len(t_plot) > 1 and window_max > window_min else []
    # Color handling: support string color names as well as RGB tuples
    color = meta.get('color', (255,255,255))
    # DEBUG: show what color is being used
    print(f"[DRAW] meta color: {meta.get('color')}, resolved: {color}")
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
        except Exception as e:
            # print(f"[DRAW] Color conversion error: {e}")
            color = (255, 0, 255)  # fallback magenta for error
    if not (isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color)):
        color = (255, 0, 255)  # fallback magenta for error
    # DEBUG: show final color
    # print(f"[DRAW] final color: {color}")
    if len(points) >= 2:
        pygame.draw.lines(screen, color, False, points, 2)
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
