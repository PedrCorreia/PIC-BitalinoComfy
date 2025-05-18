#!/usr/bin/env python
"""
Registry-based Version: PIC-2025 Visualization Interface
- Real-time plotting of signals from the registry system (RAW, PROCESSED, TWIN, SETTINGS tabs)
- Uses background threads for signal generation and registry updates
- Emulates a connector node: connects signal IDs from generator/processed to the plot registry
- UI layout and logic matches robust_view_switcher_minimal_working.py, but all data comes from the registry
"""
import pygame
import numpy as np
import time
from src.plot.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, STATUS_BAR_HEIGHT,
    BACKGROUND_COLOR, SIDEBAR_COLOR, STATUS_COLOR, BUTTON_COLOR, BUTTON_COLOR_SETTINGS,
    ACCENT_COLOR, TEXT_COLOR, VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS,
    PLOT_PADDING, TAB_HEIGHT, CONTROL_PANEL_HEIGHT, TWIN_VIEW_SEPARATOR,
    BUTTON_MARGIN, SECTION_MARGIN, TITLE_PADDING, TEXT_MARGIN, CONTROL_MARGIN, ELEMENT_PADDING,
    RAW_SIGNAL_COLOR, PROCESSED_SIGNAL_COLOR
)
TABS = [VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS]
TAB_ICONS = ['R', 'P', 'T', 'S']

# --- Registry imports ---
from src.registry.plot_generator_debug_fixed import RegistrySignalGenerator
from src.registry.plot_registry import PlotRegistry

# --- Registry-based Signal Fetcher ---
def get_signals_by_type(plot_registry, window_sec, signal_type, debug=False):
    """
    Fetch signals from the registry by type ('raw' or 'processed'), returning a list of dicts:
    [{ 'id': ..., 't': ..., 'v': ..., 'meta': ... }, ...]
    """
    all_signal_ids = plot_registry.get_all_signal_ids()
    if debug:
        print(f"[DEBUG] All signal IDs in registry: {all_signal_ids}")
    if signal_type == 'raw':
        ids = [sid for sid in all_signal_ids if 'RAW' in sid or 'ECG' in sid or 'EDA' in sid]
    elif signal_type == 'processed':
        ids = [sid for sid in all_signal_ids if 'PROC' in sid or 'WAVE' in sid]
    else:
        ids = []
    signals = []
    for sid in ids:
        data = plot_registry.get_signal(sid)
        meta = plot_registry.get_signal_metadata(sid)
        if debug:
            print(f"[DEBUG] Fetching signal '{sid}': data type={type(data)}, meta={meta}")
        if data is None:
            continue
        # --- Handle numpy array of shape (N, 2) ---
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
            t, v = data[:, 0], data[:, 1]
        # If data is a deque/list/tuple of (timestamp, value) tuples, unpack it
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
            arr = np.array(data)
            t, v = arr[:, 0], arr[:, 1]
        # Support (timestamps, values) tuple
        elif isinstance(data, tuple) and len(data) == 2:
            t, v = np.array(data[0]), np.array(data[1])
        else:
            # If only values, synthesize timestamps based on sampling rate if available
            v = np.array(data)
            sr = meta.get('sampling_rate', 100) if meta else 100
            t = np.arange(len(v)) / sr
        # Always plot at least the last 2 points if available
        if len(t) < 2:
            if len(t) == 1:
                t = t - t[0]
                signals.append({'id': sid, 't': t, 'v': v, 'meta': meta})
            continue
        t0 = t[-1] - window_sec
        mask = t >= t0
        t, v = t[mask], v[mask]
        if len(t) < 2:
            t, v = t[-2:], v[-2:]
        t = t - t[0]
        signals.append({'id': sid, 't': t, 'v': v, 'meta': meta})
        if debug:
            print(f"[DEBUG] Signal '{sid}' windowed to {len(t)} points, t0={t[0] if len(t) else 'NA'}")
    return signals

# --- Modular Plot Drawing ---
def draw_signal_plot(screen, font, signal, x, y, w, h):
    t, v, meta = signal['t'], signal['v'], signal['meta']
    vmin, vmax = np.min(v), np.max(v)
    if vmax == vmin:
        vmax = vmin + 1
    points = [(x + int(j * w / len(t)), y + h - int((v[j]-vmin)/(vmax-vmin)*h)) for j in range(len(t))]
    if len(points) >= 2:
        pygame.draw.lines(screen, meta.get('color', (255,255,255)), False, points, 2)
    label = meta.get('name', signal['id'])
    label_surface = font.render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (x + 10, y + 10))

# --- Main App ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Registry Visualization")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    # --- Start registry-based signal generator ---
    generator = RegistrySignalGenerator()
    generator.set_buffer_seconds(20)
    generator.start()
    plot_registry = PlotRegistry.get_instance()

    connector_node_id = "UI_CONNECTOR_NODE"
    selected_tab = 0
    window_sec = 10
    running = True
    time.sleep(1.0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(len(TABS)):
                    tab_y = 50 + i * (TAB_HEIGHT + BUTTON_MARGIN)
                    if 0 <= mx < SIDEBAR_WIDTH and tab_y <= my < tab_y + TAB_HEIGHT:
                        selected_tab = i
        # --- Connect all signals to the connector node (emulated) ---
        for sid in plot_registry.get_all_signal_ids():
            plot_registry.connect_node_to_signal(connector_node_id, sid)
        # --- Draw UI ---
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3
        pygame.draw.rect(screen, (20,20,20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING))
        # --- Sidebar ---
        for i, label in enumerate(TABS):
            tab_y = 50 + i * (TAB_HEIGHT + BUTTON_MARGIN)
            tab_rect = pygame.Rect(0, tab_y, SIDEBAR_WIDTH, TAB_HEIGHT)
            color = ACCENT_COLOR if i == selected_tab else BUTTON_COLOR
            pygame.draw.rect(screen, color, tab_rect)
            icon = TAB_ICONS[i]
            icon_surface = font.render(icon, True, TEXT_COLOR)
            icon_x = 10 + (SIDEBAR_WIDTH - 20 - icon_surface.get_width()) // 2
            icon_y = tab_y + (TAB_HEIGHT - icon_surface.get_height()) // 2
            screen.blit(icon_surface, (icon_x, icon_y))
        # --- Plotting ---
        if selected_tab == 3:  # SETTINGS
            window_label = font.render(f"Rolling Window (s): {window_sec}", True, TEXT_COLOR)
            screen.blit(window_label, (plot_x + 50, plot_y + 30))
        elif selected_tab == 0:  # RAW
            raw_signals = get_signals_by_type(plot_registry, window_sec, 'raw', debug=True)
            print(f"[DEBUG] Plotting RAW signals: {[s['id'] for s in raw_signals]}")
            for i, sig in enumerate(raw_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, plot_width, plot_height)
        elif selected_tab == 1:  # PROCESSED
            processed_signals = get_signals_by_type(plot_registry, window_sec, 'processed', debug=True)
            print(f"[DEBUG] Plotting PROCESSED signals: {[s['id'] for s in processed_signals]}")
            for i, sig in enumerate(processed_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, plot_width, plot_height)
        elif selected_tab == 2:  # TWIN
            raw_signals = get_signals_by_type(plot_registry, window_sec, 'raw', debug=True)
            processed_signals = get_signals_by_type(plot_registry, window_sec, 'processed', debug=True)
            print(f"[DEBUG] Plotting TWIN processed: {[s['id'] for s in processed_signals]}, raw: {[s['id'] for s in raw_signals]}")
            left_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            right_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            center_x = plot_x + left_width + TWIN_VIEW_SEPARATOR
            pygame.draw.line(screen, (50, 50, 50), (center_x, plot_y), (center_x, WINDOW_HEIGHT - PLOT_PADDING), 3)
            for i, sig in enumerate(processed_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, left_width, plot_height)
            for i, sig in enumerate(raw_signals[:3]):
                draw_signal_plot(screen, font, sig, center_x + 5, plot_y + i*plot_height, right_width, plot_height)
        # --- Status Bar ---
        all_signal_ids = plot_registry.get_all_signal_ids()
        status = f"Mode: {TABS[selected_tab]} | FPS: {int(clock.get_fps())} | Runtime: {int(time.time())}s | Signals: {len(all_signal_ids)}"
        txt = font.render(status, True, TEXT_COLOR)
        screen.blit(txt, (10, 6))
        pygame.display.flip()
        clock.tick(60)
    generator.stop()

if __name__ == "__main__":
    main()
