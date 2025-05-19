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
from src.plot.view_mode import ViewMode
# Use ViewMode enums for TABS
TABS = [ViewMode.RAW, ViewMode.PROCESSED, ViewMode.TWIN, ViewMode.SETTINGS]
TAB_ICONS = ['R', 'P', 'T', 'S']

# --- Registry imports ---
from src.registry.signal_generator import RegistrySignalGenerator
from src.registry.plot_registry import PlotRegistry
from src.plot.performance.latency_monitor import LatencyMonitor

# --- Main App ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Registry Visualization")
    font = pygame.font.SysFont("consolas", 16)
    icon_font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    # --- Start registry-based signal generator ---
    generator = RegistrySignalGenerator()
    generator.set_buffer_seconds(120)  # Make generator buffer longer than the view buffer
    generator.start()
    plot_registry = PlotRegistry.get_instance()
    latency_monitor = LatencyMonitor()

    connector_node_id = "UI_CONNECTOR_NODE"
    selected_tab = ViewMode.RAW  # Use enum for selected_tab
    window_sec = 2
    running = True
    time.sleep(1.0)
    start_time = time.time()  # <-- Track start time for relative runtime

    from src.plot.ui.sidebar import Sidebar
    from src.plot.ui.status_bar import StatusBar
    from src.plot.view.signal_view import SignalView

    # Instantiate UI components
    sidebar = Sidebar(screen, SIDEBAR_WIDTH, WINDOW_HEIGHT, font, icon_font, selected_tab, {})
    status_bar = StatusBar(screen, STATUS_BAR_HEIGHT, font, start_time)
    signal_view = SignalView(screen, None, {}, font)  # Data and lock will be set per draw

    # --- Precompute plot area variables (will be updated each frame) ---
    plot_x = SIDEBAR_WIDTH + PLOT_PADDING
    plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
    plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
    plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

    while running:
        # --- Update plot area variables each frame in case of dynamic resizing (optional) ---
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                # Use sidebar's handle_click to determine tab
                if 0 <= mx < SIDEBAR_WIDTH:
                    clicked_mode = sidebar.handle_click(my)
                    if clicked_mode is not None:
                        sidebar.current_mode = clicked_mode
                        selected_tab = clicked_mode
                # SETTINGS: +/- buttons for window_sec
                if selected_tab == ViewMode.SETTINGS:
                    plus_rect = pygame.Rect(plot_x + 220, plot_y + 25, 30, 30)
                    minus_rect = pygame.Rect(plot_x + 50, plot_y + 25, 30, 30)
                    if plus_rect.collidepoint(mx, my):
                        window_sec = min(window_sec + 1, 60)
                    elif minus_rect.collidepoint(mx, my):
                        window_sec = max(window_sec - 1, 1)
            elif event.type == pygame.KEYDOWN:
                # SETTINGS: left/right arrow keys for window_sec
                if selected_tab == ViewMode.SETTINGS:
                    if event.key == pygame.K_LEFT:
                        window_sec = max(window_sec - 1, 1)
                    elif event.key == pygame.K_RIGHT:
                        window_sec = min(window_sec + 1, 60)
        # --- Connect all signals to the connector node (emulated) ---
        for sid in plot_registry.get_all_signal_ids():
            plot_registry.connect_node_to_signal(connector_node_id, sid)
        # --- Draw UI ---
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        # plot_x, plot_y, plot_width, plot_height already set above
        pygame.draw.rect(screen, (20,20,20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING))
        # --- Sidebar ---
        sidebar.current_mode = selected_tab
        sidebar.draw()
        # --- Plotting/View ---
        # For now, use the procedural draw_signal_views as a placeholder, but ideally instantiate and use view classes for each tab
        # signal_view.draw()  # Would need to be adapted for each tab/view type
        from src.plot.ui.drawing import draw_signal_plot
        if selected_tab == ViewMode.SETTINGS:
            window_label = font.render(f"Rolling Window (s): {window_sec}", True, TEXT_COLOR)
            screen.blit(window_label, (plot_x + 50, plot_y + 30))
            plus_rect = pygame.Rect(plot_x + 220, plot_y + 25, 30, 30)
            minus_rect = pygame.Rect(plot_x + 50, plot_y + 25, 30, 30)
            pygame.draw.rect(screen, BUTTON_COLOR_SETTINGS, plus_rect)
            pygame.draw.rect(screen, BUTTON_COLOR_SETTINGS, minus_rect)
            plus_label = font.render("+", True, (0,0,0))
            minus_label = font.render("-", True, (0,0,0))
            screen.blit(plus_label, (plus_rect.x + 8, plus_rect.y + 2))
            screen.blit(minus_label, (minus_rect.x + 8, minus_rect.y + 2))
        elif selected_tab == ViewMode.RAW:
            raw_signals = plot_registry.get_signals_by_type(None, 'raw', debug=True)
            for i, sig in enumerate(raw_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, plot_width, plot_height, show_time_markers=True, window_sec=window_sec)
        elif selected_tab == ViewMode.PROCESSED:
            processed_signals = plot_registry.get_signals_by_type(None, 'processed', debug=True)
            for i, sig in enumerate(processed_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, plot_width, plot_height, show_time_markers=True, window_sec=window_sec)
        elif selected_tab == ViewMode.TWIN:
            raw_signals = plot_registry.get_signals_by_type(None, 'raw', debug=True)
            processed_signals = plot_registry.get_signals_by_type(None, 'processed', debug=True)
            left_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            right_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            center_x = plot_x + left_width + TWIN_VIEW_SEPARATOR
            pygame.draw.line(screen, (50, 50, 50), (center_x, plot_y), (center_x, plot_y + 3*plot_height), 3)
            for i, sig in enumerate(processed_signals[:3]):
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i*plot_height, left_width, plot_height, show_time_markers=True, window_sec=window_sec)
            for i, sig in enumerate(raw_signals[:3]):
                draw_signal_plot(screen, font, sig, center_x + 5, plot_y + i*plot_height, right_width, plot_height, show_time_markers=True, window_sec=window_sec)
        # --- Status Bar ---
        rel_runtime = int(time.time() - start_time)
        status_bar.draw(clock.get_fps(), latency_monitor.get_current_latency() if hasattr(latency_monitor, 'get_current_latency') else 0.0, None)
        pygame.display.flip()
        clock.tick(60)
    generator.stop()

if __name__ == "__main__":
    main()
