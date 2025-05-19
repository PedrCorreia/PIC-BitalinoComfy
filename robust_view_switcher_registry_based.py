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
    RAW_SIGNAL_COLOR, PROCESSED_SIGNAL_COLOR, TARGET_FPS
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

    print(f"[DEBUG] PlotRegistry singleton id at startup: {id(plot_registry)}")
    print(f"[DEBUG] Signal IDs at startup: {plot_registry.get_all_signal_ids()}")

    connector_node_id = "UI_CONNECTOR_NODE"
    selected_tab = ViewMode.RAW  # Use enum for selected_tab
    window_sec = 2
    running = True
    time.sleep(1.0)
    start_time = time.time()  # <-- Track start time for relative runtime

    from src.plot.ui.sidebar import Sidebar
    from src.plot.ui.status_bar import StatusBar
    from src.plot.view.signal_view import SignalView
    from src.plot.ui.settings import SettingsPanel

    # Instantiate UI components
    sidebar = Sidebar(screen, SIDEBAR_WIDTH, WINDOW_HEIGHT, font, icon_font, selected_tab, {})
    status_bar = StatusBar(screen, STATUS_BAR_HEIGHT, font, start_time)
    signal_view = SignalView(screen, None, {}, font)  # Data and lock will be set per draw
    settings_panel = SettingsPanel(font, plot_registry)

    # --- Precompute plot area variables (will be updated each frame) ---
    plot_x = SIDEBAR_WIDTH + PLOT_PADDING
    plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
    plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
    plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

    def draw_tab_content(screen, font, plot_registry, selected_tab, plot_x, plot_y, plot_width, plot_height, window_sec):
        from src.plot.ui.drawing import draw_signal_plot
        if selected_tab == ViewMode.SETTINGS:
            _ = settings_panel.draw(screen, plot_x, plot_y, plot_width, plot_height)
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

    while running:
        # --- Update plot area variables each frame in case of dynamic resizing (optional) ---
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

        # Determine FPS cap from settings panel
        fps_cap = TARGET_FPS if settings_panel.fps_cap_on else 0

        # Compute plotted_signals for event handling row count only
        plotted_signals = None
        if selected_tab == ViewMode.SETTINGS:
            raw_signals = plot_registry.get_signals_by_type(None, 'raw', debug=False)
            processed_signals = plot_registry.get_signals_by_type(None, 'processed', debug=False)
            plotted_signals = processed_signals + raw_signals

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                if selected_tab == ViewMode.SETTINGS:
                    # Use plotted_signals for row counting and event handling
                    table_y = plot_y + 60
                    row_height = 28
                    signals_list = plotted_signals if plotted_signals is not None else []
                    row = 1 + len(signals_list)
                    settings_panel.handle_event(event, plot_x, plot_y, row, row_height, plot_height, signals_list)
                # ...existing code for sidebar...
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if 0 <= mx < SIDEBAR_WIDTH:
                        clicked_mode = sidebar.handle_click(my)
                        if clicked_mode is not None:
                            sidebar.current_mode = clicked_mode
                            selected_tab = clicked_mode
        # --- Connect all signals to the connector node (emulated) ---
        for sid in plot_registry.get_all_signal_ids():
            plot_registry.connect_node_to_signal(connector_node_id, sid)
        # --- Draw UI ---
        print(f"[DEBUG] PlotRegistry singleton id in main loop: {id(plot_registry)}")
        print(f"[DEBUG] Signal IDs before UI draw: {plot_registry.get_all_signal_ids()}")
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        pygame.draw.rect(screen, (20,20,20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING))
        # --- Sidebar ---
        sidebar.current_mode = selected_tab
        # Pass has_signals to sidebar for blinking dot
        has_signals = len(plot_registry.get_all_signal_ids()) > 0
        sidebar._draw_status_dot_below_settings(has_signals=has_signals)
        sidebar.draw()
        # --- Plotting/View ---
        if selected_tab == ViewMode.SETTINGS:
            print(f"[DEBUG] PlotRegistry singleton id before settings_panel.draw: {id(plot_registry)}")
            print(f"[DEBUG] Signal IDs before settings_panel.draw: {plot_registry.get_all_signal_ids()}")
            # Let the settings panel query the registry for both raw and processed signals for the table
            settings_panel.draw(screen, plot_x, plot_y, plot_width, plot_height)
        else:
            draw_tab_content(screen, font, plot_registry, selected_tab, plot_x, plot_y, plot_width, plot_height, settings_panel.window_sec)
        # --- Status Bar ---
        rel_runtime = int(time.time() - start_time)
        status_bar.draw(clock.get_fps(), latency_monitor.get_current_latency() if hasattr(latency_monitor, 'get_current_latency') else 0.0, None)
        pygame.display.flip()
        if fps_cap > 0:
            clock.tick(fps_cap)
        else:
            clock.tick()
    generator.stop()

if __name__ == "__main__":
    main()
