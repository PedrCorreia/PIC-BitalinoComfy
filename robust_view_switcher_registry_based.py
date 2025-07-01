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
import sys
import os
# Ensure src/utils is in sys.path for dynamic imports
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'src', 'utils'))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

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
TABS = [ViewMode.RAW, ViewMode.PROCESSED, ViewMode.TWIN, ViewMode.METRICS, ViewMode.SETTINGS]
TAB_ICONS = ['R', 'P', 'T', 'M', 'S']

# --- Registry imports ---
from src.registry.signal_generator import RegistrySignalGenerator
from src.registry.plot_registry import PlotRegistry
from src.registry.signal_registry import SignalRegistry # Added import
from src.plot.performance.latency_monitor import LatencyMonitor

# --- Main App ---
def main(start_generators=True, stop_event=None):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Registry Visualization")
    font = pygame.font.SysFont("consolas", 16)
    icon_font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    # --- Start registry-based signal generator ---
    generator = None
    if start_generators:
        # User must manually add and start generators via RegistrySignalGenerator
        generator = RegistrySignalGenerator()
        # Optionally, you could add default generators here if desired
        # generator.add_generator(...)
        # generator.start_all()
    plot_registry = PlotRegistry.get_instance()
    latency_monitor = LatencyMonitor()

    connector_node_id = "UI_CONNECTOR_NODE"
    selected_tab = ViewMode.RAW  # Use enum for selected_tab
    # Use a dict for per-plot rolling window
    window_sec = {
        ViewMode.RAW: 5,
        ViewMode.PROCESSED: 5,
        ViewMode.TWIN: 5,
        ViewMode.METRICS: 30  # Added Metrics view window
    }
    running = True
    time.sleep(1.0)
    start_time = time.time()  # <-- Track start time for relative runtime

    from src.plot.ui.sidebar import Sidebar
    from src.plot.ui.status_bar import StatusBar
    from src.plot.view.signal_view import SignalView
    from src.plot.ui.settings import SettingsPanel
    from src.registry.signal_registry import SignalRegistry # Added import for SignalRegistry

    # Instantiate UI components
    sidebar = Sidebar(screen, SIDEBAR_WIDTH, WINDOW_HEIGHT, font, icon_font, selected_tab, {}, tabs=TABS, tab_icons=TAB_ICONS)
    status_bar = StatusBar(screen, STATUS_BAR_HEIGHT, font, start_time)
    settings_panel = SettingsPanel(font, plot_registry)

    # --- Precompute plot area variables (will be updated each frame) ---
    plot_x = SIDEBAR_WIDTH + PLOT_PADDING
    plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
    plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
    plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

    # --- Per-signal rolling window state ---
    signal_window_sec = {}  # {signal_id: window_sec}


    def draw_tab_content(screen, font, plot_registry, selected_tab, plot_x, plot_y, plot_width, plot_height, window_sec_dict, signal_window_sec):
        from src.plot.ui.drawing import draw_signal_plot
        def get_sig_id(sig):
            if isinstance(sig, dict):
                # Try 'id', then 'name', then fallback to string representation of the dict
                return sig.get('id', sig.get('name', str(sig)))
            if hasattr(sig, 'id'):
                return sig.id
            if hasattr(sig, 'name'):
                return sig.name
            return str(sig)
            
        def determine_plot_mode(sig):
            # Check if signal metadata indicates it's EDA
            if isinstance(sig, dict) and 'meta' in sig:
                meta = sig.get('meta', {})
                if isinstance(meta, dict): # Ensure meta is a dict before .get
                    viz_subtype = meta.get('viz_subtype')
                    # print(f"[DETERMINE_PLOT_MODE_DEBUG] Signal {get_sig_id(sig)} has meta with viz_subtype: {viz_subtype}", flush=True) # DEBUG PRINT
                    if viz_subtype == 'eda':
                        # print(f"[DETERMINE_PLOT_MODE_DEBUG] Determined mode: 'eda' for {get_sig_id(sig)} based on viz_subtype", flush=True) # DEBUG PRINT
                        return 'eda'
                    if viz_subtype == 'respiration':
                        return 'respiration'
            # Fallback for older EDA structure or other direct EDA signals (less likely now)
            if isinstance(sig, dict) and 'scl' in sig and 'scr' in sig and 't' in sig: # Check for specific EDA keys
                # print(f"[DETERMINE_PLOT_MODE_DEBUG] Determined mode: 'eda' for {get_sig_id(sig)} based on scl/scr keys", flush=True) # DEBUG PRINT
                return 'eda'
            # Check if signal metadata indicates it's EDA (older check, keep for compatibility or other types)
            if isinstance(sig, dict) and 'meta' in sig and sig.get('meta', {}).get('type') == 'eda': # Check meta type 'eda'
                # print(f"[DETERMINE_PLOT_MODE_DEBUG] Determined mode: 'eda' for {get_sig_id(sig)} based on meta.type == eda", flush=True) # DEBUG PRINT
                return 'eda'
            # print(f"[DETERMINE_PLOT_MODE_DEBUG] Determined mode: 'default' for {get_sig_id(sig)}", flush=True) # DEBUG PRINT
            return 'default'
            
        if selected_tab == ViewMode.SETTINGS:
            _ = settings_panel.draw(screen, plot_x, plot_y, plot_width, plot_height, window_sec_dict, selected_tab, signal_window_sec=signal_window_sec)
        elif selected_tab == ViewMode.RAW:
            raw_signals = plot_registry.get_signals_by_type(None, 'raw', debug=True)
            for i, sig in enumerate(raw_signals[:3]):
                current_sig_id = get_sig_id(sig)
                current_window = signal_window_sec.get(current_sig_id, window_sec_dict.get(ViewMode.RAW, 5))
                plot_mode = determine_plot_mode(sig)
                # print(f"[RAW_VIEW_DEBUG] Plotting signal: {current_sig_id}, Mode: {plot_mode}, Window: {current_window}", flush=True) # DEBUG PRINT
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i * plot_height, plot_width, plot_height, True, current_window, mode=plot_mode)
        elif selected_tab == ViewMode.PROCESSED:
            processed_signals = plot_registry.get_signals_by_type(None, 'processed', debug=True)
            # print(f"[VIEW_DEBUG_PROCESSED_LIST] Fetched processed signals: Count={len(processed_signals)}, Content={processed_signals}", flush=True) # DEBUG PRINT ADDED

            for i, sig in enumerate(processed_signals[:3]):
                current_sig_id = get_sig_id(sig)
                current_window = signal_window_sec.get(current_sig_id, window_sec_dict.get(ViewMode.PROCESSED, 5))
                plot_mode = determine_plot_mode(sig)
                # print(f"[PROCESSED_VIEW_DEBUG] Plotting signal: {current_sig_id}, Mode: {plot_mode}, Window: {current_window}", flush=True) # DEBUG PRINT
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i * plot_height, plot_width, plot_height, True, current_window, mode=plot_mode)
        elif selected_tab == ViewMode.TWIN:
            raw_signals = plot_registry.get_signals_by_type(None, 'raw', debug=True)
            processed_signals = plot_registry.get_signals_by_type(None, 'processed', debug=True)
            left_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            right_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            center_x_coord = plot_x + left_width + TWIN_VIEW_SEPARATOR # Renamed to avoid conflict
            pygame.draw.line(screen, (50, 50, 50), (center_x_coord, plot_y), (center_x_coord, plot_y + 3*plot_height), 3) # Use renamed var
            for i, sig in enumerate(processed_signals[:3]): # Plot processed on the left
                current_sig_id = get_sig_id(sig)
                current_window = signal_window_sec.get(current_sig_id, window_sec_dict.get(ViewMode.TWIN, 5))
                plot_mode = determine_plot_mode(sig)
                # print(f"[TWIN_VIEW_PROCESSED_DEBUG] Plotting signal: {current_sig_id}, Mode: {plot_mode}, Window: {current_window}", flush=True) # DEBUG PRINT
                draw_signal_plot(screen, font, sig, plot_x, plot_y + i * plot_height, left_width, plot_height, True, current_window, mode=plot_mode)
            for i, sig in enumerate(raw_signals[:3]): # Plot raw on the right
                current_sig_id = get_sig_id(sig)
                current_window = signal_window_sec.get(current_sig_id, window_sec_dict.get(ViewMode.TWIN, 5))
                plot_mode = determine_plot_mode(sig) # Raw signals are usually 'default'
                # print(f"[TWIN_VIEW_RAW_DEBUG] Plotting signal: {current_sig_id}, Mode: {plot_mode}, Window: {current_window}", flush=True) # DEBUG PRINT
                draw_signal_plot(screen, font, sig, center_x_coord + TWIN_VIEW_SEPARATOR, plot_y + i * plot_height, right_width, plot_height, True, current_window, mode=plot_mode)
        elif selected_tab == ViewMode.METRICS: # Added Metrics view logic
            from src.plot.view.metrics_view import MetricsView
            # Ensure the correct registry (SignalRegistry for metrics) is passed to MetricsView
            metrics_registry = SignalRegistry.get_instance() 
            # ADDED DEBUG: Check what signals the main UI loop sees in SignalRegistry
            all_sids_in_main_loop = metrics_registry.get_all_signal_ids()
            #print(f"[MainLoop DEBUG] SignalRegistry contents before MetricsView.draw: {len(all_sids_in_main_loop)} signals: {all_sids_in_main_loop}", flush=True)
            metrics_view = MetricsView(font, metrics_registry) # Pass metrics_registry
            # Use the full available height for metrics view
            metrics_plot_height = WINDOW_HEIGHT - STATUS_BAR_HEIGHT - (2 * PLOT_PADDING) # Adjusted height
            current_metrics_window = window_sec_dict.get(ViewMode.METRICS, 30) # Use specific or default window
            # print(f"[METRICS_VIEW_DEBUG] Drawing MetricsView with window: {current_metrics_window}", flush=True) # DEBUG PRINT
            metrics_view.draw(screen, plot_x, plot_y, plot_width, metrics_plot_height, window_sec=current_metrics_window)
    # ...existing code...

    while running:
        # --- Update plot area variables each frame in case of dynamic resizing (optional) ---
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3

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
                    row_height = 28
                    signals_list = plotted_signals if plotted_signals is not None else []
                    row = 1 + len(signals_list)
                    settings_panel.handle_event(event, plot_x, plot_y, row, row_height, plot_height, window_sec, selected_tab, signals_list, signal_window_sec=signal_window_sec)
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
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3
        pygame.draw.rect(screen, (20,20,20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING))
        # --- Sidebar ---
        sidebar.current_mode = selected_tab
        sidebar.draw()
        # --- Plotting/View ---
        if selected_tab == ViewMode.SETTINGS:
            # Draw settings panel and capture any window_sec changes
            settings_panel.draw(screen, plot_x, plot_y, plot_width, plot_height, window_sec, selected_tab, plotted_signals, signal_window_sec=signal_window_sec)
            # Sync window_sec with per-view window times from signal_window_sec if present
            # (Assume settings panel updates window_sec in-place, but if not, add logic here)
        else:
            draw_tab_content(screen, font, plot_registry, selected_tab, plot_x, plot_y, plot_width, plot_height, window_sec, signal_window_sec)
        # --- Status Bar ---
        status_bar.draw(clock.get_fps(), latency_monitor.get_current_latency() if hasattr(latency_monitor, 'get_current_latency') else 0.0, None)
        pygame.display.flip()
        clock.tick(TARGET_FPS)

if __name__ == "__main__":
    main()
