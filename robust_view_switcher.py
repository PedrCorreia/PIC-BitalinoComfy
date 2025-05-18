#!/usr/bin/env python
"""
PIC-2025 Visualization Interface

A lightweight UI for visualizing PIC-2025 signals with tabs for different view modes.
Features:
- Status bar at the top
- Sidebar with tabs (RAW, PROCESSED, TWIN, SETTINGS)
- Signal connection indicators
"""

import os
import sys
import time
import threading
import logging
import pygame
import numpy as np
from enum import Enum

# Add project root to path to ensure imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PIC-2025-UI')

# --- Local fallback definitions ---
# Local ViewMode definition (will be overridden if import succeeds)
class LocalViewMode(Enum):
    RAW = 0
    PROCESSED = 1
    TWIN = 2
    STACKED = 3
    SETTINGS = 4

# Default fallback constants (will be overridden if import succeeds)
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_SIDEBAR_WIDTH = 80
DEFAULT_STATUS_BAR_HEIGHT = 32
DEFAULT_BG_COLOR = (30, 30, 30)
DEFAULT_SIDEBAR_COLOR = (40, 40, 60)
DEFAULT_STATUS_COLOR = (50, 50, 80)
DEFAULT_DOT_ON = (0, 255, 0)
DEFAULT_FONT_COLOR = (220, 220, 220)
DEFAULT_TAB_COLOR = (80, 80, 120)
DEFAULT_TAB_SELECTED = (120, 120, 180)
DEFAULT_DOT_OFF = (80, 80, 80)

# --- Import modules ---
# Try to import ViewMode from the project modules
try:
    from src.plot.view_mode import ViewMode
    logger.info("Imported ViewMode from src.plot.view_mode")
    use_external_view_mode = True
except ImportError:
    try:
        from plot.view_mode import ViewMode
        logger.info("Imported ViewMode from plot.view_mode")
        use_external_view_mode = True
    except ImportError:
        ViewMode = LocalViewMode
        logger.warning("Could not import ViewMode, using local definition")
        use_external_view_mode = False

# Try to import constants from the project
try:
    from src.plot.constants import (
        WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, STATUS_BAR_HEIGHT, 
        BACKGROUND_COLOR, SIDEBAR_COLOR, TEXT_COLOR, OK_COLOR,
        STATUS_COLOR, VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS
    )
    # Map constants to our naming convention
    BG_COLOR = BACKGROUND_COLOR
    FONT_COLOR = TEXT_COLOR
    DOT_ON = OK_COLOR
    logger.info("Imported constants from src.plot.constants")
    constants_imported = True
    
    # Also try to import Sidebar class
    try:
        from src.plot.ui.sidebar import Sidebar
        logger.info("Imported Sidebar from src.plot.ui.sidebar")
        sidebar_imported = True
    except ImportError:
        logger.warning("Could not import Sidebar from src.plot.ui.sidebar")
        sidebar_imported = False
        
except ImportError:    
    try:
        from plot.constants import (
            WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, STATUS_BAR_HEIGHT,
            BACKGROUND_COLOR, SIDEBAR_COLOR, TEXT_COLOR, OK_COLOR,
            STATUS_COLOR, VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS
        )
        # Map constants to our naming convention
        BG_COLOR = BACKGROUND_COLOR
        FONT_COLOR = TEXT_COLOR
        DOT_ON = OK_COLOR
        logger.info("Imported constants from plot.constants")
        constants_imported = True
        
        # Also try to import Sidebar class
        try:
            from plot.ui.sidebar import Sidebar
            logger.info("Imported Sidebar from plot.ui.sidebar")
            sidebar_imported = True
        except ImportError:
            logger.warning("Could not import Sidebar from plot.ui.sidebar")
            sidebar_imported = False
            
    except ImportError:
        # Use fallback constants
        logger.warning("Could not import constants, using fallback values")
        WINDOW_WIDTH = DEFAULT_WINDOW_WIDTH
        WINDOW_HEIGHT = DEFAULT_WINDOW_HEIGHT
        SIDEBAR_WIDTH = DEFAULT_SIDEBAR_WIDTH
        STATUS_BAR_HEIGHT = DEFAULT_STATUS_BAR_HEIGHT
        BG_COLOR = DEFAULT_BG_COLOR
        SIDEBAR_COLOR = DEFAULT_SIDEBAR_COLOR
        STATUS_COLOR = DEFAULT_STATUS_COLOR
        DOT_ON = DEFAULT_DOT_ON
        FONT_COLOR = DEFAULT_FONT_COLOR
        VIEW_MODE_RAW = "RAW"
        VIEW_MODE_PROCESSED = "PROCESSED"
        VIEW_MODE_TWIN = "TWIN"
        VIEW_MODE_SETTINGS = "SETTINGS"
        constants_imported = False
        sidebar_imported = False

# Try to import PlotUnit class
try:
    from src.plot.plot_unit import PlotUnit
    logger.info("Imported PlotUnit from src.plot.plot_unit")
except ImportError:
    try:
        from plot.plot_unit import PlotUnit
        logger.info("Imported PlotUnit from plot.plot_unit")
    except ImportError:
        logger.error("Could not import PlotUnit")
        sys.exit(1)

# Try to import PlotRegistry
try:
    from src.registry.plot_registry import PlotRegistry
    logger.info("Imported PlotRegistry from src.registry.plot_registry")
except ImportError:
    try:
        from registry.plot_registry import PlotRegistry
        logger.info("Imported PlotRegistry from registry.plot_registry")
    except ImportError:
        logger.error("Could not import PlotRegistry")
        sys.exit(1)

# Additional UI constants that might not be in constants.py
DOT_OFF = DEFAULT_DOT_OFF
TAB_COLOR = DEFAULT_TAB_COLOR
TAB_SELECTED = DEFAULT_TAB_SELECTED

# Check if we should use the Sidebar component
use_sidebar_component = 'sidebar_imported' in locals() and sidebar_imported

# Define the 4 tabs
TABS = [
    (VIEW_MODE_RAW, ViewMode.RAW),
    (VIEW_MODE_PROCESSED, ViewMode.PROCESSED),
    (VIEW_MODE_TWIN, ViewMode.TWIN),
    (VIEW_MODE_SETTINGS, ViewMode.SETTINGS),
]

# --- Helper Functions ---
def get_registry_instance():
    """Get the PlotRegistry instance."""
    try:
        registry = PlotRegistry.get_instance()
        return registry
    except Exception as e:
        logger.error(f"Error getting registry: {e}")
        return None

def get_registry_signals():
    """Get signals from the registry with error handling."""
    registry = get_registry_instance()
    if not registry or not hasattr(registry, 'signals'):
        return {}
    
    try:
        return registry.signals.copy()
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return {}

def generate_static_plots():
    """Generate static plots for each view mode with appropriate visualization characteristics."""
    plots = {}
    sample_count = 1000
    t = np.linspace(0, 10, sample_count)
    
    # Raw signals with different waveforms and noise
    # Sine wave raw
    plots['SINE_RAW'] = 0.9 * np.sin(2 * np.pi * 0.5 * t)
    plots['SINE_RAW'] += np.random.normal(0, 0.08, size=sample_count)  # Add more noise
    
    # Square wave raw
    plots['SQUARE_RAW'] = 0.8 * np.sign(np.sin(2 * np.pi * 0.3 * t))
    plots['SQUARE_RAW'] += np.random.normal(0, 0.1, size=sample_count)  # Add noise
    
    # Sawtooth wave raw
    plots['SAWTOOTH_RAW'] = 0.7 * (2 * (t * 0.4 % 1) - 1)
    plots['SAWTOOTH_RAW'] += np.random.normal(0, 0.07, size=sample_count)  # Add noise
    
    # Processed signals - same waves but without noise and slight phase shifts
    plots['SINE_PROCESSED'] = 0.9 * np.sin(2 * np.pi * 0.5 * t + 0.2)
    plots['SQUARE_PROCESSED'] = 0.8 * np.sign(np.sin(2 * np.pi * 0.3 * t + 0.1))
    plots['SAWTOOTH_PROCESSED'] = 0.7 * (2 * ((t + 0.1) * 0.4 % 1) - 1)
    
    return plots

def render_static_plot(surface, data, x, y, width, height, color=(255, 255, 255)):
    """Render a static plot on the surface."""
    if len(data) < 2:
        return
    
    # Scale data to fit in the plot area
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        data_max = data_min + 1  # Avoid division by zero
    
    # Calculate points
    points = []
    for i in range(min(len(data), width)):
        # Scale x coordinate from data index to plot width
        px = x + i * (width / min(len(data), width))
        # Scale y coordinate from data value to plot height
        py = y + height - ((data[int(i * len(data) / width)] - data_min) / 
                           (data_max - data_min)) * height
        points.append((int(px), int(py)))
    
    # Draw the line
    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, 2)

def register_demo_signals():
    """Create demo signals for testing."""
    registry = get_registry_instance()
    if not registry:
        return False
    
    try:
        # Check if we already have signals
        if hasattr(registry, 'signals') and registry.signals:
            logger.info("Registry already has signals")
            return True
            
        logger.info("Creating demo signals")
        duration = 10.0
        sample_rate = 100
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create some basic signals
        registry.register_signal('ECG_RAW', 0.8 * np.sin(2 * np.pi * 0.5 * t), {
            'name': 'ECG Raw', 
            'color': (255, 0, 0),
            'type': 'raw'
        })
        
        registry.register_signal('RESP_RAW', 0.7 * np.sin(0.3 * np.pi * t), {
            'name': 'Respiration Raw', 
            'color': (0, 0, 255),
            'type': 'raw'
        })
        
        # Create processed versions
        registry.register_signal('PROC_ECG', 0.8 * np.sin(2 * np.pi * 0.5 * t + 0.2), {
            'name': 'ECG Processed', 
            'color': (200, 100, 100),
            'type': 'processed'
        })
        
        registry.register_signal('PROC_RESP', 0.7 * np.sin(0.3 * np.pi * t + 0.1), {
            'name': 'Resp Processed', 
            'color': (100, 100, 200),
            'type': 'processed'
        })
        
        logger.info("Demo signals created")
        return True
    except Exception as e:
        logger.error(f"Error creating signals: {e}")
        return False

# --- Main UI App ---
def main():
    """Main UI application for PIC-2025 visualization."""
    print("\n=== PIC-2025 Visualization Interface ===\n")
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Visualization")
    
    # Try to import font settings from constants
    try:
        from src.plot.constants import DEFAULT_FONT, DEFAULT_FONT_SIZE
        font = pygame.font.SysFont(DEFAULT_FONT, DEFAULT_FONT_SIZE)
    except ImportError:
        try:
            from plot.constants import DEFAULT_FONT, DEFAULT_FONT_SIZE
            font = pygame.font.SysFont(DEFAULT_FONT, DEFAULT_FONT_SIZE)
        except ImportError:
            # Fallback if constants not available
            font = pygame.font.SysFont("consolas", 16)
        
    clock = pygame.time.Clock()

    # Initialize PlotUnit
    try:
        # Get the singleton instance
        plot_unit = PlotUnit.get_instance()
        
        # Configure dimensions
        if hasattr(plot_unit, 'width'):
            plot_unit.width = WINDOW_WIDTH
        if hasattr(plot_unit, 'height'):
            plot_unit.height = WINDOW_HEIGHT
        if hasattr(plot_unit, 'sidebar_width'):
            plot_unit.sidebar_width = SIDEBAR_WIDTH
        if hasattr(plot_unit, 'status_bar_height'):
            plot_unit.status_bar_height = STATUS_BAR_HEIGHT
            
        # Re-calculate plot_width if available
        if hasattr(plot_unit, 'plot_width'):
            plot_unit.plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH
        
        # Set running flag to True
        if hasattr(plot_unit, 'running'):
            plot_unit.running = True
        
        # Create data lock if needed
        if not hasattr(plot_unit, 'data_lock'):
            plot_unit.data_lock = threading.Lock()
            
        # Create empty data dictionary if needed
        if not hasattr(plot_unit, 'data'):
            plot_unit.data = {}
        
        # Start visualization thread
        if hasattr(plot_unit, 'start') and callable(plot_unit.start):
            if not getattr(plot_unit, 'initialized', False):
                plot_unit.start()
                logger.info("PlotUnit visualization started")
            else:
                logger.info("PlotUnit already initialized")
        else:
            logger.error("PlotUnit has no start method")
            return
    except Exception as e:
        logger.error(f"Error initializing PlotUnit: {e}")
        return
    
    # Create demo signals
    register_demo_signals()
    
    # Generate static plots for visualization
    static_plots = generate_static_plots()
    
    # UI state variables
    selected_tab = 0  # Start with RAW view
    running = True
    signal_thread = None
    stop_signal_thread = threading.Event()
    connection_status = [False] * len(TABS)
    last_fps = 0
    signal_count = 0
    current_view_mode = ViewMode.RAW
    
    # Create settings dictionary
    settings = {
        'caps_enabled': True,          # Enable FPS cap
        'light_mode': False,           # Dark mode by default
        'performance_mode': False,     # Quality mode by default
        'connected_nodes': 0,          # Updated dynamically
    }
    
    # Initialize sidebar if component is available
    sidebar_instance = None
    if use_sidebar_component:
        # Try to create an icon font
        try:
            icon_font = pygame.font.SysFont(None, 28)  # Default icon font
        except:
            icon_font = font
        
        # Create the sidebar instance
        try:
            sidebar_instance = Sidebar(screen, SIDEBAR_WIDTH, WINDOW_HEIGHT, font, icon_font, current_view_mode, settings)
            logger.info("Initialized Sidebar component")
        except Exception as e:
            logger.error(f"Error initializing Sidebar: {e}")
            sidebar_instance = None
    def signal_updater(tab_idx):
        """Background thread to update signals for the current view."""
        while not stop_signal_thread.is_set():
            try:
                signals = get_registry_signals()
                
                with plot_unit.data_lock:
                    plot_unit.data.clear()
                    
                    # Get the ViewMode for this tab
                    view_mode = TABS[tab_idx][1]
                    
                    # If there are no signals from the registry, use our static plots
                    if not signals:
                        signals = static_plots
                    
                    # RAW view - show only raw signals
                    if view_mode == ViewMode.RAW:
                        for k, v in signals.items():
                            if 'RAW' in k:
                                plot_unit.data[k] = np.copy(v)
                    
                    # PROCESSED view - show only processed signals
                    elif view_mode == ViewMode.PROCESSED:
                        for k, v in signals.items():
                            if 'PROCESSED' in k:
                                plot_unit.data[k] = np.copy(v)
                    
                    # TWIN view - all signals (raw and processed will be separated automatically)
                    elif view_mode == ViewMode.TWIN:
                        for k, v in signals.items():
                            if 'RAW' in k or 'PROCESSED' in k:
                                plot_unit.data[k] = np.copy(v)
                    
                    # SETTINGS view - don't add any signals, as we'll show buttons instead
                    elif view_mode == ViewMode.SETTINGS:
                        # Just add a minimal placeholder if needed
                        plot_unit.data.clear()  # Clear all signals for Settings view
                
                connection_status[tab_idx] = True
                time.sleep(0.5)  # Update twice per second
            except Exception as e:
                logger.error(f"Error in signal updater: {e}")
                connection_status[tab_idx] = False
                time.sleep(1.0)  # Longer delay on error
    
    def start_signal_thread(tab_idx):
        """Start the background signal update thread."""
        nonlocal signal_thread, stop_signal_thread, current_view_mode
        
        # Clean up existing thread if any
        stop_signal_thread.set()
        if signal_thread and signal_thread.is_alive():
            signal_thread.join(timeout=1.0)
        
        # Get target mode
        target_mode = TABS[tab_idx][1]
        
        # Set the view mode
        try:
            # Since we're now importing ViewMode directly from the module,
            # we don't need to do conversion between different enum types
            if hasattr(plot_unit, 'set_mode'):
                plot_unit.set_mode(target_mode)
                logger.info(f"Set view mode to {target_mode.name}")
            elif hasattr(plot_unit, '_set_mode'):
                plot_unit._set_mode(target_mode)
                logger.info(f"Set view mode to {target_mode.name}")
            else:
                # Fall back to setting attributes directly
                if hasattr(plot_unit, 'current_mode'):
                    plot_unit.current_mode = target_mode
                
                # Update sidebar if available
                if hasattr(plot_unit, 'sidebar') and plot_unit.sidebar:
                    plot_unit.sidebar.current_mode = target_mode
                
                # Update event handler if available
                if hasattr(plot_unit, 'event_handler') and plot_unit.event_handler:
                    plot_unit.event_handler.current_mode = target_mode
        except Exception as e:
            logger.error(f"Error setting view mode: {e}")
        
        # Update current mode
        current_view_mode = target_mode
        
        # Start new signal thread
        stop_signal_thread = threading.Event()
        signal_thread = threading.Thread(target=signal_updater, args=(tab_idx,), daemon=True)
        signal_thread.start()
        
        logger.info(f"Changed view mode to {TABS[tab_idx][0]}")

    def stop_signal_updates():
        """Safely stop the signal update thread."""
        nonlocal stop_signal_thread, signal_thread
        
        stop_signal_thread.set()
        if signal_thread and signal_thread.is_alive():
            signal_thread.join(timeout=1.0)
        
        # Clear all signals
        try:
            with plot_unit.data_lock:
                plot_unit.data.clear()
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
        
        # Reset connection status
        for i in range(len(connection_status)):
            connection_status[i] = False    # Start with RAW view
    start_signal_thread(selected_tab)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                if sidebar_instance:
                    # Use Sidebar component's click handling
                    if mx < SIDEBAR_WIDTH:
                        clicked_tab = sidebar_instance.handle_click(my)
                        if clicked_tab is not None and clicked_tab != selected_tab:
                            # Switch tabs
                            logger.info(f"User clicked on sidebar tab {clicked_tab}")
                            stop_signal_updates()
                            selected_tab = clicked_tab
                            start_signal_thread(selected_tab)
                else:
                    # Fallback manual click handling
                    # Check for tab clicks in sidebar
                    tab_height = 60
                    for i, (label, _) in enumerate(TABS):
                        tab_y = 50 + i*tab_height
                        if 0 <= mx < SIDEBAR_WIDTH and tab_y <= my < tab_y + tab_height:
                            if selected_tab != i:
                                # Switch tabs
                                logger.info(f"User clicked on {label} tab")
                                stop_signal_updates()
                                selected_tab = i
                                start_signal_thread(selected_tab)
        
        # Update metrics
        if hasattr(plot_unit, 'fps_counter') and hasattr(plot_unit.fps_counter, 'get_fps'):
            try:
                last_fps = plot_unit.fps_counter.get_fps()
            except Exception:
                last_fps = 0
        
        try:
            signal_count = len(plot_unit.data)
        except Exception:
            signal_count = 0
          # Draw background
        screen.fill(BG_COLOR)
        
        # Draw sidebar
        if sidebar_instance:
            # Convert enum value to the enum the sidebar expects if needed
            if use_external_view_mode:
                sidebar_instance.current_mode = current_view_mode
            else:
                # This handles the case where we're using a local ViewMode enum
                # but the sidebar expects the imported one
                mode_value = current_view_mode.value
                try:
                    from src.plot.view_mode import ViewMode as ExternalViewMode
                    sidebar_instance.current_mode = list(ExternalViewMode)[mode_value]
                except (ImportError, IndexError):
                    try:
                        from plot.view_mode import ViewMode as ExternalViewMode
                        sidebar_instance.current_mode = list(ExternalViewMode)[mode_value]
                    except (ImportError, IndexError):
                        # Can't update mode, just keep going
                        pass
            
            # Draw sidebar using the Sidebar component
            sidebar_instance.draw()
        else:
            # Fallback to drawing the sidebar manually
            pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
            
            # Draw tabs
            tab_height = 60
            for i, (label, _) in enumerate(TABS):
                tab_y = 50 + i*tab_height
                
                # Tab background
                color = TAB_SELECTED if i == selected_tab else TAB_COLOR
                pygame.draw.rect(screen, color, (5, tab_y, SIDEBAR_WIDTH-10, tab_height-10))
                
                # Tab label
                txt = font.render(label, True, FONT_COLOR)
                screen.blit(txt, (SIDEBAR_WIDTH//2 - txt.get_width()//2, tab_y + 10))
                
                # Connection indicator dot
                dot_color = DOT_ON if connection_status[i] else DOT_OFF
                pygame.draw.circle(screen, dot_color, (SIDEBAR_WIDTH//2, tab_y + 40), 6)
        
        # Draw status bar
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        status = f"Mode: {TABS[selected_tab][0]} | Signals: {signal_count} | FPS: {int(last_fps)}"
        txt = font.render(status, True, FONT_COLOR)
        screen.blit(txt, (10, 6))
        
        # Draw static plots if PlotUnit isn't handling the rendering
        if not getattr(plot_unit, 'initialized', True):
            # Plot area coordinates (outside sidebar)
            plot_x = SIDEBAR_WIDTH + 10
            plot_y = STATUS_BAR_HEIGHT + 10
            plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 20
            plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 20) // 3  # Divide space for 3 plots
            
            # Draw plot area background
            pygame.draw.rect(screen, (20, 20, 20), 
                            (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 20))
            
            # Draw static plots based on current tab
            if selected_tab == 0:  # RAW
                # Draw raw signal plots with warm colors
                signals = [k for k in static_plots if 'RAW' in k]
                colors = [(255, 180, 0), (255, 120, 0), (255, 60, 0)]  # Warm colors for raw
                
                # Render the signals
                for i, signal in enumerate(signals[:3]):  # Show up to 3 plots
                    if signal in static_plots:
                        render_static_plot(
                            screen, 
                            static_plots[signal],
                            plot_x, 
                            plot_y + i * plot_height, 
                            plot_width,
                            plot_height - 10,
                            colors[i % len(colors)]
                        )
                        
                        # Draw signal label
                        label = font.render(signal, True, FONT_COLOR)
                        screen.blit(label, (plot_x + 10, plot_y + i * plot_height + 10))
                        
            elif selected_tab == 1:  # PROCESSED
                # Draw processed signal plots with cool colors
                signals = [k for k in static_plots if 'PROCESSED' in k]
                colors = [(0, 180, 255), (0, 120, 255), (0, 60, 255)]  # Cool colors for processed
                
                # Render the signals
                for i, signal in enumerate(signals[:3]):  # Show up to 3 plots
                    if signal in static_plots:
                        render_static_plot(
                            screen, 
                            static_plots[signal],
                            plot_x, 
                            plot_y + i * plot_height, 
                            plot_width,
                            plot_height - 10,
                            colors[i % len(colors)]
                        )
                        
                        # Draw signal label
                        label = font.render(signal, True, FONT_COLOR)
                        screen.blit(label, (plot_x + 10, plot_y + i * plot_height + 10))
                        
            elif selected_tab == 2:  # TWIN
                # For twin view, divide the screen horizontally
                left_width = plot_width // 2 - 5
                right_width = plot_width // 2 - 5
                center_x = plot_x + left_width + 5
                
                # Draw a separator line
                pygame.draw.line(screen, (50, 50, 50), 
                                 (center_x, plot_y), 
                                 (center_x, WINDOW_HEIGHT - 10), 3)
                
                # Draw processed signals on left side
                processed_signals = [k for k in static_plots if 'PROCESSED' in k]
                cool_colors = [(0, 180, 255), (0, 120, 255), (0, 60, 255)]  # Cool colors for processed
                
                for i, signal in enumerate(processed_signals[:3]):
                    if signal in static_plots:
                        render_static_plot(
                            screen, 
                            static_plots[signal],
                            plot_x, 
                            plot_y + i * plot_height, 
                            left_width,
                            plot_height - 10,
                            cool_colors[i % len(cool_colors)]
                        )
                        
                        # Draw signal label
                        label = font.render(signal, True, FONT_COLOR)
                        screen.blit(label, (plot_x + 10, plot_y + i * plot_height + 10))
                
                # Draw raw signals on right side
                raw_signals = [k for k in static_plots if 'RAW' in k]
                warm_colors = [(255, 180, 0), (255, 120, 0), (255, 60, 0)]  # Warm colors for raw
                
                for i, signal in enumerate(raw_signals[:3]):
                    if signal in static_plots:
                        render_static_plot(
                            screen, 
                            static_plots[signal],
                            center_x + 5, 
                            plot_y + i * plot_height, 
                            right_width,
                            plot_height - 10,
                            warm_colors[i % len(warm_colors)]
                        )
                        
                        # Draw signal label
                        label = font.render(signal, True, FONT_COLOR)
                        screen.blit(label, (center_x + 15, plot_y + i * plot_height + 10))
                        
            elif selected_tab == 3:  # SETTINGS
                # Draw settings buttons instead of plots
                button_height = 40
                button_width = 180
                button_margin = 20
                start_y = plot_y + 30
                
                # Define buttons (fake interactive elements)
                settings_buttons = [
                    {"label": "Reset Plots", "color": (180, 60, 60)},
                    {"label": "Performance Mode", "color": (60, 60, 180), "toggle": True, "state": settings.get('performance_mode', False)},
                    {"label": "FPS Cap", "color": (60, 60, 180), "toggle": True, "state": settings.get('caps_enabled', True)},
                    {"label": "Light Mode", "color": (60, 60, 180), "toggle": True, "state": settings.get('light_mode', False)},
                ]
                
                # Draw each button
                for i, btn in enumerate(settings_buttons):
                    y_pos = start_y + i * (button_height + button_margin)
                    
                    # Button background
                    color = btn["color"]
                    if btn.get("toggle", False) and btn.get("state", False):
                        # Highlight active toggle buttons
                        color = tuple(min(c + 60, 255) for c in color)
                    
                    pygame.draw.rect(screen, color, 
                                    (plot_x + 50, y_pos, button_width, button_height),
                                    border_radius=5)
                    
                    # Button text
                    btn_label = font.render(btn["label"], True, FONT_COLOR)
                    screen.blit(btn_label, (plot_x + 50 + (button_width - btn_label.get_width()) // 2, 
                                            y_pos + (button_height - btn_label.get_height()) // 2))
                
                # Draw settings title
                title = font.render("SETTINGS", True, FONT_COLOR)
                screen.blit(title, (plot_x + 50 + (button_width - title.get_width()) // 2, plot_y))
                
                # Draw connection info
                info_text = f"Connected nodes: {settings.get('connected_nodes', 0)}"
                info = font.render(info_text, True, FONT_COLOR)
                screen.blit(info, (plot_x + 50, start_y + len(settings_buttons) * (button_height + button_margin) + 20))
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    # Clean up before exit
    stop_signal_updates()
    pygame.quit()
    logger.info("UI shut down")

if __name__ == "__main__":
    main()
