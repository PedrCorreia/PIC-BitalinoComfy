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
from collections import deque

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

# Try to import LiveSignalAdapter
try:
    from src.plot.live_signal_adapter import LiveSignalAdapter
    logger.info("Imported LiveSignalAdapter from src.plot.live_signal_adapter")
    live_adapter_imported = True
except ImportError:
    try:
        from plot.live_signal_adapter import LiveSignalAdapter
        logger.info("Imported LiveSignalAdapter from plot.live_signal_adapter")
        live_adapter_imported = True
    except ImportError:
        logger.warning("Could not import LiveSignalAdapter, will use direct signal copying")
        live_adapter_imported = False

# Try to import SignalRegistry
try:
    from src.registry.signal_registry import SignalRegistry
    logger.info("Imported SignalRegistry from src.registry.signal_registry")
    signal_registry_imported = True
except ImportError:
    try:
        from registry.signal_registry import SignalRegistry
        logger.info("Imported SignalRegistry from registry.signal_registry")
        signal_registry_imported = True
    except ImportError:
        logger.warning("Could not import SignalRegistry, will use PlotRegistry only")
        signal_registry_imported = False

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
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.flatten()
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

def generate_live_signals(registry):
    """
    Generate live signals that update over time.
    
    Args:
        registry: Registry to add signals to
    
    Returns:
        bool: True if signals were generated, False otherwise
    """
    try:
        if not hasattr(registry, 'signals'):
            return False
            
        # Generate timestamps
        t = np.linspace(0, 10, 1000)
        
        # Create sine wave with frequency that changes over time
        sine_raw = 0.8 * np.sin(2 * np.pi * (0.5 + 0.1 * np.sin(0.1 * t)) * t)
        sine_raw += np.random.normal(0, 0.08, size=len(sine_raw))
        registry.signals['SINE_RAW'] = sine_raw
        
        # Create clean version for processed view
        sine_processed = 0.8 * np.sin(2 * np.pi * (0.5 + 0.1 * np.sin(0.1 * t)) * t)
        registry.signals['SINE_PROCESSED'] = sine_processed
        
        # Create square wave that varies in duty cycle
        duty = 0.5 + 0.3 * np.sin(0.2 * t)
        square_raw = np.zeros_like(t)
        for i in range(len(t)):
            if (t[i] % 1) < duty[i]:
                square_raw[i] = 0.8
            else:
                square_raw[i] = -0.8
        square_raw += np.random.normal(0, 0.1, size=len(square_raw))
        registry.signals['SQUARE_RAW'] = square_raw
        
        # Clean version
        square_processed = np.zeros_like(t)
        for i in range(len(t)):
            if (t[i] % 1) < duty[i]:
                square_processed[i] = 0.8
            else:
                square_processed[i] = -0.8
        registry.signals['SQUARE_PROCESSED'] = square_processed
        
        # Create sawtooth wave
        sawtooth_raw = 0.7 * (2 * (t * 0.4 % 1) - 1)
        sawtooth_raw += np.random.normal(0, 0.07, size=len(sawtooth_raw))
        registry.signals['SAWTOOTH_RAW'] = sawtooth_raw
        
        # Clean version
        sawtooth_processed = 0.7 * (2 * (t * 0.4 % 1) - 1)
        registry.signals['SAWTOOTH_PROCESSED'] = sawtooth_processed
        
        # Add metadata if the registry supports it
        if hasattr(registry, 'metadata'):
            registry.metadata = {
                'SINE_RAW': {'color': (255, 100, 100), 'source': 'synthetic'},
                'SINE_PROCESSED': {'color': (100, 100, 255), 'source': 'synthetic'},
                'SQUARE_RAW': {'color': (255, 150, 100), 'source': 'synthetic'},
                'SQUARE_PROCESSED': {'color': (100, 150, 255), 'source': 'synthetic'},
                'SAWTOOTH_RAW': {'color': (255, 200, 100), 'source': 'synthetic'},
                'SAWTOOTH_PROCESSED': {'color': (100, 200, 255), 'source': 'synthetic'},
            }
        
        return True
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return False

def update_live_signals(registry):
    """
    Update the live signals with small changes to simulate real-time data.
    
    Args:
        registry: Registry containing signals to update
    
    Returns:
        bool: True if signals were updated, False otherwise
    """
    try:
        if not hasattr(registry, 'signals'):
            return False
            
        # Update each signal by shifting and adding new noise
        for key in list(registry.signals.keys()):
            if key.endswith('_RAW') or key.endswith('_PROCESSED'):
                signal = registry.signals[key]
                
                if len(signal) > 0:
                    # Phase shift
                    shifted = np.roll(signal, 10)
                    
                    # Add noise only to RAW signals
                    if key.endswith('_RAW'):
                        noise_level = 0.02
                        shifted += np.random.normal(0, noise_level, size=len(shifted))
                    
                    registry.signals[key] = shifted
        
        # Update timestamps in metadata if available
        if hasattr(registry, 'metadata'):
            now = time.time()
            for key in registry.metadata:
                if 'timestamp' in registry.metadata[key]:
                    registry.metadata[key]['timestamp'] = now
        
        return True
    except Exception as e:
        logger.error(f"Error updating live signals: {e}")
        return False

class SignalGeneratorThread(threading.Thread):
    """Thread for generating and updating signals in the background."""
    
    def __init__(self, registry):
        """
        Initialize the signal generator thread.
        
        Args:
            registry: The registry to update with signals
        """
        super().__init__()
        self.registry = registry
        self.running = True
        self.daemon = True
        self.update_interval = 0.1  # Update signals 10 times per second
        
    def run(self):
        """Run the signal generator thread."""
        logger.info("Signal generator thread starting")
        
        # Generate initial signals
        generate_live_signals(self.registry)
        
        # Update signals periodically
        while self.running:
            update_live_signals(self.registry)
            time.sleep(self.update_interval)
        
        logger.info("Signal generator thread stopped")
    
    def stop(self):
        """Stop the signal generator thread."""
        self.running = False

# --- Synthetic Data Generator for Live Signals ---
class SyntheticSignalGenerator:
    def __init__(self, signal_id, kind, freq=1.0, noise=0.05, phase=0.0, amplitude=1.0):
        self.signal_id = signal_id
        self.kind = kind  # 'sine', 'square', 'sawtooth'
        self.freq = freq
        self.noise = noise
        self.phase = phase
        self.amplitude = amplitude
        self.metadata = {'name': f'{kind.capitalize()} {signal_id}', 'color': (0, 200, 100)}
    def generate(self, t):
        if self.kind == 'sine':
            return self.amplitude * np.sin(2 * np.pi * self.freq * t + self.phase) + np.random.normal(0, self.noise)
        elif self.kind == 'square':
            return self.amplitude * np.sign(np.sin(2 * np.pi * self.freq * t + self.phase)) + np.random.normal(0, self.noise)
        elif self.kind == 'sawtooth':
            return self.amplitude * (2 * ((self.freq * t + self.phase/(2*np.pi)) % 1) - 1) + np.random.normal(0, self.noise)
        else:
            return np.random.normal(0, self.noise)

# --- Live Signal Generator Thread using SyntheticSignalGenerator ---
class LiveSignalGeneratorThread(threading.Thread):
    def __init__(self, generator, update_interval=0.05, window_seconds=10, buffer_seconds=60):
        super().__init__()
        self.generator = generator
        self.update_interval = update_interval
        self.running = True
        self.daemon = True
        self.window_seconds = window_seconds
        self.buffer_seconds = buffer_seconds
        self.maxlen = int(buffer_seconds / update_interval)
        self.t_deque = deque(maxlen=self.maxlen)
        self.data_deque = deque(maxlen=self.maxlen)
    def run(self):
        try:
            from src.registry.signal_registry import SignalRegistry
            registry = SignalRegistry.get_instance()
        except ImportError:
            from registry.signal_registry import SignalRegistry
            registry = SignalRegistry.get_instance()
        while self.running:
            now = time.time()
            val = self.generator.generate(now)
            self.t_deque.append(now)
            self.data_deque.append(val)
            meta = self.generator.metadata.copy()
            meta['timestamp'] = float(self.t_deque[-1]) if len(self.t_deque) else now
            registry.register_signal(self.generator.signal_id, (list(self.t_deque), list(self.data_deque)), meta)
            time.sleep(self.update_interval)
    def stop(self):
        self.running = False
    def set_window_seconds(self, window_seconds):
        self.window_seconds = window_seconds
        self.maxlen = int(window_seconds / self.update_interval)
        self.t_deque = deque(self.t_deque, maxlen=self.maxlen)
        self.data_deque = deque(self.data_deque, maxlen=self.maxlen)

# --- Main UI App ---
def main():
    """Main UI application for PIC-2025 visualization."""
    print("\n=== PIC-2025 Visualization Interface ===\n")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Visualization")
    try:
        from src.plot.constants import DEFAULT_FONT, DEFAULT_FONT_SIZE
        font = pygame.font.SysFont(DEFAULT_FONT, DEFAULT_FONT_SIZE)
    except ImportError:
        font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    # --- Registry and PlotUnit Setup ---
    try:
        plot_unit = PlotUnit.get_instance()
        plot_unit.width = WINDOW_WIDTH
        plot_unit.height = WINDOW_HEIGHT
        plot_unit.sidebar_width = SIDEBAR_WIDTH
        plot_unit.status_bar_height = STATUS_BAR_HEIGHT
        plot_unit.plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH
        plot_unit.running = True
        if not hasattr(plot_unit, 'data_lock'):
            plot_unit.data_lock = threading.Lock()
        if not hasattr(plot_unit, 'data'):
            plot_unit.data = {}
        if hasattr(plot_unit, 'start') and callable(plot_unit.start):
            if not getattr(plot_unit, 'initialized', False):
                plot_unit.start()
                plot_unit.initialized = True
    except Exception as e:
        logger.error(f"Error initializing PlotUnit: {e}")
        print(f"Error initializing PlotUnit: {e}")
        input("Press Enter to exit...")
        return

    # --- Connect PlotUnit to PlotRegistry using adapter if available ---
    try:
        from src.registry.plot_generator_debug_fixed import PlotUnitRegistryAdapter
        adapter = PlotUnitRegistryAdapter(plot_unit)
        adapter.connect()
        logger.info("PlotUnitRegistryAdapter connected PlotUnit to PlotRegistry.")
    except ImportError:
        try:
            from src.registry.plot_generator_debug import PlotUnitRegistryAdapter
            adapter = PlotUnitRegistryAdapter(plot_unit)
            adapter.connect()
            logger.info("PlotUnitRegistryAdapter (fallback) connected PlotUnit to PlotRegistry.")
        except ImportError:
            logger.warning("PlotUnitRegistryAdapter not available; skipping registry connection.")
    except Exception as e:
        logger.warning(f"Could not connect PlotUnitRegistryAdapter: {e}")

    # --- Signal Registry Setup ---
    try:
        registry = get_registry_instance()
        if registry and hasattr(registry, 'signals'):
            registry.signals.clear()
        if registry and hasattr(registry, 'metadata'):
            registry.metadata.clear()
    except Exception as e:
        logger.warning(f"Could not clear registry before starting live signals: {e}")

    # --- Live Signal Generators ---
    generator_threads = [
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SINE_RAW', 'sine', freq=0.5, noise=0.08, amplitude=0.9)),
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SINE_PROCESSED', 'sine', freq=0.5, noise=0.01, amplitude=0.9, phase=0.2)),
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SQUARE_RAW', 'square', freq=0.3, noise=0.1, amplitude=0.8)),
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SQUARE_PROCESSED', 'square', freq=0.3, noise=0.01, amplitude=0.8, phase=0.1)),
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SAWTOOTH_RAW', 'sawtooth', freq=0.4, noise=0.07, amplitude=0.7)),
        LiveSignalGeneratorThread(SyntheticSignalGenerator('SAWTOOTH_PROCESSED', 'sawtooth', freq=0.4, noise=0.01, amplitude=0.7, phase=0.1)),
    ]
    for thread in generator_threads:
        thread.start()

    # --- LiveSignalAdapter: Bridge SignalRegistry to PlotRegistry ---
    signal_adapter = None
    if live_adapter_imported and signal_registry_imported:
        try:
            plot_registry = PlotRegistry.get_instance()
            signal_registry = SignalRegistry.get_instance()
            signal_adapter = LiveSignalAdapter.get_instance()
            signal_adapter.connect_registries(signal_registry, plot_registry)
            signal_adapter.start(view_mode=ViewMode.RAW)
        except Exception as e:
            logger.warning(f"Could not start LiveSignalAdapter: {e}")

    # --- UI State ---
    selected_tab = 0
    running = True
    last_fps = 0
    signal_count = 0
    current_view_mode = ViewMode.RAW
    settings = {
        'caps_enabled': True,
        'light_mode': False,
        'performance_mode': False,
        'connected_nodes': 0,
        'window_seconds': 10,
    }
    # --- Sidebar ---
    sidebar_instance = None
    if use_sidebar_component:
        try:
            icon_font = pygame.font.SysFont(None, 28)
        except:
            icon_font = font
        try:
            sidebar_instance = Sidebar(screen, SIDEBAR_WIDTH, WINDOW_HEIGHT, font, icon_font, current_view_mode, settings)
        except Exception as e:
            logger.error(f"Error initializing Sidebar: {e}")
            sidebar_instance = None

    # --- Per-plot deques ---
    plot_deques = {}
    plot_deque_maxlen = int(settings['window_seconds'] * 20)
    def update_plot_deques():
        for k, v in plot_unit.data.items():
            if isinstance(v, tuple) and len(v) == 2:
                t, data = v
                if k not in plot_deques:
                    plot_deques[k] = {
                        't': deque(maxlen=plot_deque_maxlen),
                        'data': deque(maxlen=plot_deque_maxlen)
                    }
                dq = plot_deques[k]
                # Instead of appending, always replace with the latest data from the registry
                dq['t'].clear()
                dq['data'].clear()
                for ti, di in zip(t, data):
                    if np.isscalar(ti) and np.isscalar(di) and np.isreal(ti) and np.isreal(di):
                        dq['t'].append(float(np.real(ti)))
                        dq['data'].append(float(np.real(di)))
        # Remove deques for signals no longer present
        for k in list(plot_deques.keys()):
            if k not in plot_unit.data:
                del plot_deques[k]

    # --- Main Loop ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if sidebar_instance and mx < SIDEBAR_WIDTH:
                    clicked_tab = sidebar_instance.handle_click(my)
                    if clicked_tab is not None and clicked_tab != selected_tab:
                        selected_tab = clicked_tab
                        current_view_mode = TABS[selected_tab][1]
                        if hasattr(sidebar_instance, 'update_dynamic_state'):
                            sidebar_instance.update_dynamic_state(current_view_mode, settings)
                else:
                    tab_height = 60
                    for i, (label, _) in enumerate(TABS):
                        tab_y = 50 + i*tab_height
                        if 0 <= mx < SIDEBAR_WIDTH and tab_y <= my < tab_y + tab_height:
                            if selected_tab != i:
                                selected_tab = i
                                current_view_mode = TABS[selected_tab][1]
        # --- Update metrics and state ---
        try:
            plot_registry = PlotRegistry.get_instance()
            if hasattr(plot_registry, 'signals') and isinstance(plot_registry.signals, dict):
                signal_count = len(plot_registry.signals)
            else:
                signal_count = 0
        except Exception:
            signal_count = 0
        settings['connected_nodes'] = signal_count
        update_plot_deques()
        if sidebar_instance and hasattr(sidebar_instance, 'update_dynamic_state'):
            sidebar_instance.update_dynamic_state(current_view_mode, settings)
        # --- Draw UI ---
        screen.fill(BG_COLOR)
        plot_x = SIDEBAR_WIDTH + 10
        plot_y = STATUS_BAR_HEIGHT + 10
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 20
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 20) // 3
        pygame.draw.rect(screen, (20, 20, 20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 20))
        now = time.time()
        # --- Plot Drawing ---
        def draw_live_plot(signal, color, x, y, w, h):
            if signal in plot_deques:
                dq = plot_deques[signal]
                t = np.asarray(dq['t'])
                data = np.asarray(dq['data'])
                if t.ndim > 1:
                    t = t.flatten()
                if data.ndim > 1:
                    data = data.flatten()
                if len(t) < 2 or len(data) < 2 or len(t) != len(data):
                    return
                t0 = t[0]
                t = t - t0
                window = settings.get('window_seconds', 10)
                mask = t >= (t[-1] - window)
                if len(mask) != len(t):
                    return
                t = t[mask]
                data = data[mask]
                if len(t) < 2 or len(data) < 2:
                    return
                data_min = np.min(data)
                data_max = np.max(data)
                if data_max == data_min:
                    data_max = data_min + 1
                points = []
                for j in range(min(len(t), w)):
                    idx = int(j * len(t) / w)
                    px = x + j * (w / min(len(t), w))
                    py = y + h - ((data[idx] - data_min) / (data_max - data_min)) * h
                    points.append((int(px), int(py)))
                if len(points) >= 2:
                    pygame.draw.lines(screen, color, False, points, 2)
                label = font.render(f"{signal}  t=[{t[0]:.1f},{t[-1]:.1f}]", True, FONT_COLOR)
                screen.blit(label, (x + 10, y + 10))
        # --- Tab-specific plot logic ---
        if selected_tab == 3:  # SETTINGS
            button_height = 40
            button_width = 180
            button_margin = 20
            start_y = plot_y + 30
            settings_buttons = [
                {"label": "Reset Plots", "color": (180, 60, 60)},
                {"label": "Performance Mode", "color": (60, 60, 180), "toggle": True, "state": settings.get('performance_mode', False)},
                {"label": "FPS Cap", "color": (60, 60, 180), "toggle": True, "state": settings.get('caps_enabled', True)},
                {"label": "Light Mode", "color": (60, 60, 180), "toggle": True, "state": settings.get('light_mode', False)},
            ]
            for i, btn in enumerate(settings_buttons):
                y_pos = start_y + i * (button_height + button_margin)
                color = btn["color"]
                if btn.get("toggle", False) and btn.get("state", False):
                    color = tuple(min(c + 60, 255) for c in color)
                pygame.draw.rect(screen, color, (plot_x + 50, y_pos, button_width, button_height), border_radius=5)
                btn_label = font.render(btn["label"], True, FONT_COLOR)
                screen.blit(btn_label, (plot_x + 50 + (button_width - btn_label.get_width()) // 2, y_pos + (button_height - btn_label.get_height()) // 2))
            # --- Rolling window size control ---
            window_label = font.render(f"Rolling Window (s): {settings['window_seconds']}", True, FONT_COLOR)
            slider_x = plot_x + 50
            slider_y = start_y + len(settings_buttons) * (button_height + button_margin) + 40
            slider_w = 200
            slider_h = 8
            min_window = 2
            max_window = 60
            # Draw slider background
            pygame.draw.rect(screen, (80, 80, 80), (slider_x, slider_y, slider_w, slider_h), border_radius=4)
            # Draw slider handle
            handle_pos = int((settings['window_seconds'] - min_window) / (max_window - min_window) * (slider_w - 16))
            pygame.draw.rect(screen, (180, 180, 80), (slider_x + handle_pos, slider_y - 6, 16, slider_h + 12), border_radius=6)
            screen.blit(window_label, (slider_x, slider_y - 28))
            # Save slider rect for click detection
            window_slider_rect = pygame.Rect(slider_x, slider_y - 6, slider_w, slider_h + 12)
            # Info text
            info_text = f"Connected nodes: {settings.get('connected_nodes', 0)}"
            info = font.render(info_text, True, FONT_COLOR)
            screen.blit(info, (plot_x + 50, slider_y + 32))
        else:
            # Live plot mode for RAW, PROCESSED, TWIN
            # DEBUG: print available signals
            logger.info(f"plot_unit.data keys: {list(plot_unit.data.keys())}")
            if selected_tab == 0:  # RAW
                # Accept any signal ending with _RAW (not just containing 'RAW')
                plot_signals = [k for k in plot_unit.data if k.endswith('_RAW') or k.upper().endswith('RAW')]
                plot_colors = [(255, 180, 0), (255, 120, 0), (255, 60, 0)]
                for i, signal in enumerate(plot_signals[:3]):
                    draw_live_plot(signal, plot_colors[i % len(plot_colors)], plot_x, plot_y + i * plot_height, plot_width, plot_height)
                if not plot_signals:
                    msg = font.render(f"No plot data available. Signals: {list(plot_unit.data.keys())}", True, (200, 50, 50))
                    screen.blit(msg, (plot_x + 20, plot_y + 20))
            elif selected_tab == 1:  # PROCESSED
                plot_signals = [k for k in plot_unit.data if k.endswith('_PROCESSED') or 'PROC' in k.upper()]
                plot_colors = [(0, 180, 255), (0, 120, 255), (0, 60, 255)]
                for i, signal in enumerate(plot_signals[:3]):
                    draw_live_plot(signal, plot_colors[i % len(plot_colors)], plot_x, plot_y + i * plot_height, plot_width, plot_height)
                if not plot_signals:
                    msg = font.render(f"No plot data available. Signals: {list(plot_unit.data.keys())}", True, (200, 50, 50))
                    screen.blit(msg, (plot_x + 20, plot_y + 20))
            elif selected_tab == 2:  # TWIN
                left_width = plot_width // 2 - 5
                right_width = plot_width // 2 - 5
                center_x = plot_x + left_width + 5
                pygame.draw.line(screen, (50, 50, 50), (center_x, plot_y), (center_x, WINDOW_HEIGHT - 10), 3)
                processed_signals = [k for k in plot_unit.data if k.endswith('_PROCESSED') or 'PROC' in k.upper()]
                cool_colors = [(0, 180, 255), (0, 120, 255), (0, 60, 255)]
                for i, signal in enumerate(processed_signals[:3]):
                    draw_live_plot(signal, cool_colors[i % len(cool_colors)], plot_x, plot_y + i * plot_height, left_width, plot_height)
                raw_signals = [k for k in plot_unit.data if k.endswith('_RAW') or k.upper().endswith('RAW')]
                warm_colors = [(255, 180, 0), (255, 120, 0), (255, 60, 0)]
                for i, signal in enumerate(raw_signals[:3]):
                    draw_live_plot(signal, warm_colors[i % len(warm_colors)], center_x + 5, plot_y + i * plot_height, right_width, plot_height)
                if not processed_signals and not raw_signals:
                    msg = font.render(f"No plot data available. Signals: {list(plot_unit.data.keys())}", True, (200, 50, 50))
                    screen.blit(msg, (plot_x + 20, plot_y + 20))
        # --- Status Bar ---
        runtime = int(now - getattr(main, '_start_time', now))
        if not hasattr(main, '_start_time'):
            main._start_time = now
            runtime = 0
        status = f"Mode: {TABS[selected_tab][0]} | Signals: {signal_count} | FPS: {int(last_fps)} | Runtime: {runtime}s"
        txt = font.render(status, True, FONT_COLOR)
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        screen.blit(txt, (10, 6))
        # --- Sidebar ---
        if sidebar_instance:
            if hasattr(sidebar_instance, 'update_dynamic_state'):
                sidebar_instance.update_dynamic_state(current_view_mode, settings)
            sidebar_instance.draw()
        else:
            for i, (label, _) in enumerate(TABS):
                tab_rect = pygame.Rect(0, 50 + i * 60, SIDEBAR_WIDTH, 60)
                color = TAB_SELECTED if i == selected_tab else TAB_COLOR
                pygame.draw.rect(screen, color, tab_rect)
                tab_label = font.render(label, True, FONT_COLOR)
                screen.blit(tab_label, (10, 50 + i * 60 + 18))
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
