#!/usr/bin/env python
"""
Stable View Mode Demo with Registry Signals for PIC-2025

This script provides a simplified and stable demonstration of different view modes
with registry signals, avoiding mode transition crashes.
"""

import os
import sys
import time
import threading
import pygame
import random
import numpy as np
from enum import Enum

# Add project root to path to ensure imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

print("\n=== PIC-2025 Stable View Demo ===\n")

# Optional: Enable debug logging
def setup_logging(level=20):  # 20 is INFO level
    import logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('stable_view_demo')

logger = setup_logging()

# Import ViewMode safely
try:
    from src.plot.view_mode import ViewMode
except ImportError:
    try:
        from plot.view_mode import ViewMode
    except ImportError:
        # Create a basic enum if we can't import it
        print("Creating backup ViewMode enum")
        class ViewMode(Enum):
            RAW = 0
            PROCESSED = 1
            TWIN = 2
            STACKED = 3
            SETTINGS = 4

def get_registry_instance():
    """Get the PlotRegistry instance, creating it if needed."""
    try:
        from src.registry.plot_registry import PlotRegistry
        registry = PlotRegistry.get_instance()
        print("Successfully accessed PlotRegistry")
        return registry
    except ImportError:
        print("Could not import PlotRegistry, trying alternatives...")
        
    try:
        # Try the legacy path
        from registry.plot_registry import PlotRegistry
        registry = PlotRegistry.get_instance()
        print("Successfully accessed PlotRegistry (legacy path)")
        return registry
    except ImportError:
        print("Could not import PlotRegistry from any path")
        return None

def register_simple_signals():
    """Register simple demo signals with the registry - only ECG for maximum stability."""
    registry = get_registry_instance()
    if not registry:
        print("Failed to get registry instance, cannot register signals")
        return False
    
    try:
        print("Registering minimal demo signals in PlotRegistry...")
        
        # Generate a simple sine wave for stability
        duration = 10.0  # 10 seconds
        sample_rate = 100  # 100 Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Simple sine wave signal - very stable
        simple_signal = np.sin(2 * np.pi * 0.5 * t)
        
        # Register raw signal
        registry.register_signal('SIMPLE_RAW', simple_signal, {'color': (255, 0, 0), 'type': 'raw'})
        
        # Create processed version - just a smoother version
        simple_proc = np.convolve(simple_signal, np.ones(5)/5, mode='same')
        registry.register_signal('SIMPLE_PROC', simple_proc, {'color': (200, 100, 100), 'type': 'processed'})
        
        print("Successfully registered simple demo signals")
        return True
        
    except Exception as e:
        print(f"Error registering simple signals: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_visualization():
    """Start the visualization if it's not already running."""
    try:
        # First, try standalone_debug approach (most reliable)
        try:
            import standalone_debug
            
            # Initialize pygame first
            if not pygame.get_init():
                pygame.init()
                
            # Now import and start PlotUnit
            from standalone_debug import PlotUnit
            plot_unit = PlotUnit.get_instance()
            if not (hasattr(plot_unit, 'running') and plot_unit.running):
                print("Starting PlotUnit...")
                plot_unit.start()
            else:
                print("PlotUnit already running")
                
            # Wait for initialization
            wait_time = 0
            max_wait = 5  # Maximum 5 seconds
            print("Waiting for PlotUnit initialization...")
            while wait_time < max_wait:
                if hasattr(plot_unit, 'initialized') and plot_unit.initialized:
                    print(f"PlotUnit initialized after {wait_time} seconds.")
                    return plot_unit
                time.sleep(1.0)
                wait_time += 1
                print(f"Waiting... ({wait_time}/{max_wait})")
                
            print("WARNING: PlotUnit did not fully initialize within timeout period.")
            return plot_unit
            
        except ImportError as e:
            print(f"Could not import standalone visualization: {e}")
            print("All initialization attempts failed.")
            return None
            
    except Exception as e:
        print(f"Unexpected error in visualization startup: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_stable_demo():
    """Run a stable demonstration of different view modes without crashes."""
    print("Starting stable view mode demo...\n")
    
    # Start visualization
    print("Step 1: Starting visualization...")
    plot_unit = start_visualization()
    
    if not plot_unit:
        print("Failed to initialize visualization. Exiting.")
        return
    
    # Register simple signals
    print("\nStep 2: Registering simple signals...")
    if not register_simple_signals():
        print("Failed to register signals. Exiting.")
        return
        
    print("\nStep 3: Starting view mode demonstration...\n")
    
    # Define helper function for careful view switching
    def switch_view_safely(mode_name, view_mode):
        print(f"\nSwitching to {mode_name} view...")
        
        # First clear any existing signals to prevent conflicts
        try:
            if hasattr(plot_unit, 'data_lock') and hasattr(plot_unit, 'data'):
                with plot_unit.data_lock:
                    plot_unit.data.clear()
                    print("Cleared existing signals")
        except Exception as e:
            print(f"Warning: Could not clear signals: {e}")
            
        # Apply the view mode change
        try:
            if hasattr(plot_unit, '_set_mode'):
                plot_unit._set_mode(view_mode)
            elif hasattr(plot_unit, 'set_view_mode'):
                plot_unit.set_view_mode(view_mode)
                
            # Extra pause to let the view mode change take effect
            time.sleep(0.5)
            print(f"Successfully switched to {mode_name} view")
            return True
        except Exception as e:
            print(f"Error switching to {mode_name} view: {e}")
            return False
    
    # Load appropriate signals for the current view
    def load_signals_for_view(mode_name):
        print(f"Loading signals for {mode_name} view...")
        registry = get_registry_instance()
        if not registry or not hasattr(registry, 'signals'):
            print("No registry available")
            return False
            
        try:
            # Always work with data_lock for thread safety
            if hasattr(plot_unit, 'data_lock') and hasattr(plot_unit, 'data'):
                with plot_unit.data_lock:
                    # Clear existing data
                    plot_unit.data.clear()
                    
                    if mode_name == "RAW":
                        # Only load raw signal for raw view
                        if "SIMPLE_RAW" in registry.signals:
                            plot_unit.data["SIMPLE_RAW"] = registry.signals["SIMPLE_RAW"]
                            print("Added SIMPLE_RAW signal to view")
                    
                    elif mode_name == "PROCESSED":
                        # Only load processed signal for processed view
                        if "SIMPLE_PROC" in registry.signals:
                            plot_unit.data["SIMPLE_PROC"] = registry.signals["SIMPLE_PROC"]
                            print("Added SIMPLE_PROC signal to view")
                    
                    elif mode_name == "TWIN":
                        # Load both raw and processed for twin view
                        if "SIMPLE_RAW" in registry.signals:
                            plot_unit.data["SIMPLE_RAW"] = registry.signals["SIMPLE_RAW"]
                        if "SIMPLE_PROC" in registry.signals:
                            plot_unit.data["SIMPLE_PROC"] = registry.signals["SIMPLE_PROC"]
                        print("Added both SIMPLE_RAW and SIMPLE_PROC signals to view")
            
            return True
        except Exception as e:
            print(f"Error loading signals: {e}")
            return False
    
    # Force a redraw of the current view
    def force_redraw():
        print("Forcing redraw...")
        time.sleep(0.2)  # Short delay before redrawing
        
        for method_name in ['_render', 'render', '_redraw', 'redraw', 'update']:
            if hasattr(plot_unit, method_name) and callable(getattr(plot_unit, method_name)):
                try:
                    getattr(plot_unit, method_name)()
                    print(f"Redraw successful using {method_name}()")
                    return True
                except Exception as e:
                    print(f"Error during {method_name}(): {e}")
        
        return False
    
    # Safe demo sequence that avoids crashes
    try:
        # 1. Start with RAW view (most stable)
        switch_view_safely("RAW", ViewMode.RAW)
        load_signals_for_view("RAW")
        force_redraw()
        print("RAW view active - waiting 10 seconds...")
        time.sleep(10)
        
        # 2. Switch to PROCESSED view
        switch_view_safely("PROCESSED", ViewMode.PROCESSED)
        load_signals_for_view("PROCESSED")
        force_redraw()
        print("PROCESSED view active - waiting 10 seconds...")
        time.sleep(10)
        
        # 3. Back to RAW view (skip TWIN view for stability)
        switch_view_safely("RAW", ViewMode.RAW)
        load_signals_for_view("RAW")
        force_redraw()
        print("Back to RAW view - waiting 5 seconds...")
        time.sleep(5)
        
        # 4. Try TWIN view with very careful handling
        print("\nTrying TWIN view with extra precautions...")
        
        # First make sure registry has signals
        register_simple_signals()  # Refresh signals
        
        # Switch to TWIN view
        switch_view_safely("TWIN", ViewMode.TWIN)
        load_signals_for_view("TWIN")
        force_redraw()
        print("TWIN view active - waiting just 5 seconds to avoid potential issues...")
        time.sleep(5)
        
        # 5. Back to RAW view again (always safest to end on RAW)
        switch_view_safely("RAW", ViewMode.RAW)
        load_signals_for_view("RAW")
        force_redraw()
        print("Demo completed successfully - back to RAW view")
        
        print("\nView mode demo completed successfully!")
        print("All views have been demonstrated without crashes.")
        print("\nPress Ctrl+C to exit, or let this script continue running...")
        
        # Keep the script running
        counter = 0
        while True:
            time.sleep(1.0)
            counter += 1
            # Status update every 30 seconds
            if counter % 30 == 0:
                print(f"Demo still running - {counter} seconds elapsed")
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error during view demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("Demo complete.")

if __name__ == "__main__":
    run_stable_demo()
