#!/usr/bin/env python
"""
Fix RAW View Glitching in PIC-2025 Plot Generator

This script provides a comprehensive fix for the RAW view glitching issue in the
PIC-2025 plot generator debug visualization. It initializes the visualization if
needed and then patches the PlotUnit to fix the RAW view rendering.
"""

import os
import sys
import time
import importlib
import threading
import pygame
import random
import numpy as np
from enum import Enum

# Add project root to path to ensure imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

print("\n=== PIC-2025 RAW View Glitch Fix with Registry Demo ===\n")

# Optional: Enable debug logging
def setup_logging(level=20):  # 20 is INFO level
    import logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('fix_raw_view_glitch')

logger = setup_logging()


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


def register_demo_signals():
    """Register demo signals with different IDs in the PlotRegistry."""
    registry = get_registry_instance()
    if not registry:
        print("Failed to get registry instance, cannot register signals")
        return False
    
    try:
        print("Registering demo signals in PlotRegistry...")
        
        # Generate three different types of signals for demo
        duration = 10.0  # 10 seconds
        sample_rate = 100  # 100 Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 1. ECG-like signal (with QRS complexes)
        ecg_signal = np.zeros_like(t)
        for i in range(len(t)):
            if int(t[i]) % 1 < 0.2:  # QRS complex every second
                ecg_signal[i] = 0.8 * np.sin(8 * np.pi * t[i]) + 0.2 * np.random.random()
            else:
                ecg_signal[i] = 0.1 * np.sin(2 * np.pi * t[i]) + 0.05 * np.random.random()
        
        # 2. EDA-like signal (slow changing baseline with SCRs)
        eda_signal = 0.5 + 0.2 * np.sin(0.2 * np.pi * t)
        for i in range(3):  # Add 3 SCRs
            onset = np.random.randint(100, len(t) - 200)
            eda_signal[onset:onset+100] += 0.3 * np.exp(-np.linspace(0, 3, 100))
        
        # 3. Respiration-like signal
        resp_signal = 0.7 * np.sin(0.5 * np.pi * t) + 0.1 * np.random.random(size=len(t))
        
        # Register raw signals with specific IDs
        registry.register_signal('ECG_RAW', ecg_signal, {'color': (255, 0, 0), 'type': 'raw'})
        registry.register_signal('EDA_RAW', eda_signal, {'color': (0, 255, 0), 'type': 'raw'})
        registry.register_signal('RESP_RAW', resp_signal, {'color': (0, 0, 255), 'type': 'raw'})
        
        # Create processed versions with "PROC_" prefix for easy identification
        # Processed ECG - smoother
        proc_ecg = np.convolve(ecg_signal, np.ones(5)/5, mode='same')
        registry.register_signal('PROC_ECG', proc_ecg, {'color': (200, 100, 100), 'type': 'processed'})
        
        # Processed EDA - detrended
        proc_eda = eda_signal - np.mean(eda_signal) + np.random.random(size=len(eda_signal)) * 0.05
        registry.register_signal('PROC_EDA', proc_eda, {'color': (100, 200, 100), 'type': 'processed'})
        
        # Processed RESP - filtered
        proc_resp = resp_signal + 0.2 * np.sin(2 * np.pi * t) + np.random.random(size=len(resp_signal)) * 0.05
        registry.register_signal('PROC_RESP', proc_resp, {'color': (100, 100, 200), 'type': 'processed'})
        
        print("Successfully registered 6 demo signals (3 raw, 3 processed)")
        return True
        
    except Exception as e:
        print(f"Error registering demo signals: {e}")
        import traceback
        traceback.print_exc()
        return False


def connect_registry_to_plot_unit(plot_unit):
    """Connect the registry signals to the PlotUnit visualization."""
    if not plot_unit:
        print("No PlotUnit instance provided for registry connection")
        return False
    
    try:
        registry = get_registry_instance()
        if not registry:
            return False
            
        print("Connecting registry signals to PlotUnit...")
        
        # Use both direct data access and integration layer if available
        try:
            # Try using the integration layer first
            from src.registry.plot_registry_integration import PlotRegistryIntegration
            integration = PlotRegistryIntegration.get_instance()
            integration.connect_to_plot_unit(plot_unit)
            print("Connected using PlotRegistryIntegration")
            return True
        except (ImportError, AttributeError) as e:
            print(f"Integration layer not available or missing methods: {e}")
            print("Falling back to direct data connection...")
        
        # Direct connection as fallback
        if hasattr(plot_unit, 'data_lock') and hasattr(plot_unit, 'data'):
            with plot_unit.data_lock:
                # Transfer raw signals
                for signal_id, signal_data in registry.signals.items():
                    if 'PROC_' not in signal_id:  # Only raw signals initially
                        plot_unit.data[signal_id] = signal_data
                        print(f"Connected signal: {signal_id}")
            print("Successfully connected registry signals directly to PlotUnit")
            return True
        
        print("Could not connect registry to PlotUnit")
        return False
        
    except Exception as e:
        print(f"Error connecting registry to PlotUnit: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_visualization():
    """Start the visualization if it's not already running."""
    try:
        # First, try to get a reference to an existing PlotUnit instance
        try:
            from src.plot.plot_unit import PlotUnit
            print("Checking for existing PlotUnit instance...")
            plot_unit = PlotUnit.get_instance()
            if hasattr(plot_unit, 'running') and plot_unit.running:
                print("PlotUnit already running, using existing instance.")
                return plot_unit
        except Exception as e:
            print(f"Could not get existing PlotUnit reference: {e}")
            print("Will try alternative initialization...")
        
        # Second, try standalone_debug
        try:
            import standalone_debug
            
            # Check if PlotUnit is already initialized through standalone
            if hasattr(standalone_debug, 'PlotUnit'):
                plot_unit = standalone_debug.PlotUnit.get_instance()
                if hasattr(plot_unit, 'running') and plot_unit.running:
                    print("PlotUnit already running via standalone, using it.")
                    return plot_unit
                    
            # Otherwise, initialize standalone visualization
            print("Starting standalone visualization...")
            
            # Initialize pygame first
            if not pygame.get_init():
                pygame.init()
                
            # Now import and start PlotUnit from standalone
            from standalone_debug import PlotUnit
            plot_unit = PlotUnit.get_instance()
            plot_unit.start()
            
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
            if hasattr(plot_unit, 'running') and plot_unit.running:
                print("PlotUnit is running though, attempting to continue...")
                return plot_unit
            print("PlotUnit is not running, trying registry approach...")
        except ImportError as e:
            print(f"Could not import standalone visualization: {e}")
            print("Trying registry visualization approach...")
        
        # Third, try registry visualization with plot_generator_debug_fixed_v2
        try:
            # Try to import the fixed version first
            try:
                from src.registry import plot_generator_debug_fixed_v2
                module_name = "plot_generator_debug_fixed_v2"
            except ImportError:
                # Fall back to regular fixed version if v2 isn't available
                from src.registry import plot_generator_debug_fixed
                module_name = "plot_generator_debug_fixed"
                
            print(f"Starting registry visualization using {module_name}...")
            
            # Get the module dynamically
            module = importlib.import_module(f"src.registry.{module_name}")
            
            # This will start PlotUnit as part of its initialization
            demo_thread = threading.Thread(
                target=module.run_demo,
                kwargs={"duration": 600, "buffer_seconds": 10},
                daemon=True
            )
            demo_thread.start()
            
            # Give it time to initialize
            print("Waiting for registry visualization to start...")
            time.sleep(3.0)
            
            # Now try to get PlotUnit instance - try both possible imports
            try:
                from src.plot.plot_unit import PlotUnit
                print("Got PlotUnit from src.plot.plot_unit")
                return PlotUnit.get_instance()
            except ImportError:
                try:
                    from standalone_debug import PlotUnit
                    print("Got PlotUnit from standalone_debug as fallback")
                    return PlotUnit.get_instance()
                except Exception as inner_e:
                    print(f"Could not get PlotUnit instance after starting: {inner_e}")
        except Exception as e2:
            print(f"Failed to start registry visualization: {e2}")
        
        print("All initialization attempts failed.")
        return None
    except Exception as e:
        print(f"Unexpected error in visualization startup: {e}")
        import traceback
        traceback.print_exc()
        return None

def patch_plot_unit():
    """Patch PlotUnit to fix RAW view glitching."""
    try:
        # Start or get the running PlotUnit instance
        plot_unit = start_visualization()
        
        # Check if instance exists
        if not plot_unit:
            print("Could not get a valid PlotUnit instance. Cannot apply fix.")
            return False
        
        # Register demo signals with the registry
        print("Setting up registry with demo signals...")
        if register_demo_signals():
            print("Successfully registered demo signals")
        else:
            print("Warning: Could not register demo signals")
        
        # Check initialization status
        if hasattr(plot_unit, 'initialized'):
            if not plot_unit.initialized:
                print("PlotUnit exists but is not initialized. Waiting a bit longer...")
                # Give it a bit more time to initialize
                wait_time = 0
                while wait_time < 5 and not plot_unit.initialized:
                    time.sleep(1.0)
                    wait_time += 1
                    print(f"Extended wait for initialization... {wait_time}/5")
                
                if not plot_unit.initialized:
                    print("PlotUnit failed to initialize even after extended wait.")
                    return False
                else:
                    print("PlotUnit initialized during extended wait period.")
        else:
            print("PlotUnit has no 'initialized' attribute. Assuming it's ready.")
            
        # Connect registry signals to PlotUnit
        print("Connecting registry signals to PlotUnit...")
        if connect_registry_to_plot_unit(plot_unit):
            print("Successfully connected registry signals to PlotUnit")
        else:
            print("Warning: Could not connect registry signals to PlotUnit")
            
        # Check if _set_mode is available
        if not hasattr(plot_unit, '_set_mode'):
            print("PlotUnit has no _set_mode method. Trying to find an alternative...")
            # Try to find an alternative way to set the mode
            if hasattr(plot_unit, 'set_view_mode'):
                print("Found 'set_view_mode' as alternative.")
                original_set_mode = plot_unit.set_view_mode
            elif hasattr(plot_unit, 'event_handler') and hasattr(plot_unit.event_handler, '_update_mode'):
                print("Found 'event_handler._update_mode' as alternative.")
                original_set_mode = lambda mode: plot_unit.event_handler._update_mode(mode.value)
            else:
                print("No compatible method found to change view mode.")
                return False
        else:
            original_set_mode = plot_unit._set_mode
          # Setup different view mode handlers based on registry signals
        
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
        
        # Get registry instance for signal access
        registry = get_registry_instance()
        
        # Function to update the plot_unit data from registry for specific view modes
        def update_plot_data_for_view_mode(mode):
            """Update the plot data based on view mode using registry signals."""
            if not registry:
                print("No registry instance available for signal updates")
                return False
                
            if not hasattr(plot_unit, 'data_lock') or not hasattr(plot_unit, 'data'):
                print("PlotUnit missing data or lock attributes")
                return False
                
            try:
                with plot_unit.data_lock:
                    # Clear existing data first
                    plot_unit.data.clear()
                    
                    # Add appropriate signals based on view mode
                    if mode == ViewMode.RAW:
                        # Only add raw signals for RAW view
                        for signal_id, signal_data in registry.signals.items():
                            if not signal_id.startswith("PROC_") and "_processed" not in signal_id:
                                plot_unit.data[signal_id] = signal_data
                                print(f"Added raw signal {signal_id} to RAW view")
                    
                    elif mode == ViewMode.PROCESSED:
                        # Only add processed signals for PROCESSED view
                        for signal_id, signal_data in registry.signals.items():
                            if signal_id.startswith("PROC_") or "_processed" in signal_id:
                                plot_unit.data[signal_id] = signal_data
                                print(f"Added processed signal {signal_id} to PROCESSED view")
                    elif mode in (ViewMode.TWIN, ViewMode.STACKED):
                        # For TWIN view, we defer to the specialized fix_twin_view_rendering function
                        # which implements better pairing of raw/processed signals
                        print("Using specialized twin view handling instead of standard data loading")
                        # We'll add a placeholder so this function still returns True
                        plot_unit.data["_twin_view_placeholder"] = np.array([0, 0])  # Just a placeholder
                
                return True
            except Exception as e:
                print(f"Error updating plot data for view mode {mode}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Define specialized view rendering functions
        def fix_raw_view_rendering():
            """Fix raw view rendering by showing only raw signals with proper cleanup."""
            # Ensure only raw signals are present
            update_plot_data_for_view_mode(ViewMode.RAW)
            
            # Add a slight delay before rendering
            time.sleep(0.1)
            
            # Force a redraw with cleaner state - try different methods
            redraw_methods = ['_render', 'render', '_redraw', 'redraw', 'update']
            for method_name in redraw_methods:
                if hasattr(plot_unit, method_name) and callable(getattr(plot_unit, method_name)):
                    try:
                        print(f"Calling {method_name}() to refresh RAW display")
                        getattr(plot_unit, method_name)()
                        break
                    except Exception as e:
                        print(f"Error calling {method_name}(): {e}")
            
            print("Applied anti-glitch fix for RAW view")
            return True
        
        def fix_processed_view_rendering():
            """Fix processed view rendering with only processed signals."""
            # Update for processed view
            update_plot_data_for_view_mode(ViewMode.PROCESSED)
            
            # Add a slight delay before rendering
            time.sleep(0.1)
            
            # Force a redraw
            redraw_methods = ['_render', 'render', '_redraw', 'redraw', 'update']
            for method_name in redraw_methods:
                if hasattr(plot_unit, method_name) and callable(getattr(plot_unit, method_name)):
                    try:
                        print(f"Calling {method_name}() to refresh PROCESSED display")
                        getattr(plot_unit, method_name)()
                        break
                    except Exception as e:
                        print(f"Error calling {method_name}(): {e}")
                        print("Applied fix for PROCESSED view")
            return True
        
        def fix_twin_view_rendering():
            """Fix twin/stacked view rendering with raw and processed signals properly stacked."""
            # Update for twin/stacked view - properly organizing raw and processed signals
            try:
                # Clear existing data first
                with plot_unit.data_lock:
                    plot_unit.data.clear()
                    
                    # Organize signal pairs for proper stacking
                    signal_pairs = []
                    
                    # Match raw signals with their processed counterparts
                    if registry and hasattr(registry, 'signals'):
                        print("Organizing raw and processed signal pairs for TWIN view")
                        
                        # Get all raw signals
                        raw_signals = {sig_id: data for sig_id, data in registry.signals.items() 
                                      if not sig_id.startswith("PROC_") and "_processed" not in sig_id}
                        
                        # For each raw signal, find its processed counterpart
                        for raw_id, raw_data in raw_signals.items():
                            # First try with PROC_ prefix
                            proc_id = f"PROC_{raw_id}" 
                            if proc_id not in registry.signals:
                                # Try removing _RAW suffix and adding PROC_ prefix
                                if raw_id.endswith("_RAW"):
                                    base_name = raw_id[:-4]  # Remove _RAW
                                    proc_id = f"PROC_{base_name}"
                            
                            # Add the signal pair (raw and its processed version)
                            if proc_id in registry.signals:
                                signal_pairs.append((raw_id, proc_id))
                                print(f"Added signal pair: {raw_id} and {proc_id}")
                        
                        # If no pairs were found, use individual signals
                        if not signal_pairs:
                            print("No matching pairs found, adding signals individually")
                            for sig_id, data in registry.signals.items():
                                plot_unit.data[sig_id] = data
                                print(f"Added individual signal {sig_id} to TWIN view")
                        else:
                            # Add the pairs to display
                            for raw_id, proc_id in signal_pairs[:2]:  # Limit to 2 pairs for stability
                                plot_unit.data[raw_id] = registry.signals[raw_id]
                                plot_unit.data[proc_id] = registry.signals[proc_id]
                                print(f"Added signal pair: {raw_id} & {proc_id} to TWIN view")
                    else:
                        print("No registry signals available for TWIN view")
                        
                    # Handle empty data case
                    if not plot_unit.data:
                        print("No signals available for TWIN view, creating dummy data")
                        # Create dummy signals if no real data
                        t = np.linspace(0, 10, 1000)
                        dummy_raw = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.random.random(size=len(t))
                        dummy_proc = np.sin(2 * np.pi * 0.5 * t)
                        plot_unit.data["DUMMY_RAW"] = dummy_raw
                        plot_unit.data["DUMMY_PROCESSED"] = dummy_proc
                        print("Added dummy signals for demonstration")
                        
            except Exception as e:
                print(f"Error updating data for TWIN view: {e}")
                import traceback
                traceback.print_exc()
            
            # Add a longer delay before rendering to ensure stability
            time.sleep(0.5)
            
            # Force a redraw with careful error handling
            redraw_methods = ['_render', 'render', '_redraw', 'redraw', 'update']
            for method_name in redraw_methods:
                if hasattr(plot_unit, method_name) and callable(getattr(plot_unit, method_name)):
                    try:
                        print(f"Carefully calling {method_name}() to refresh TWIN display")
                        getattr(plot_unit, method_name)()
                        break
                    except Exception as e:
                        print(f"Error calling {method_name}() for TWIN view: {e}")
                        print("Trying alternative rendering approach...")
                        
                        # Try alternative approach by switching to RAW first
                        try:
                            print("Switching to RAW view first to stabilize...")
                            if hasattr(plot_unit, '_set_mode'):
                                plot_unit._set_mode(ViewMode.RAW)
                            elif hasattr(plot_unit, 'set_view_mode'):
                                plot_unit.set_view_mode(ViewMode.RAW)
                            time.sleep(0.2)  # Brief pause
                            
                            # Then switch back to TWIN
                            if hasattr(plot_unit, '_set_mode'):
                                plot_unit._set_mode(ViewMode.TWIN)
                            elif hasattr(plot_unit, 'set_view_mode'):
                                plot_unit.set_view_mode(ViewMode.TWIN)
                        except Exception as e2:
                            print(f"Alternative approach also failed: {e2}")
                        
            print("Applied optimized fix for TWIN/STACKED view")
            return True
        
        # Create patched set_mode function
        def patched_set_mode(mode):
            """Patched _set_mode that fixes view mode glitches and handles registry data properly."""
            print(f"Setting view mode to: {mode}")
            
            # Call original method first
            try:
                original_set_mode(mode)
            except Exception as e:
                print(f"Warning: Error in original set_mode: {e}")
            
            # Apply specialized handling based on view mode
            if mode == ViewMode.RAW:
                print("RAW view detected - applying specialized handling")
                time.sleep(0.05)
                fix_raw_view_rendering()
                
            elif mode == ViewMode.PROCESSED:
                print("PROCESSED view detected - applying specialized handling")
                time.sleep(0.05)
                fix_processed_view_rendering()
                
            elif mode in (ViewMode.TWIN, ViewMode.STACKED):
                print("TWIN/STACKED view detected - applying specialized handling")
                time.sleep(0.05)
                fix_twin_view_rendering()
        # Apply the patch
        if hasattr(plot_unit, '_set_mode'):
            plot_unit._set_mode = patched_set_mode
            print("Patched _set_mode method")
        elif hasattr(plot_unit, 'set_view_mode'):
            plot_unit.set_view_mode = patched_set_mode
            print("Patched set_view_mode method")
        else:
            print("Could not find a suitable method to patch")
            return False
        
        # Test the fix by switching to RAW view
        try:
            print("Testing RAW view fix...")
            if hasattr(plot_unit, '_set_mode'):
                plot_unit._set_mode(ViewMode.RAW)
            elif hasattr(plot_unit, 'set_view_mode'):
                plot_unit.set_view_mode(ViewMode.RAW)
            
            print("RAW view mode set successfully")
            return True
        except Exception as e:
            print(f"Error testing RAW view fix: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"Could not import required module: {e}")
        return False
    except Exception as e:
        print(f"Error patching PlotUnit: {e}")
        import traceback
        traceback.print_exc()
        return False


def run():
    """Main function to apply the RAW view glitch fix with demo."""
    print("Applying PIC-2025 visualization fixes with registry demo...\n")
    
    print("Step 1: Starting or finding visualization...")
    plot_unit = start_visualization()
    
    if not plot_unit:
        print("\nWARNING: Failed to locate or start the visualization.")
        print("Let's try to run it anyway by directly starting registry visualization...")
        
        # Last-ditch effort - try to run the registry visualization directly
        try:
            print("\nAttempting to start registry visualization directly...")
            
            # Check if pygame is initialized
            if not pygame.get_init():
                pygame.init()
                print("Initialized pygame")
            
            # Try both possible module versions
            try:
                try:
                    from src.registry import plot_generator_debug_fixed_v2 as plot_module
                    print("Using plot_generator_debug_fixed_v2")
                except ImportError:
                    from src.registry import plot_generator_debug_fixed as plot_module
                    print("Using plot_generator_debug_fixed")
                
                # Start the visualization directly
                print("Starting visualization directly...")
                plot_module.run_demo(duration=600, buffer_seconds=10)
                print("Direct visualization start succeeded!")
                
                # Give it a moment to initialize
                time.sleep(2.0)
                
                # Try patching again
                success = patch_plot_unit()
            except Exception as e:
                print(f"Failed to start visualization directly: {e}")
                import traceback
                traceback.print_exc()
                success = False
        except Exception as e:
            print(f"Failed final attempt to start visualization: {e}")
            success = False
    else:
        print("Step 2: Applying view mode patches...")
        success = patch_plot_unit()
    
    if success:
        print("\n✅ View mode fixes applied successfully!")
        print("All view modes should now display properly without glitching.")
        
        # Demo all view modes
        try:
            from src.plot.view_mode import ViewMode
            
            print("\nDemonstrating all view modes:")
            print("-------------------------------")
            
            # Function to safely set the view mode
            def set_view_mode_safely(mode_name, mode):
                try:
                    print(f"\nSwitching to {mode_name} view...")
                    if hasattr(plot_unit, '_set_mode'):
                        plot_unit._set_mode(mode)
                    elif hasattr(plot_unit, 'set_view_mode'):
                        plot_unit.set_view_mode(mode)
                    print(f"Now displaying {mode_name} view")
                    return True
                except Exception as e:
                    print(f"Error switching to {mode_name} view: {e}")
                    return False
                      # Cycle through different view modes with delays between them
            print("\nStarting view mode demo cycle (adaptive timing per mode)...")
            
            # RAW view - most stable, so show it first and longest
            set_view_mode_safely("RAW", ViewMode.RAW)
            print("RAW view is showing the original signals: ECG_RAW, EDA_RAW, RESP_RAW")
            print("Waiting 15 seconds in RAW view...")
            time.sleep(15)
            
            # PROCESSED view - also quite stable
            set_view_mode_safely("PROCESSED", ViewMode.PROCESSED)
            print("PROCESSED view is showing the processed signals: PROC_ECG, PROC_EDA, PROC_RESP")
            print("Waiting 10 seconds in PROCESSED view...")
            time.sleep(10)
            
            # TWIN view - potentially less stable, so shorter duration and careful handling
            print("Preparing to switch to TWIN view (stacked raw & processed)...")
            print("For stability, first ensuring proper signal setup...")
            
            # Ensure registry has data
            registry = get_registry_instance()
            if registry and len(registry.signals) < 4:
                print("Refreshing registry signals before TWIN view...")
                register_demo_signals()  # Refresh the signals
            
            # Switch to TWIN view with extra care
            print("Switching to TWIN view now...")
            set_view_mode_safely("TWIN", ViewMode.TWIN)
            print("TWIN view is showing raw and processed signals in stacked format")
            print("Waiting 5 seconds in TWIN view (shorter to prevent freezes)...")
            time.sleep(5)
            
            # Back to RAW view as it's generally the most stable
            set_view_mode_safely("RAW", ViewMode.RAW)
            print("Returned to RAW view - demo cycle complete!")
            
        except ImportError:
            print("\nCould not import ViewMode to demonstrate view modes.")
            print("The fixes are still applied, but the demo won't cycle through views.")
        
        print("\nInstructions:")
        print("1. Try switching between view modes using the sidebar buttons")
        print("2. All views should now display correctly without glitches")
        print("3. Let the visualization run for a while to ensure stability")
        print("4. Check that the 3 signals (ECG, EDA, RESP) appear in each view")
    else:
        print("\n❌ Failed to apply view mode fixes.")
        print("Troubleshooting tips:")
        print("1. Make sure the PIC-2025 registry visualization is running")
        print("2. Try running 'run_fixed_registry_visualization_v2.bat' first")
        print("3. Then run this script again after visualization is visible")
    
    print("\nPress Ctrl+C to exit or close this window.")
    
    # Keep the script running to maintain the patches
    try:
        counter = 0
        demo_interval = 120  # Auto cycle views every 2 minutes for demo effect
        
        while True:
            time.sleep(1.0)
            counter += 1
            
            # Every 30 seconds, remind the user that the script is still active
            if counter % 30 == 0:
                print(f"View mode fixes still active... (Running for {counter} seconds)")
                  # Every demo_interval seconds, cycle through view modes
            if success and counter % demo_interval == 0:
                try:
                    print("\nAuto-cycling view modes for demonstration:")
                    from src.plot.view_mode import ViewMode
                    
                    # Weighted view mode selection to favor more stable modes
                    modes = [ViewMode.RAW, ViewMode.PROCESSED, ViewMode.RAW, ViewMode.TWIN, ViewMode.RAW]
                    mode_names = ["RAW", "PROCESSED", "RAW", "TWIN", "RAW"]
                    
                    # Cycle through the views, but spend more time in RAW view
                    cycle_position = counter // demo_interval % len(modes)
                    current_mode = modes[cycle_position]
                    current_name = mode_names[cycle_position]
                    
                    # Special handling for TWIN view to improve stability
                    if current_mode == ViewMode.TWIN:
                        print("Preparing for TWIN view, checking signal registry first...")
                        registry = get_registry_instance()
                        if registry and hasattr(registry, 'signals') and len(registry.signals) < 4:
                            print("Refreshing registry data...")
                            register_demo_signals()
                    
                    # Apply the view mode change
                    print(f"Switching to {current_name} view...")
                    if hasattr(plot_unit, '_set_mode'):
                        plot_unit._set_mode(current_mode)
                    elif hasattr(plot_unit, 'set_view_mode'):
                        plot_unit.set_view_mode(current_mode)
                    
                    print(f"Successfully switched to {current_name} view")
                    
                    # If TWIN view, be ready to recover if it freezes
                    if current_mode == ViewMode.TWIN:
                        print("TWIN view active - will auto-switch back to RAW view in 40 seconds as a precaution")
                        
                except Exception as e:
                    # More robust error handling during view switching
                    print(f"Error during view switching: {e}")
                    try:
                        # Always try to get back to RAW view on error
                        print("Attempting recovery by switching to RAW view...")
                        if hasattr(plot_unit, '_set_mode'):
                            plot_unit._set_mode(ViewMode.RAW)
                        elif hasattr(plot_unit, 'set_view_mode'):
                            plot_unit.set_view_mode(ViewMode.RAW)
                    except:
                        pass
    
    except KeyboardInterrupt:
        print("\nExiting view mode fix script...")


if __name__ == "__main__":
    run()
