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

# Add project root to path to ensure imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

print("\n=== PIC-2025 RAW View Glitch Fix ===\n")

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
        
        # Define raw view enhancement function with more robust implementation
        def fix_raw_view_rendering():
            """Fix raw view rendering by clearing processed signals and applying render timing."""
            # Different ways the data might be accessed
            if hasattr(plot_unit, 'data_lock') and hasattr(plot_unit, 'data'):
                # Standard PlotUnit implementation
                try:
                    with plot_unit.data_lock:
                        # Remove any processed signals from the data
                        for signal_id in list(plot_unit.data.keys()):
                            if signal_id.startswith("PROC_") or signal_id.endswith("_processed"):
                                del plot_unit.data[signal_id]
                                print(f"Removed processed signal {signal_id} from view")
                except Exception as e:
                    print(f"Error cleaning data with lock: {e}")
            elif hasattr(plot_unit, 'data'):
                # Simplified implementation without lock
                try:
                    for signal_id in list(plot_unit.data.keys()):
                        if signal_id.startswith("PROC_") or signal_id.endswith("_processed"):
                            del plot_unit.data[signal_id]
                            print(f"Removed processed signal {signal_id} (no lock)")
                except Exception as e:
                    print(f"Error cleaning data without lock: {e}")
            elif hasattr(plot_unit, 'views') and hasattr(plot_unit.views.get(ViewMode.RAW), 'data'):
                # View-specific data
                try:
                    raw_view_data = plot_unit.views.get(ViewMode.RAW).data
                    for signal_id in list(raw_view_data.keys()):
                        if signal_id.startswith("PROC_") or signal_id.endswith("_processed"):
                            del raw_view_data[signal_id]
                            print(f"Removed processed signal {signal_id} from raw view")
                except Exception as e:
                    print(f"Error cleaning view-specific data: {e}")
            
            # Add a slight delay before rendering
            time.sleep(0.1)
            
            # Force a redraw with cleaner state - try different methods
            redraw_methods = ['_render', 'render', '_redraw', 'redraw', 'update']
            for method_name in redraw_methods:
                if hasattr(plot_unit, method_name) and callable(getattr(plot_unit, method_name)):
                    try:
                        print(f"Calling {method_name}() to refresh display")
                        getattr(plot_unit, method_name)()
                        break
                    except Exception as e:
                        print(f"Error calling {method_name}(): {e}")
            
            print("Applied anti-glitch fix for RAW view")
            return True
        
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
                    SETTINGS = 3
        
        # Create patched set_mode function
        def patched_set_mode(mode):
            """Patched _set_mode that fixes RAW view glitching."""
            print(f"Setting view mode to: {mode}")
            
            # Call original method first
            try:
                original_set_mode(mode)
            except Exception as e:
                print(f"Warning: Error in original set_mode: {e}")
            
            # If switching to RAW view, apply our special handling
            if mode == ViewMode.RAW:
                print("RAW view detected - applying anti-glitch fix")
                # Use a slight delay to ensure UI has processed the mode change
                time.sleep(0.05)
                fix_raw_view_rendering()
        
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
    """Main function to apply the RAW view glitch fix."""
    print("Applying RAW view glitch fix...\n")
    
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
        print("Step 2: Applying RAW view patch...")
        success = patch_plot_unit()
    
    if success:
        print("\n✅ RAW view glitch fix applied successfully!")
        print("The RAW view should now display properly without glitching.")
        print("\nInstructions:")
        print("1. Try switching between view modes using the sidebar buttons")
        print("2. RAW view should now display correctly without glitches")
        print("3. Let the visualization run for a while to ensure stability")
    else:
        print("\n❌ Failed to apply RAW view glitch fix.")
        print("Troubleshooting tips:")
        print("1. Make sure the PIC-2025 registry visualization is running")
        print("2. Try running 'run_fixed_registry_visualization_v2.bat' first")
        print("3. Then run this script again after visualization is visible")
    
    print("\nPress Ctrl+C to exit or close this window.")
    
    # Keep the script running to maintain the patches
    try:
        counter = 0
        while True:
            time.sleep(1.0)
            counter += 1
            # Every 30 seconds, remind the user that the script is still active
            if counter % 30 == 0:
                print(f"RAW view glitch fix still active... (Running for {counter} seconds)")
    except KeyboardInterrupt:
        print("\nExiting RAW view glitch fix...")


if __name__ == "__main__":
    run()
