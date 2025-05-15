#!/usr/bin/env python
"""
Plot Generator Debug - Registry Integration

This module provides an adapter between the PIC-2025 registry system and the
standalone debug visualization application. It generates synthetic signals,
registers them in the SignalRegistry, connects them through PlotRegistry,
and updates the signals in real time for visualization in PlotUnit.
"""

import os
import sys
import time
import threading
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional

# Make sure we can import from the project structure
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

try:
    # Import registry components
    from src.registry.signal_registry import SignalRegistry
    from src.registry.plot_registry import PlotRegistry
    from src.registry.plot_registry_integration import PlotRegistryIntegration

    # Import constants if available
    try:
        from src.plot.constants import (
            ECG_SIGNAL_COLOR, EDA_SIGNAL_COLOR, RAW_SIGNAL_COLOR,
            PROCESSED_SIGNAL_COLOR
        )
    except ImportError:
        # Define fallback colors
        ECG_SIGNAL_COLOR = (255, 0, 0)  # Red
        EDA_SIGNAL_COLOR = (220, 120, 0)  # Orange
        RAW_SIGNAL_COLOR = (220, 180, 0)  # Yellow
        PROCESSED_SIGNAL_COLOR = (0, 180, 220)  # Blue
        
    print("SUCCESS: Imported registry components")
except ImportError as e:
    print(f"ERROR: Could not import registry components: {e}")
    raise


class RegistrySignalGenerator:
    """
    Generator for synthetic signals to be registered with the SignalRegistry.
    Uses the proper registry architecture for handling signals.
    """    def __init__(self):
        # Get instances of registry components
        self.signal_registry = SignalRegistry.get_instance()
        self.plot_registry = PlotRegistry.get_instance()
        
        # Configure synthetic signal properties
        self.sampling_rate = 1000  # Hz
        self.signals = {}  # Store active signal IDs and their metadata
        
        # Signal capacity limits (system constraint)
        self.raw_signal_limit = 3
        self.processed_signal_limit = 3
        self.raw_signals_count = 0
        self.processed_signals_count = 0
        
        # Set up threading for continuous signal generation
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()
        
        print("RegistrySignalGenerator initialized with limits: 3 raw, 3 processed signals")

    def start(self):
        """Start the background signal generation thread."""
        if self.running:
            print("Signal generator already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._generate_signals)
        self.thread.daemon = True
        self.thread.start()
        
        print("Signal generator started")
        
    def stop(self):
        """Stop the background signal generation thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("Signal generator stopped")

    def _generate_signals(self):
        """Background thread function to continuously generate signals."""
        print("Signal generation thread started")
        
        # Create initial signals
        self._create_ecg_signal()
        self._create_eda_signal()
        self._create_synthetic_signal("SINE", self._generate_sine)
        
        # Generate timestamp for the x-axis
        start_time = time.time()
        sample_index = 0
        
        # Main loop for continuous signal generation
        while self.running:
            try:
                # Update each active signal
                with self.data_lock:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Generate new data points for each signal
                    for signal_id, metadata in self.signals.items():
                        # Only update if signal exists in registry
                        if self.signal_registry.get_signal(signal_id) is not None:
                            # Get the generation function for this signal
                            generate_func = metadata.get('generator_func')
                            if generate_func:
                                # Generate new data point
                                new_value = generate_func(elapsed, sample_index)
                                
                                # Update signal in registry with new value
                                signal_data = self.signal_registry.get_signal(signal_id)
                                
                                # Ensure signal_data is a numpy array
                                if not isinstance(signal_data, np.ndarray):
                                    signal_data = np.array([])
                                
                                # Append new data
                                updated_data = np.append(signal_data, new_value)
                                
                                # Trim if growing too large (keep last 1000 samples)
                                if len(updated_data) > 1000:
                                    updated_data = updated_data[-1000:]
                                    
                                # Update the signal in registry
                                self.signal_registry.register_signal(
                                    signal_id=signal_id,
                                    signal_data=updated_data,
                                    metadata=metadata
                                )
                
                # Increment sample index
                sample_index += 1
                
                # Wait for next sample (respect sampling rate)
                time.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                print(f"Error in signal generator thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)  # Prevent busy-waiting in case of repeated errors

    def _create_ecg_signal(self):
        """Create an ECG signal in the registry."""
        signal_id = "ECG"
        initial_data = np.array([])
        
        # Define metadata
        metadata = {
            'name': 'ECG Signal',
            'color': ECG_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_ecg
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        
        print(f"Created ECG signal with ID: {signal_id}")
        return signal_id

    def _create_eda_signal(self):
        """Create an EDA signal in the registry."""
        signal_id = "EDA"
        initial_data = np.array([])
        
        # Define metadata
        metadata = {
            'name': 'EDA Signal',
            'color': EDA_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_eda
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        
        print(f"Created EDA signal with ID: {signal_id}")
        return signal_id
        
    def _create_synthetic_signal(self, name, generator_func):
        """Create a custom synthetic signal in the registry."""
        signal_id = f"SYN_{name}"
        initial_data = np.array([])
        
        # Define metadata
        metadata = {
            'name': f'Synthetic {name}',
            'color': PROCESSED_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': generator_func
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        
        print(f"Created synthetic signal with ID: {signal_id}")
        return signal_id

    def _generate_ecg(self, elapsed_time, sample_index):
        """Generate a synthetic ECG signal value."""
        # Simple ECG-like pattern (periodic with sharp peaks)
        t = elapsed_time
        base_freq = 1.2  # Hz (heart rate ~72 BPM)
        
        # Basic sine
        value = np.sin(2 * np.pi * base_freq * t)
        
        # Add sharp peaks
        spike = np.exp(-80 * ((t * base_freq) % 1 - 0.2) ** 2)
        value = value * 0.3 + spike * 0.7
        
        # Add noise and scaling
        value = value + np.random.normal(0, 0.05)
        
        return value

    def _generate_eda(self, elapsed_time, sample_index):
        """Generate a synthetic EDA (skin conductance) signal value."""
        t = elapsed_time
        
        # Slower oscillation with trend
        slow_component = 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
        
        # Add stochastic component (random walk)
        random_walk = np.sin(0.5 * t) * np.random.normal(0, 0.02)
        
        # Combine components
        value = slow_component + random_walk
        
        return value

    def _generate_sine(self, elapsed_time, sample_index):
        """Generate a simple sine wave."""
        t = elapsed_time
        frequency = 0.5  # Hz
        
        value = np.sin(2 * np.pi * frequency * t)
        value = value * 0.8 + np.random.normal(0, 0.1)
        
        return value


class PlotUnitRegistryAdapter:
    """
    Adapter that connects the PlotUnit visualization with the PlotRegistry.
    This serves as a "connector" between the registry system and the visualization.
    """
    
    def __init__(self, plot_unit=None):
        """
        Initialize the adapter.
        
        Args:
            plot_unit: The PlotUnit instance to connect to (will get instance if None)
        """
        try:
            # Get registry instances
            self.plot_registry = PlotRegistry.get_instance()
            self.integration = PlotRegistryIntegration.get_instance()
            
            # Get or find the PlotUnit instance
            if plot_unit is None:
                # Import PlotUnit dynamically since it might not be in a consistent location
                try:
                    from standalone_debug import PlotUnit
                    self.plot_unit = PlotUnit.get_instance()
                except ImportError:
                    try:
                        from src.plot.plot_unit import PlotUnit
                        self.plot_unit = PlotUnit.get_instance()
                    except ImportError:
                        print("ERROR: Could not import PlotUnit. Connect manually.")
                        self.plot_unit = None
            else:
                self.plot_unit = plot_unit
                
            # Connection setup
            self.running = False
            self.thread = None
            self.last_update_time = time.time()
            self.blink_state = False
            self.blink_timer = 0
            self.signal_count = 0
            
            print("PlotUnitRegistryAdapter initialized")
            
        except ImportError as e:
            print(f"ERROR: Could not import required components: {e}")
            raise
    
    def connect(self):
        """Connect the PlotUnit to the PlotRegistry and start monitoring."""
        if self.running:
            print("Adapter already running")
            return
        
        if self.plot_unit is None:
            print("ERROR: No PlotUnit instance available")
            return
        
        # Start the monitoring thread
        self.running = True
        self.thread = threading.Thread(target=self._monitor_registry)
        self.thread.daemon = True
        self.thread.start()
        
        print("PlotUnitRegistryAdapter started - connecting PlotUnit to registry")
    def disconnect(self):
        """Disconnect the adapter and stop monitoring."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("PlotUnitRegistryAdapter stopped")
        
    def _monitor_registry(self):
        """Monitor the registry for changes and update the PlotUnit."""
        print("Registry monitoring thread started")
        
        while self.running:
            try:
                # Get all signals from the registry
                with self.integration._registry_lock:
                    signals_to_render = {}
                    for signal_id in self.integration.registered_signals:
                        signal_data = self.plot_registry.get_signal(signal_id)
                        if signal_data is not None:
                            metadata = self.plot_registry.get_signal_metadata(signal_id)
                            signals_to_render[signal_id] = {
                                'data': signal_data,
                                'metadata': metadata
                            }
                
                # Store the signal count
                self.signal_count = len(signals_to_render)
                
                # Update the "connected_nodes" setting to reflect the number of signals in registry
                if hasattr(self.plot_unit, 'settings'):
                    self.plot_unit.settings['connected_nodes'] = self.signal_count
                
                # Handle blinking dot logic - blink at 1Hz (on for 0.5s, off for 0.5s)
                current_time = time.time()
                if current_time - self.blink_timer >= 0.5:  # Toggle every 0.5 seconds
                    self.blink_state = not self.blink_state
                    self.blink_timer = current_time
                    
                    # Set the registry connection status for the sidebar
                    if hasattr(self.plot_unit, 'settings'):
                        if self.signal_count > 0:
                            # If we have signals, set registry_connected based on blink state
                            self.plot_unit.settings['registry_connected'] = self.blink_state
                        else:
                            # If no signals, make sure the indicator is off
                            self.plot_unit.settings['registry_connected'] = False
                
                # Update the PlotUnit with the signals
                if signals_to_render and hasattr(self.plot_unit, 'data_lock') and hasattr(self.plot_unit, 'data'):
                    with self.plot_unit.data_lock:
                        for signal_id, signal_info in signals_to_render.items():
                            # Update the PlotUnit data dictionary
                            self.plot_unit.data[signal_id] = {
                                'values': signal_info['data'],
                                'metadata': signal_info['metadata'],
                                'last_update': datetime.datetime.now().strftime("%H:%M:%S"),
                                'color': signal_info['metadata'].get('color', (255, 255, 255))
                            }
                    
                    # Update last update time and build signal_times dictionary for status bar
                    self.last_update_time = time.time()
                    
                    # Create a signal_times dictionary for the status bar
                    signal_times = {}
                    for signal_id in signals_to_render:
                        signal_times[signal_id] = self.last_update_time
                    
                    # Update status bar if the method exists
                    if hasattr(self.plot_unit, 'status_bar') and hasattr(self.plot_unit.status_bar, 'draw'):
                        latency = 0.01  # Default latency value
                        self.plot_unit.status_bar.draw(
                            self.plot_unit.fps,
                            latency,
                            signal_times
                        )
                
                # Wait a short time before checking again
                time.sleep(0.05)  # 50ms update rate
                
            except Exception as e:
                print(f"Error in registry monitoring thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)


def run_demo(duration=3000):    """
    Run a demonstration of the registry integration with the PlotUnit visualization.
    
    Args:
        duration: How long to run the demo for, in seconds
    """
    print("\n=== Starting PIC-2025 Registry Visualization Demo ===\n")
    
    # Import UI enhancements for registry connection
    try:
        import src.plot.ui.sidebar_registry_enhancement
        print("Sidebar enhanced with registry connection indicator")
    except ImportError as e:
        print(f"Note: Sidebar registry indicator not available: {e}")
        
    try:
        import src.plot.view.settings_view_registry_enhancement
        print("Settings view enhanced with registry information")
    except ImportError as e:
        print(f"Note: Settings view registry information not available: {e}")
        
    try:
        import src.plot.ui.status_bar_registry_enhancement
        print("Status bar enhanced with registry signal information")
    except ImportError as e:
        print(f"Note: Status bar registry enhancement not available: {e}")
    
    # Step 1: Initialize and start the signal generator
    print("Initializing signal generator...")
    generator = RegistrySignalGenerator()
    generator.start()
    print("Signal generator started, registering signals...")
    time.sleep(1.0)  # Give time for initial signals to be registered
      # Step 2: Start the PlotUnit visualization
    try:
        print("Importing PlotUnit...")
        from standalone_debug import PlotUnit
        
        print("Starting PlotUnit visualization...")
        plot_unit = PlotUnit.get_instance()
        
        # Add registry_connected setting if it doesn't exist
        if hasattr(plot_unit, 'settings') and 'registry_connected' not in plot_unit.settings:
            plot_unit.settings['registry_connected'] = False
            
        plot_unit.start()
        
        # Wait for initialization
        wait_seconds = 0
        while not plot_unit.initialized and wait_seconds < 5:
            print(f"Waiting for PlotUnit initialization... ({wait_seconds+1}/5)")
            time.sleep(1.0)
            wait_seconds += 1
        
        if not plot_unit.initialized:
            print("ERROR: PlotUnit failed to initialize within timeout")
            generator.stop()
            return
    except ImportError as e:
        print(f"ERROR: Could not import PlotUnit: {e}")
        print("Please start PlotUnit visualization manually")
        plot_unit = None
    
    # Step 3: Connect PlotUnit to the registry
    print("Connecting PlotUnit to registry...")
    adapter = PlotUnitRegistryAdapter(plot_unit)
    adapter.connect()
    
    # Step 4: Run the demo for the specified duration
    print(f"\nDemo running for {duration} seconds. Press Ctrl+C to exit early.")
    try:
        # Switch to RAW view mode if possible
        if plot_unit and hasattr(plot_unit, '_set_mode'):
            from src.plot.view_mode import ViewMode
            plot_unit._set_mode(ViewMode.RAW)
        
        # Run for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            # Show periodic status updates
            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0:
                signal_count = len(generator.signals)
                print(f"Running demo... {elapsed}/{duration} seconds - {signal_count} signals active")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    # Step 5: Clean up
    print("\nDemo finished. Cleaning up...")
    adapter.disconnect()
    generator.stop()
    
    print("Demo complete. PlotUnit will continue running.")
    print("Press Ctrl+C in the PlotUnit window or terminal to exit completely.")


if __name__ == "__main__":
    run_demo(duration=600)  # Run for 1 minute by default
