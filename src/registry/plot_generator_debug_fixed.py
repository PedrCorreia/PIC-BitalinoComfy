#!/usr/bin/env python
"""
Plot Generator Debug - Registry Integration (Fixed Version)

This module provides an adapter between the PIC-2025 registry system and the
standalone debug visualization application. It generates synthetic signals,
registers them in the SignalRegistry, connects them through PlotRegistry,
and updates the signals in real time for visualization in PlotUnit.

This version implements 3 different signal generators and configurable data buffer.
"""

import os
import sys
import time
import threading
import numpy as np
import datetime
import traceback
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
    
    Provides 3 raw and 3 processed signal generators.
    """

    def __init__(self):
        # Get instances of registry components
        self.signal_registry = SignalRegistry.get_instance()
        self.plot_registry = PlotRegistry.get_instance()
        
        # Configure synthetic signal properties
        self.sampling_rate = 100  # Hz
        self.signals = {}  # Store active signal IDs and their metadata
        
        # Signal generators available (not strict limits, just preference)
        self.generators = {
            'raw': {
                'ECG': {'created': False, 'id': None},
                'EDA': {'created': False, 'id': None},
                'RAW_SINE': {'created': False, 'id': None}
            },
            'processed': {
                'WAVE1': {'created': False, 'id': None},
                'WAVE2': {'created': False, 'id': None},
                'WAVE3': {'created': False, 'id': None}
            }
        }
        
        # Data buffer settings (seconds of data to keep)
        self.buffer_seconds = 10  # Default to 10 seconds of data
        self.max_buffer_size = int(self.buffer_seconds * self.sampling_rate)
        
        # Set up threading for continuous signal generation
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()
        
        print(f"RegistrySignalGenerator initialized with {self.buffer_seconds}s buffer")

    def set_buffer_seconds(self, seconds):
        """Set the buffer size in seconds."""
        self.buffer_seconds = max(1, seconds)  # Minimum 1 second
        self.max_buffer_size = int(self.buffer_seconds * self.sampling_rate)
        print(f"Buffer size set to {self.buffer_seconds} seconds ({self.max_buffer_size} samples)")
        return self.buffer_seconds

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
        
        # Create initial signals (only once)
        self.create_ecg_signal()
        self.create_eda_signal()
        self.create_sine_signal(processed=True, name="WAVE1")
        
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
                                
                                # Trim to keep only the buffer size worth of data
                                if len(updated_data) > self.max_buffer_size:
                                    updated_data = updated_data[-self.max_buffer_size:]
                                    
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
                traceback.print_exc()
                time.sleep(1.0)  # Prevent busy-waiting in case of repeated errors

    def create_ecg_signal(self):
        """Create an ECG signal in the registry (raw signal)."""
        # Check if already created
        if self.generators['raw']['ECG']['created']:
            print(f"ECG signal already exists with ID: {self.generators['raw']['ECG']['id']}")
            return self.generators['raw']['ECG']['id']
            
        signal_id = "ECG"
        initial_data = np.array([])
        
        # Define metadata
        metadata = {
            'name': 'ECG Signal',
            'color': ECG_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_ecg,
            'type': 'raw'  # Mark as raw
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        self.generators['raw']['ECG']['created'] = True
        self.generators['raw']['ECG']['id'] = signal_id
        
        print(f"Created ECG signal with ID: {signal_id}")
        return signal_id

    def create_eda_signal(self):
        """Create an EDA signal in the registry (raw signal)."""
        # Check if already created
        if self.generators['raw']['EDA']['created']:
            print(f"EDA signal already exists with ID: {self.generators['raw']['EDA']['id']}")
            return self.generators['raw']['EDA']['id']
            
        signal_id = "EDA"
        initial_data = np.array([])
        
        # Define metadata
        metadata = {
            'name': 'EDA Signal',
            'color': EDA_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': self._generate_eda,
            'type': 'raw'  # Mark as raw
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        self.generators['raw']['EDA']['created'] = True
        self.generators['raw']['EDA']['id'] = signal_id
        
        print(f"Created EDA signal with ID: {signal_id}")
        return signal_id
        
    def create_sine_signal(self, processed=False, name=None):
        """
        Create a sine wave signal (either raw or processed).
        
        Args:
            processed: Whether this is a processed signal
            name: Optional name for the signal (from available generators)
        """
        # Default name if none provided
        if not name:
            name = "WAVE1" if processed else "RAW_SINE"
        
        # Check signal category
        category = 'processed' if processed else 'raw'
        
        # Check if the generator exists and if it's already created
        if name not in self.generators[category]:
            print(f"WARNING: No {name} generator defined in {category} category")
            return None
            
        if self.generators[category][name]['created']:
            print(f"Signal {name} already exists with ID: {self.generators[category][name]['id']}")
            return self.generators[category][name]['id']
        
        # Create ID with appropriate prefix for signal type
        prefix = "PROC" if processed else "RAW"
        signal_id = f"{prefix}_{name}"
        initial_data = np.array([])
        
        # Define generator function based on name
        if name in ["WAVE1", "WAVE2", "WAVE3", "RAW_SINE"]:
            # Different frequencies for different waves
            freq_map = {
                "WAVE1": 0.5,   # 0.5 Hz
                "WAVE2": 0.25,  # 0.25 Hz
                "WAVE3": 0.1,   # 0.1 Hz
                "RAW_SINE": 1.0 # 1.0 Hz
            }
            frequency = freq_map.get(name, 0.5)
            
            # Create custom generator with specific frequency
            def custom_generator(elapsed_time, sample_index):
                t = elapsed_time
                value = np.sin(2 * np.pi * frequency * t)
                
                # Add appropriate noise level
                noise_level = 0.05 if processed else 0.1
                value = value * 0.9 + np.random.normal(0, noise_level)
                return value
                
            generator_func = custom_generator
        else:
            # Default to simple sine wave
            generator_func = self._generate_sine
        
        # Define metadata
        metadata = {
            'name': f'{"Processed" if processed else "Raw"} {name}',
            'color': PROCESSED_SIGNAL_COLOR if processed else RAW_SIGNAL_COLOR,
            'sampling_rate': self.sampling_rate,
            'source': 'synthetic',
            'generator_func': generator_func,
            'type': 'processed' if processed else 'raw'
        }
        
        # Register with SignalRegistry first
        self.signal_registry.register_signal(signal_id, initial_data, metadata)
        
        # Register for visualization in PlotRegistry
        self.plot_registry.register_signal(signal_id, initial_data, metadata)
        
        # Store in active signals
        self.signals[signal_id] = metadata
        self.generators[category][name]['created'] = True
        self.generators[category][name]['id'] = signal_id
        
        print(f"Created {name} signal with ID: {signal_id}")
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
        
    def add_custom_signal(self, name, processed=False):
        """
        Add a custom signal to the registry (public API method).
        
        Args:
            name: Name for the signal
            processed: Whether this is a processed signal (True) or raw signal (False)
            
        Returns:
            str: Signal ID if successful, None if already exists
        """
        # Find an available generator in the appropriate category
        category = 'processed' if processed else 'raw'
        
        # If name matches an existing generator, use that one
        if name in self.generators[category]:
            return self.create_sine_signal(processed=processed, name=name)
            
        # Otherwise find the first available generator
        for gen_name, info in self.generators[category].items():
            if not info['created']:
                print(f"Using available generator {gen_name} for {name}")
                return self.create_sine_signal(processed=processed, name=gen_name)
                
        # If all generators are in use, let the user know
        print(f"All {category} signal generators are already in use")
        return None


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
            try:                # Get all signals from the registry
                try:
                    # Use PlotRegistryIntegration's class lock if available
                    lock_to_use = getattr(self.integration, '_registry_lock', 
                                        getattr(self.integration.__class__, '_lock', None))
                    
                    # If no lock is found, proceed without locking
                    if lock_to_use:
                        lock_to_use.acquire()
                    
                    signals_to_render = {}
                    # Access registered signals through the registry if not available in integration
                    signal_ids = getattr(self.integration, 'registered_signals', 
                                    self.plot_registry.get_all_signal_ids())
                    
                    for signal_id in signal_ids:
                        signal_data = self.plot_registry.get_signal(signal_id)
                        if signal_data is not None:
                            metadata = self.plot_registry.get_signal_metadata(signal_id)
                            signals_to_render[signal_id] = {
                                'data': signal_data,
                                'metadata': metadata
                            }
                finally:
                    # Release the lock if we acquired it
                    if lock_to_use and lock_to_use.locked():
                        lock_to_use.release()
                
                # Store the signal count
                self.signal_count = len(signals_to_render)
                
                # Track raw vs processed signal counts
                raw_count = 0
                processed_count = 0
                for _, info in signals_to_render.items():
                    metadata = info.get('metadata', {})
                    if metadata.get('type') == 'processed':
                        processed_count += 1
                    else:
                        raw_count += 1
                
                # Update the settings with signal counts
                if hasattr(self.plot_unit, 'settings'):
                    self.plot_unit.settings['connected_nodes'] = self.signal_count
                    self.plot_unit.settings['raw_signal_count'] = raw_count
                    self.plot_unit.settings['processed_signal_count'] = processed_count
                
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
                        try:
                            latency = 0.01  # Default latency value
                            self.plot_unit.status_bar.draw(
                                self.plot_unit.fps,
                                latency,
                                signal_times
                            )
                        except TypeError as e:
                            # Fallback if the enhanced draw method has issues
                            print(f"Note: Using fallback status bar update due to: {e}")
                            self.plot_unit.status_bar.draw(
                                self.plot_unit.fps,
                                latency
                            )
                
                # Wait a short time before checking again
                time.sleep(0.05)  # 50ms update rate
                
            except Exception as e:
                print(f"Error in registry monitoring thread: {e}")
                traceback.print_exc()
                time.sleep(1.0)


def run_demo(duration=600, buffer_seconds=10):
    """
    Run a demonstration of the registry integration with the PlotUnit visualization.
    
    Args:
        duration: How long to run the demo for, in seconds (default: 10 minutes)
        buffer_seconds: How many seconds of signal data to keep (default: 10 seconds)
    """
    print("\n=== Starting PIC-2025 Registry Visualization Demo (Fixed Version) ===\n")
    
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
    generator.set_buffer_seconds(buffer_seconds)
    generator.start()
    print("Signal generator started...")
    time.sleep(1.0)  # Give time for initial signals to be registered
    
    # Step 2: Start the PlotUnit visualization
    try:
        print("Importing PlotUnit...")
        from standalone_debug import PlotUnit
        
        print("Starting PlotUnit visualization...")
        plot_unit = PlotUnit.get_instance()
        
        # Add registry_connected setting if it doesn't exist
        if hasattr(plot_unit, 'settings'):
            if 'registry_connected' not in plot_unit.settings:
                plot_unit.settings['registry_connected'] = False
            if 'raw_signal_count' not in plot_unit.settings:
                plot_unit.settings['raw_signal_count'] = 0
            if 'processed_signal_count' not in plot_unit.settings:
                plot_unit.settings['processed_signal_count'] = 0
            if 'buffer_seconds' not in plot_unit.settings:
                plot_unit.settings['buffer_seconds'] = buffer_seconds
            
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
    print(f"\nDemo running for {duration} seconds with {buffer_seconds}s buffer. Press Ctrl+C to exit early.")
    try:
        # Switch to RAW view mode if possible
        if plot_unit and hasattr(plot_unit, '_set_mode'):
            from src.plot.view_mode import ViewMode
            plot_unit._set_mode(ViewMode.RAW)
        
        # Demonstrate adding additional signals with delays
        start_time = time.time()
        first_signal_added = False
        second_signal_added = False
        
        while time.time() - start_time < duration:
            # Show periodic status updates
            elapsed = int(time.time() - start_time)
            
            # Add a processed signal after 5 seconds
            if elapsed >= 5 and not first_signal_added:
                print("\nAdding processed signal WAVE2...")
                signal_id = generator.create_sine_signal(processed=True, name="WAVE2")
                if signal_id:
                    print(f"Added processed signal: {signal_id}")
                first_signal_added = True
                
            # Add another processed signal after 10 seconds
            if elapsed >= 10 and not second_signal_added:
                print("\nAdding processed signal WAVE3...")
                signal_id = generator.create_sine_signal(processed=True, name="WAVE3")
                if signal_id:
                    print(f"Added processed signal: {signal_id}")
                second_signal_added = True
            
            # Show signal counts and buffer size every 5 seconds
            if elapsed % 5 == 0 and elapsed > 0:
                raw_count = len([info for name, info in generator.generators['raw'].items() if info['created']])
                proc_count = len([info for name, info in generator.generators['processed'].items() if info['created']])
                print(f"Running demo... {elapsed}/{duration} seconds - " 
                      f"{raw_count}/3 raw signals, {proc_count}/3 processed signals, "
                      f"{generator.buffer_seconds}s buffer")
            
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
    import sys
    
    # Check if duration and buffer seconds were provided as command line arguments
    duration = 600  # Default: 10 minutes
    buffer_seconds = 10  # Default: 10 seconds
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
            print(f"Running with custom duration: {duration} seconds")
        except ValueError:
            print(f"Invalid duration argument: {sys.argv[1]}. Using default: {duration}s")
            
    if len(sys.argv) > 2:
        try:
            buffer_seconds = int(sys.argv[2])
            print(f"Using custom buffer size: {buffer_seconds} seconds")
        except ValueError:
            print(f"Invalid buffer argument: {sys.argv[2]}. Using default: {buffer_seconds}s")
    
    run_demo(duration=duration, buffer_seconds=buffer_seconds)
