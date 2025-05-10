import threading
import time
import numpy as np
from collections import deque
import weakref
from .plot import PygamePlot, PYGAME_AVAILABLE

class SyntheticDataGenerator:
    """
    Class for generating synthetic physiological data like EDA, ECG, and RR.
    Includes real-time plotting capabilities using the PygamePlot class.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.data_deque = deque(maxlen=1000)  # Will be adjusted by buffer_size
        self.running = False
        self.thread = None
        self.signal_type = "EDA"
        self.sampling_rate = 100
        self.duration = 10
        self.buffer_size = 10  # Default buffer size in seconds
        self.plot_thread = None
        self.last_plot_update = 0
        self._plot_update_interval = 0.05
        self._plot_nodes = weakref.WeakSet()
        self.signal_complete = False
        self.start_time = None
        self.plot_after_complete = True  # Keep plot window open after completion
        self.fps = 60
        print("SyntheticDataGenerator initialized")

    def _background_generator(self):
        """Generate signal data in real-time with precise timing"""
        i = 0
        self.start_time = time.time()
        self.signal_complete = False
        target_time = self.start_time
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Check if signal should end based on duration
            if elapsed >= self.duration:
                print(f"Signal duration {self.duration}s reached")
                with self.lock:
                    self.signal_complete = True
                    
                    # DON'T close plot windows if keep_window is True
                    if not self.plot_after_complete:
                        self._close_plot()
                    self.running = False
                break
            
            # Generate data with real-world timestamps (not just sample indices)
            real_time = elapsed  # Use actual elapsed time as the x-value
            
            # Generate exactly one sample with correct timestamp
            if self.signal_type == "EDA":
                baseline = 0.5 * np.sin(2 * np.pi * 0.01 * real_time)
                y_value = baseline + 0.05 * np.random.randn()
                if np.random.rand() < 0.01:
                    y_value += np.random.uniform(0.2, 0.4)
            elif self.signal_type == "ECG":
                heart_rate = 60
                rr_interval = 60.0 / heart_rate
                t_mod = real_time % rr_interval
                
                p_wave = 0.1 * np.exp(-((t_mod - 0.1) ** 2) / (2 * 0.01 ** 2))
                q_wave = -0.15 * np.exp(-((t_mod - 0.2) ** 2) / (2 * 0.008 ** 2))
                r_wave = 1.0 * np.exp(-((t_mod - 0.22) ** 2) / (2 * 0.005 ** 2))
                s_wave = -0.25 * np.exp(-((t_mod - 0.24) ** 2) / (2 * 0.008 ** 2))
                t_wave = 0.3 * np.exp(-((t_mod - 0.35) ** 2) / (2 * 0.02 ** 2))
                
                y_value = p_wave + q_wave + r_wave + s_wave + t_wave + 0.005 * np.random.randn()
            elif self.signal_type == "RR":
                y_value = 60 + 5 * np.sin(2 * np.pi * 0.1 * real_time) + np.random.randn()
            else:
                y_value = np.random.randn()  # Default to random noise if unknown type
            
            # Add data to deque with real-world timestamp
            with self.lock:
                self.data_deque.append((real_time, float(y_value)))
            
            # Real-time synchronization for precise timing
            i += 1
            target_time = self.start_time + (i / self.sampling_rate)
            sleep_time = target_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:  # Only log when more than 1 second behind
                print(f"Generator falling behind by {-sleep_time:.1f}s")

    def _ensure_thread(self, signal_type, duration, sampling_rate, buffer_size, auto_restart=True, keep_window=True):
        """Ensure that the data generation thread is running with the correct parameters"""
        restart = (
            signal_type != self.signal_type or
            sampling_rate != self.sampling_rate or
            duration != self.duration or
            buffer_size != self.buffer_size or
            self.signal_complete or
            not self.running
        )
        
        self.plot_after_complete = keep_window
        self.buffer_size = buffer_size
        
        if restart:
            # Explicitly clean up internal state to ensure fresh start
            self._plot_nodes.clear()  # Clear any references to old plot windows
            self.plot_thread = None   # Reset plot thread reference
            
            # Clean up properly
            self.signal_type = signal_type
            self.sampling_rate = sampling_rate
            self.duration = duration
            
            # Set buffer size based on specified seconds
            max_samples = int(sampling_rate * buffer_size)  # Convert seconds to samples
            self.data_deque = deque(maxlen=max_samples)
            
            self.signal_complete = False
            self.running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.1)
            
            # Only close plot if we're restarting (not if first run)
            if self.thread is not None:
                self._close_plot() 
            
            self.running = True
            self.thread = threading.Thread(target=self._background_generator, daemon=True)
            self.thread.start()
            print(f"Started signal generation with {buffer_size}s buffer ({max_samples} samples)")

    def _plot_data(self, fx, y, as_points=False):
        """Start an optimized real-time plotting thread with configurable FPS"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot start plot")
            return False
            
        try:
            # Configure the plot with our fps and sampling rate settings
            plot_node = PygamePlot()
            plot_node.FPS = self.fps
            plot_node.sampling_rate = self.sampling_rate  # Pass sampling rate to plot
            self._plot_nodes.add(plot_node)
            
            # Adjust plot update interval based on FPS
            self._plot_update_interval = 1.0 / self.fps
            
            if self.plot_thread is None or not self.plot_thread.is_alive():
                self.plot_thread = threading.Thread(
                    target=self._continuous_plot_updater,
                    daemon=True
                )
                self.plot_thread.start()
            
            print(f"Real-time plot started with FPS: {self.fps}")
            return True
        except Exception as e:
            print(f"Error starting real-time plot: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _continuous_plot_updater(self):
        """Update plots respecting real-time progression"""
        while self.running or (self.signal_complete and self.plot_after_complete):
            try:
                # Only update at most at our fps rate
                current_time = time.time()
                if current_time - self.last_plot_update >= self._plot_update_interval and len(self._plot_nodes) > 0:
                    # Get data snapshot
                    with self.lock:
                        data_snapshot = list(self.data_deque)
                    
                    # Process data
                    if data_snapshot:
                        times, values = zip(*data_snapshot)
                        
                        # Adjust times to be relative to signal start
                        # This ensures plot shows proper timestamps whether we're viewing 
                        # the start, middle or end of the signal
                        if self.signal_complete:
                            # For completed signals, show the final buffer
                            fx_list = list(times)
                        else:
                            # For ongoing signals, show only the buffer_size window
                            current_runtime = current_time - self.start_time
                            t_min = max(0, current_runtime - self.buffer_size)
                            fx_list = [t for t in times if t >= t_min]
                            values = [values[i] for i, t in enumerate(times) if t >= t_min]
                        
                        # Update plots
                        for plot_node in list(self._plot_nodes):
                            try:
                                plot_node.plot(fx_list, list(values), False)
                            except Exception:
                                self._plot_nodes.discard(plot_node)
                    
                    self.last_plot_update = current_time
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in plot update: {e}")
                time.sleep(0.1)
                
        print("Plot updater stopped")

    def _close_plot(self):
        """Close plot windows and fully reset plot state"""
        print("Closing all plot windows")
        plot_nodes = list(self._plot_nodes)
        self._plot_nodes.clear()
        
        for plot_node in plot_nodes:
            try:
                plot_node._stop_event.set()
            except Exception as e:
                print(f"Error closing plot: {e}")
        
        # Force thread termination and cleanup
        if self.plot_thread and self.plot_thread.is_alive():
            try:
                self.plot_thread.join(timeout=0.2)
            except Exception:
                pass
        
        self.plot_thread = None
        # Additional cleanup to ensure we can re-create plots
        self.last_plot_update = 0

    def generate(self, signal_type, duration, sampling_rate, buffer_size, plot=True, fps=60, auto_restart=True, keep_window=True):
        """Generate signal data with proper buffer handling"""
        # Update settings
        self.fps = fps
        
        # If completed but window still open, accept new parameters and force full reset
        if self.signal_complete:
            print("Signal completed, forcing full reset for new signal.")
            self.signal_complete = False
            self.running = False
            
            # Force plot cleanup to ensure we can create new ones
            self._close_plot()
            self._plot_nodes.clear()
            self.plot_thread = None
        
        # Ensure thread is running with correct parameters
        self._ensure_thread(signal_type, duration, sampling_rate, buffer_size, auto_restart, keep_window)
        
        # Handle plotting - make sure it's reset properly
        if plot and PYGAME_AVAILABLE:
            # Always attempt to start a plot when plot=True
            if len(self._plot_nodes) == 0:
                success = self._plot_data([], [])
                print(f"Starting plot: {'Success' if success else 'Failed'}")
        elif not plot and not (self.signal_complete and self.plot_after_complete):
            self._close_plot()
        
        # Get current data
        with self.lock:
            data_copy = list(self.data_deque)
            is_complete = self.signal_complete
        
        # Process data for output
        if data_copy:
            times, values = zip(*data_copy)
            
            # Show buffer info and time remaining
            if self.start_time and not is_complete:
                elapsed = time.time() - self.start_time
                remaining = max(0, self.duration - elapsed)
                print(f"Signal time: {elapsed:.1f}/{self.duration:.1f}s, Buffer: {len(times)} samples ({len(times)/self.sampling_rate:.1f}s)")
            elif is_complete:
                print(f"Signal complete. Showing {len(times)} samples ({len(times)/self.sampling_rate:.1f}s)")
            
            return list(times), list(values), bool(plot), (list(times), list(values))
        else:
            return [], [], bool(plot), ([], [])

# Note: SynthNode moved to comfy/synthetic_generator.py to keep all node definitions in the comfy directory
