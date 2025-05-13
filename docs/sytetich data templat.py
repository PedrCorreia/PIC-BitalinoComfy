import threading
import time
import numpy as np
from collections import deque, OrderedDict
import weakref
from ..src.plot import PygamePlot, PYGAME_AVAILABLE

class SyntheticDataGenerator:
    """
    Class for generating synthetic physiological data like EDA, ECG, and RR.
    Includes real-time plotting capabilities using the PygamePlot class.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.sampling_rate = 100
        self.duration = 10
        self.buffer_size = 10  # Default buffer size in seconds
        self.plot_thread = None
        self.last_plot_update = 0
        self._plot_update_interval = 0.05
        self._plot_nodes = weakref.WeakSet()
        self.signal_complete = False
        self.start_time = None
        self.plot_after_complete = True  # Always keep plot window open after completion
        self.fps = 60
        self.x_min = 0
        self.x_max = 10  # Default to buffer size
        self.adaptive_x_axis = True  # Auto-adjust x-axis range
        self.performance_mode = False
        self.line_thickness = 1
        self.enable_downsampling = False
        self.slow_noise_phase = 0.0
        self.slow_drift_value = 0.0
        self.next_artifact_time = 0.0
        self.artifact_active = False
        self.artifact_value = 0.0
        self.artifact_duration = 0.0

        # Replace single signal type with a dictionary of signal data
        self.enabled_signals = {"EDA": False, "ECG": False, "RR": False}
        self.signal_data = {
            "EDA": deque(maxlen=1000),
            "ECG": deque(maxlen=1000),
            "RR": deque(maxlen=1000)
        }
        
        # Signal-specific parameters for each type - use plot.py colors
        self.signal_params = {
            "EDA": {
                "scr_events": [],
                "baseline": 2.0,
                "color": PygamePlot.SIGNAL_COLORS["EDA"]  # Use centralized color
            },
            "ECG": {
                "heart_rate": 60,
                "color": PygamePlot.SIGNAL_COLORS["ECG"]  # Use centralized color
            },
            "RR": {
                "breathing_rate": 15,
                "color": PygamePlot.SIGNAL_COLORS["RR"]  # Use centralized color
            }
        }
        print("SyntheticDataGenerator initialized")

    def _background_generator(self):
        """Generate multi-signal data in real-time with precise timing"""
        i = 0
        self.start_time = time.time()
        self.signal_complete = False
        target_time = self.start_time
        
        # Reset data structures for each signal
        with self.lock:
            self.x_min = 0
            self.x_max = self.buffer_size
            for signal_type in self.signal_data:
                self.signal_data[signal_type].clear()
        
        # Initialize noise parameters for high-quality sensors
        self.slow_noise_phase = 0.0
        self.slow_drift_value = 0.0
        self.next_artifact_time = np.random.uniform(15.0, 30.0)
        self.artifact_active = False
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Check if signal should end based on duration
            if elapsed >= self.duration:
                print(f"Signal duration {self.duration}s reached")
                with self.lock:
                    self.signal_complete = True
                    if not self.plot_after_complete:
                        self._close_plot()
                    self.running = False
                break
            
            # Generate data with real-world timestamps
            real_time = elapsed
            
            # Generate common noise components
            baseline_freq = 0.2
            baseline_wander = 0.03 * np.sin(2 * np.pi * baseline_freq * real_time + self.slow_noise_phase)
            
            if i % int(self.sampling_rate * 10) == 0:
                drift_target = 0.01 * np.random.randn()
                self.slow_drift_value = 0.95 * self.slow_drift_value + 0.05 * drift_target
            
            if not self.artifact_active and real_time >= self.next_artifact_time:
                if np.random.rand() < 0.3:
                    self.artifact_active = True
                    artifact_type = np.random.choice(['minor_shift', 'brief_dropout'], p=[0.7, 0.3])
                    
                    if artifact_type == 'brief_dropout':
                        self.artifact_value = -0.5
                        self.artifact_duration = np.random.uniform(0.05, 0.1)
                    else:  # minor_shift
                        self.artifact_value = np.random.uniform(0.03, 0.08) * (1 if np.random.random() > 0.5 else -1)
                        self.artifact_duration = np.random.uniform(0.2, 0.5)
                        
                    self.artifact_end_time = real_time + self.artifact_duration
                
                self.next_artifact_time = real_time + np.random.uniform(15.0, 60.0)
            
            artifact_contribution = 0
            if self.artifact_active:
                if real_time < self.artifact_end_time:
                    time_in_artifact = real_time - (self.artifact_end_time - self.artifact_duration)
                    normalized_time = time_in_artifact / self.artifact_duration
                    
                    if normalized_time < 0.2:
                        scale = normalized_time / 0.2
                    elif normalized_time > 0.8:
                        scale = (1.0 - normalized_time) / 0.2
                    else:
                        scale = 1.0
                        
                    artifact_contribution = self.artifact_value * scale
                else:
                    self.artifact_active = False
            
            sensor_noise = baseline_wander + self.slow_drift_value + artifact_contribution
            electronic_noise = 0.005 * np.random.randn()
            
            with self.lock:
                if self.enabled_signals["EDA"]:
                    y_value = self._generate_eda_value(real_time, sensor_noise, electronic_noise)
                    self.signal_data["EDA"].append((real_time, float(y_value)))
                
                if self.enabled_signals["ECG"]:
                    y_value = self._generate_ecg_value(real_time, sensor_noise, electronic_noise)
                    self.signal_data["ECG"].append((real_time, float(y_value)))
                
                if self.enabled_signals["RR"]:
                    y_value = self._generate_rr_value(real_time, sensor_noise, electronic_noise)
                    self.signal_data["RR"].append((real_time, float(y_value)))
                
                if self.adaptive_x_axis:
                    self.x_max = real_time
                    self.x_min = max(0, self.x_max - self.buffer_size)
            
            if i % int(self.sampling_rate * 20) == 0:
                self.slow_noise_phase += np.random.uniform(-0.1, 0.1)
            
            i += 1
            target_time = self.start_time + (i / self.sampling_rate)
            sleep_time = target_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                #print(f"Generator falling behind by {-sleep_time:.1f}s")
                pass

    def _generate_eda_value(self, real_time, sensor_noise, electronic_noise):
        """Generate EDA signal value at the given time"""
        params = self.signal_params["EDA"]
        baseline = params["baseline"] + 0.3 * np.sin(2 * np.pi * 0.008 * real_time)
        
        if np.random.rand() < 0.001:
            scr_amplitude = np.random.uniform(0.2, 0.8)
            scr_rise_time = np.random.uniform(1.0, 3.0)
            scr_decay_time = np.random.uniform(3.0, 8.0)
            scr_start_time = real_time
            params["scr_events"].append((scr_amplitude, scr_rise_time, scr_decay_time, scr_start_time))
        
        scr_contribution = 0
        remaining_events = []
        for scr in params["scr_events"]:
            amp, rise, decay, start = scr
            t_since_start = real_time - start
            
            if t_since_start < rise + decay:
                if t_since_start < rise:
                    normalized = t_since_start / rise
                    contrib = amp * (1 - np.exp(-5 * normalized))
                else:
                    normalized = (t_since_start - rise) / decay
                    contrib = amp * np.exp(-3 * normalized)
                
                scr_contribution += contrib
                remaining_events.append(scr)
        
        params["scr_events"] = remaining_events
        
        signal_component = baseline + scr_contribution
        return signal_component + (sensor_noise * 0.4) + electronic_noise

    def _generate_ecg_value(self, real_time, sensor_noise, electronic_noise):
        """Generate ECG signal value at the given time"""
        params = self.signal_params["ECG"]
        heart_rate = params["heart_rate"] + 3 * np.sin(2 * np.pi * 0.05 * real_time)
        rr_interval = 60.0 / heart_rate
        t_mod = real_time % rr_interval
        
        p_wave = 0.15 * np.exp(-((t_mod - 0.1 * rr_interval) ** 2) / (2 * (0.02 * rr_interval) ** 2))
        q_wave = -0.1 * np.exp(-((t_mod - 0.2 * rr_interval) ** 2) / (2 * (0.01 * rr_interval) ** 2))
        r_wave = 1.0 * np.exp(-((t_mod - 0.22 * rr_interval) ** 2) / (2 * (0.008 * rr_interval) ** 2))
        s_wave = -0.3 * np.exp(-((t_mod - 0.24 * rr_interval) ** 2) / (2 * (0.01 * rr_interval) ** 2))
        t_wave = 0.3 * np.exp(-((t_mod - 0.35 * rr_interval) ** 2) / (2 * (0.03 * rr_interval) ** 2))
        
        signal_component = p_wave + q_wave + r_wave + s_wave + t_wave
        return signal_component + (sensor_noise * 0.1) + electronic_noise

    def _generate_rr_value(self, real_time, sensor_noise, electronic_noise):
        """Generate respiratory rate (RR) signal value at the given time"""
        params = self.signal_params["RR"]
        breathing_rate = params["breathing_rate"] + 1.5 * np.sin(2 * np.pi * 0.01 * real_time)
        breathing_freq = breathing_rate / 60.0
        
        inhale_exhale_ratio = 0.4
        phase = 2 * np.pi * breathing_freq * real_time
        
        if (phase % (2 * np.pi)) < (2 * np.pi * inhale_exhale_ratio):
            normalized_phase = (phase % (2 * np.pi)) / (2 * np.pi * inhale_exhale_ratio)
            breathing = np.sin(np.pi * normalized_phase / 2)
        else:
            normalized_phase = ((phase % (2 * np.pi)) - 2 * np.pi * inhale_exhale_ratio) / (2 * np.pi * (1 - inhale_exhale_ratio))
            breathing = 1 - normalized_phase
        
        signal_component = 60 + 3 * breathing
        return signal_component + (sensor_noise * 0.3) + electronic_noise

    def _ensure_multi_threads(self, enabled_signals, duration, sampling_rate, buffer_size, auto_restart=True, keep_window=True):
        """Ensure that the data generation thread is running with multiple signals enabled"""
        # Check if signal configuration has changed
        signal_config_changed = (
            enabled_signals != self.enabled_signals or
            sampling_rate != self.sampling_rate or
            duration != self.duration or
            buffer_size != self.buffer_size
        )
        
        needs_restart = (
            signal_config_changed or
            self.signal_complete or
            not self.running
        )
        
        self.plot_after_complete = keep_window
        self.buffer_size = buffer_size
        
        # If configuration changed, we need to restart
        if needs_restart:
            # First clean up any existing plot nodes
            self._cleanup_plot_nodes()
            self.plot_thread = None
            
            # Update configuration
            self.enabled_signals = enabled_signals.copy()
            self.sampling_rate = sampling_rate
            self.duration = duration
            
            # Reset data buffers with new size
            max_samples = int(sampling_rate * buffer_size)
            for signal_type in self.signal_data:
                self.signal_data[signal_type] = deque(maxlen=max_samples)
            
            # Reset state flags
            self.signal_complete = False
            self.running = False
            
            # Stop the generation thread if it's running
            if self.thread and self.thread.is_alive():
                self.running = False  # Signal thread to stop
                self.thread.join(timeout=0.2)  # Give it a moment to clean up
            
            # Close any existing plots
            self._close_plot()
            
            # Start new generation thread
            self.running = True
            self.thread = threading.Thread(target=self._background_generator, daemon=True)
            self.thread.start()
            
            active_signals = [s for s, enabled in self.enabled_signals.items() if enabled]
            print(f"Started multi-signal generation: {', '.join(active_signals)} - {buffer_size}s buffer ({max_samples} samples)")
    
    def _cleanup_plot_nodes(self):
        """Clean up plot nodes safely"""
        plot_nodes_to_close = list(self._plot_nodes)
        self._plot_nodes.clear()
        
        for plot_node in plot_nodes_to_close:
            try:
                plot_node._stop_event.set()
                # Allow time for resources to be released
                time.sleep(0.05)
                # If resize is needed, signal that
                plot_node._resize_needed = True
            except Exception as e:
                print(f"Error during plot node cleanup: {e}")
    
    def _plot_multi_data(self):
        """Start an optimized real-time plotting thread with multi-signal support"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available, cannot start plot")
            return False
            
        try:
            # Count active signals to calculate appropriate window height
            active_signals = [s for s, enabled in self.enabled_signals.items() if enabled]
            num_signals = len(active_signals)
            
            # Calculate window height based on number of signals
            # Minimum 180px per signal plus margins
            dynamic_height = max(
                480,  # Minimum height
                num_signals * PygamePlot.MIN_HEIGHT_PER_SIGNAL + 100  # Additional 100px for margins and labels
            )
            
            # Check if we already have a plot window that we can resize
            existing_plot = None
            for plot_node in self._plot_nodes:
                existing_plot = plot_node
                break
            
            if existing_plot:
                # Update existing plot window
                existing_plot.resize_window(PygamePlot.DEFAULT_WIDTH, dynamic_height)
                existing_plot.FPS = self.fps
                existing_plot.sampling_rate = self.sampling_rate
                existing_plot.LINE_THICKNESS = self.line_thickness
                existing_plot.multi_signal_mode = True
                existing_plot.enabled_signals = self.enabled_signals
                existing_plot.signal_colors = PygamePlot.SIGNAL_COLORS.copy()
                plot_node = existing_plot
            else:
                # Create new plot window
                plot_node = PygamePlot(
                    width=PygamePlot.DEFAULT_WIDTH,
                    height=dynamic_height,
                    performance_mode=self.performance_mode
                )
                plot_node.FPS = self.fps
                plot_node.sampling_rate = self.sampling_rate
                plot_node.adaptive_x_axis = self.adaptive_x_axis
                plot_node.LINE_THICKNESS = self.line_thickness
                
                plot_node.multi_signal_mode = True
                plot_node.enabled_signals = self.enabled_signals
                plot_node.signal_colors = PygamePlot.SIGNAL_COLORS.copy()
                
                self._plot_nodes.add(plot_node)
            
            self._plot_update_interval = 1.0 / self.fps
            
            if self.plot_thread is None or not self.plot_thread.is_alive():
                self.plot_thread = threading.Thread(
                    target=self._continuous_multi_plot_updater,
                    daemon=True
                )
                self.plot_thread.start()
            
            print(f"Real-time plot started for {', '.join(active_signals)} with FPS: {self.fps}, window height: {dynamic_height}px")
            return True
        except Exception as e:
            print(f"Error starting real-time plot: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _continuous_multi_plot_updater(self):
        """Update plots for multiple signals"""
        while self.running or (self.signal_complete and self.plot_after_complete):
            try:
                current_time = time.time()
                if current_time - self.last_plot_update >= self._plot_update_interval and len(self._plot_nodes) > 0:
                    with self.lock:
                        data_snapshots = {}
                        for signal_type, enabled in self.enabled_signals.items():
                            if enabled:
                                data_snapshots[signal_type] = list(self.signal_data[signal_type])
                    
                    x_min = float('inf')
                    x_max = float('-inf')
                    
                    for signal_data in data_snapshots.values():
                        if signal_data:
                            times, _ = zip(*signal_data)
                            x_min = min(x_min, min(times))
                            x_max = max(x_max, max(times))
                    
                    if x_min == float('inf'):
                        x_min = 0
                    if x_max == float('-inf'):
                        x_max = self.buffer_size
                    
                    for plot_node in list(self._plot_nodes):
                        try:
                            plot_node.plot_multi(
                                data_snapshots,
                                x_min=x_min, 
                                x_max=x_max,
                                enable_downsampling=self.enable_downsampling
                            )
                        except Exception as e:
                            print(f"Error updating plot: {e}")
                            self._plot_nodes.discard(plot_node)
                    
                    self.last_plot_update = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in plot update: {e}")
                time.sleep(0.1)
                
        print("Plot updater stopped")

    def _close_plot(self):
        """Close plot windows and fully reset plot state"""
        print("Closing all plot windows")
        self._cleanup_plot_nodes()
        
        # Force thread termination and cleanup
        if self.plot_thread and self.plot_thread.is_alive():
            try:
                self.plot_thread.join(timeout=0.2)
            except Exception:
                pass
        
        self.plot_thread = None
        # Additional cleanup to ensure we can re-create plots
        self.last_plot_update = 0

    def generate_multi(self, show_eda, show_ecg, show_rr, duration, sampling_rate, 
                      buffer_size, plot=True, fps=60, auto_restart=True, keep_window=True, 
                      performance_mode=False, line_thickness=1, enable_downsampling=False):
        """Generate multiple signals based on the enabled types"""
        self.fps = fps
        self.performance_mode = performance_mode
        self.line_thickness = line_thickness
        self.enable_downsampling = enable_downsampling
        
        enabled_signals = {
            "EDA": show_eda,
            "ECG": show_ecg,
            "RR": show_rr
        }
        
        if not any(enabled_signals.values()):
            enabled_signals["EDA"] = True
            
        self._ensure_multi_threads(enabled_signals, duration, sampling_rate, buffer_size, auto_restart=True, keep_window=True)
        
        with self.lock:
            data = {}
            for signal_type, enabled in self.enabled_signals.items():
                if enabled:
                    data[signal_type] = list(self.signal_data[signal_type])
            is_complete = self.signal_complete
        
        active_signals = [sig for sig, enabled in self.enabled_signals.items() if enabled]
        
        if active_signals and any(data.values()):
            if plot:
                self._plot_multi_data()
                
            if self.start_time and not is_complete:
                elapsed = time.time() - self.start_time
                remaining = max(0, self.duration - elapsed)
                
                total_samples = sum(len(signal_data) for signal_data in data.values())
                
                print(f"Signal time: {elapsed:.1f}/{self.duration:.1f}s, "
                      f"Total samples: {total_samples} across {len(active_signals)} signals")
            elif is_complete:
                total_samples = sum(len(signal_data) for signal_data in data.values())
                print(f"Signal complete. {total_samples} total samples across {len(active_signals)} signals")
            
            primary_signal = active_signals[0]
            if data[primary_signal]:
                fx, y = zip(*data[primary_signal])
                return list(fx), list(y), bool(plot), data
            
        return [], [], bool(plot), data