import threading
import time
import numpy as np
from collections import deque
import weakref
from ..src.plot import PygamePlot, PYGAME_AVAILABLE

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
        # Add min/max tracking for better axis control
        self.x_min = 0
        self.x_max = 10  # Default to buffer size
        self.adaptive_x_axis = True  # Auto-adjust x-axis range
        self.performance_mode = False
        self.window_width = 640
        self.window_height = 480
        self.line_thickness = 1
        self.enable_downsampling = False
        # Initialize noise parameters for high-quality sensors
        self.slow_noise_phase = 0.0
        self.slow_drift_value = 0.0
        self.next_artifact_time = 0.0
        self.artifact_active = False
        self.artifact_value = 0.0
        self.artifact_duration = 0.0
        # Add SCR tracking for EDA signals
        self.scr_events = []
        print("SyntheticDataGenerator initialized")

    def _background_generator(self):
        """Generate signal data in real-time with precise timing"""
        i = 0
        self.start_time = time.time()
        self.signal_complete = False
        target_time = self.start_time
        
        # Reset min/max tracking
        with self.lock:
            self.x_min = 0
            self.x_max = self.buffer_size
            
        # Initialize noise parameters for high-quality sensors
        self.slow_noise_phase = 0.0
        self.slow_drift_value = 0.0
        self.next_artifact_time = np.random.uniform(15.0, 30.0)  # Much less frequent artifacts
        self.artifact_active = False
        
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
            
            # Generate data with real-world timestamps
            real_time = elapsed
            
            # Generate refined noise components for high-quality sensors
            
            # 1. Very subtle baseline wander (primarily respiratory influence)
            baseline_freq = 0.2  # ~0.2 Hz = typical respiratory frequency
            baseline_wander = 0.03 * np.sin(2 * np.pi * baseline_freq * real_time + self.slow_noise_phase)
            
            # 2. Ultra-slow thermal drift (changes over minutes)
            if i % int(self.sampling_rate * 10) == 0:  # Update drift every 10 seconds
                drift_target = 0.01 * np.random.randn()
                self.slow_drift_value = 0.95 * self.slow_drift_value + 0.05 * drift_target
            
            # 3. Very rare artifacts (every 15-60 seconds for high-quality sensors)
            if not self.artifact_active and real_time >= self.next_artifact_time:
                if np.random.rand() < 0.3:  # Only 30% chance of artifact occurring
                    self.artifact_active = True
                    artifact_type = np.random.choice(['minor_shift', 'brief_dropout'], p=[0.7, 0.3])
                    
                    if artifact_type == 'brief_dropout':
                        self.artifact_value = -0.5  # Signal drops partially
                        self.artifact_duration = np.random.uniform(0.05, 0.1)  # Very brief (50-100ms)
                    else:  # minor_shift
                        self.artifact_value = np.random.uniform(0.03, 0.08) * (1 if np.random.random() > 0.5 else -1)
                        self.artifact_duration = np.random.uniform(0.2, 0.5)  # Short duration
                        
                    self.artifact_end_time = real_time + self.artifact_duration
                
                self.next_artifact_time = real_time + np.random.uniform(15.0, 60.0)
                
            # Apply artifact if active
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
            
            # Combine all noise components - much more subtle for high-quality sensors
            sensor_noise = baseline_wander + self.slow_drift_value + artifact_contribution
            
            # Add ultra-low amplitude electronic noise (white noise)
            electronic_noise = 0.005 * np.random.randn()
            
            # Generate signal-specific components with appropriate noise levels
            if self.signal_type == "EDA":
                baseline = 2.0 + 0.3 * np.sin(2 * np.pi * 0.008 * real_time)
                
                if np.random.rand() < 0.001:
                    scr_amplitude = np.random.uniform(0.2, 0.8)
                    scr_rise_time = np.random.uniform(1.0, 3.0)
                    scr_decay_time = np.random.uniform(3.0, 8.0)
                    scr_start_time = real_time
                    self.scr_events.append((scr_amplitude, scr_rise_time, scr_decay_time, scr_start_time))
                
                scr_contribution = 0
                remaining_events = []
                for scr in self.scr_events:
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
                
                self.scr_events = remaining_events
                
                signal_component = baseline + scr_contribution
                y_value = signal_component + (sensor_noise * 0.4) + electronic_noise
                
            elif self.signal_type == "ECG":
                heart_rate = 60 + 3 * np.sin(2 * np.pi * 0.05 * real_time)
                rr_interval = 60.0 / heart_rate
                t_mod = real_time % rr_interval
                
                p_wave = 0.15 * np.exp(-((t_mod - 0.1 * rr_interval) ** 2) / (2 * (0.02 * rr_interval) ** 2))
                q_wave = -0.1 * np.exp(-((t_mod - 0.2 * rr_interval) ** 2) / (2 * (0.01 * rr_interval) ** 2))
                r_wave = 1.0 * np.exp(-((t_mod - 0.22 * rr_interval) ** 2) / (2 * (0.008 * rr_interval) ** 2))
                s_wave = -0.3 * np.exp(-((t_mod - 0.24 * rr_interval) ** 2) / (2 * (0.01 * rr_interval) ** 2))
                t_wave = 0.3 * np.exp(-((t_mod - 0.35 * rr_interval) ** 2) / (2 * (0.03 * rr_interval) ** 2))
                
                signal_component = p_wave + q_wave + r_wave + s_wave + t_wave
                
                y_value = signal_component + (sensor_noise * 0.1) + electronic_noise
                
            elif self.signal_type == "RR":
                breathing_rate = 15 + 1.5 * np.sin(2 * np.pi * 0.01 * real_time)
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
                
                y_value = signal_component + (sensor_noise * 0.3) + electronic_noise
                
            else:
                y_value = np.sin(2 * np.pi * 0.1 * real_time) + (sensor_noise * 0.1) + electronic_noise
            
            with self.lock:
                self.data_deque.append((real_time, float(y_value)))
                
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
            self._plot_nodes.clear()
            self.plot_thread = None
            
            self.signal_type = signal_type
            self.sampling_rate = sampling_rate
            self.duration = duration
            
            max_samples = int(sampling_rate * buffer_size)
            self.data_deque = deque(maxlen=max_samples)
            
            self.signal_complete = False
            self.running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.1)
            
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
            plot_node = PygamePlot(
                width=self.window_width, 
                height=self.window_height,
                performance_mode=self.performance_mode
            )
            plot_node.FPS = self.fps
            plot_node.sampling_rate = self.sampling_rate
            plot_node.adaptive_x_axis = self.adaptive_x_axis
            plot_node.signal_type = self.signal_type
            plot_node.LINE_THICKNESS = self.line_thickness
            
            self._plot_nodes.add(plot_node)
            
            self._plot_update_interval = 1.0 / self.fps
            
            if self.plot_thread is None or not self.plot_thread.is_alive():
                self.plot_thread = threading.Thread(
                    target=self._continuous_plot_updater,
                    daemon=True
                )
                self.plot_thread.start()
            
            print(f"Real-time plot started with FPS: {self.fps}, Performance mode: {self.performance_mode}, Downsampling: {self.enable_downsampling}")
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
                current_time = time.time()
                if current_time - self.last_plot_update >= self._plot_update_interval and len(self._plot_nodes) > 0:
                    with self.lock:
                        data_snapshot = list(self.data_deque)
                    
                    if data_snapshot:
                        times, values = zip(*data_snapshot)
                        
                        x_min = min(times)
                        x_max = max(times)
                        
                        if self.signal_complete:
                            fx_list = list(times)
                        else:
                            fx_list = list(times)
                        
                        for plot_node in list(self._plot_nodes):
                            try:
                                plot_node.plot(
                                    fx_list, list(values), False, 
                                    x_min=x_min, x_max=x_max, 
                                    signal_type=self.signal_type,
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
        plot_nodes = list(self._plot_nodes)
        self._plot_nodes.clear()
        
        for plot_node in plot_nodes:
            try:
                plot_node._stop_event.set()
            except Exception as e:
                print(f"Error closing plot: {e}")
        
        if self.plot_thread and self.plot_thread.is_alive():
            try:
                self.plot_thread.join(timeout=0.2)
            except Exception:
                pass
        
        self.plot_thread = None
        self.last_plot_update = 0

    def generate(self, signal_type, duration, sampling_rate, buffer_size, plot=True, 
                fps=60, auto_restart=True, keep_window=True, performance_mode=False, 
                window_width=640, window_height=480, line_thickness=1, enable_downsampling=False):
        """Generate signal data with proper buffer handling"""
        self.fps = fps
        self.performance_mode = performance_mode
        self.window_width = window_width
        self.window_height = window_height
        self.line_thickness = line_thickness
        self.signal_type = signal_type
        self.enable_downsampling = enable_downsampling
        
        self._ensure_thread(signal_type, duration, sampling_rate, buffer_size, auto_restart, keep_window)
        
        with self.lock:
            data = list(self.data_deque)
            is_complete = self.signal_complete
        
        if data:
            times, values = zip(*data)
            
            if plot:
                self._plot_data(times, values)
                
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
