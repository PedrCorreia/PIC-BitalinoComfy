"""
Unified Signal Generator for PIC-2025.
This node combines the functionality of RegistrySyntheticGenerator and RegistrySignalGenerator
into a single cohesive generator that can produce both physiological and standard waveforms.
"""

import sys
import os
import uuid
import numpy as np
import torch
import random
import time
import threading
import logging
from typing import Dict, List, Tuple, Union, Optional
import math
from ...src.utils.synthetic_data import SyntheticDataGenerator
from ...src.registry.signal_registry import SignalRegistry

# Set up logger
logger = logging.getLogger("ComfyUI")
class UnifiedSignalGenerator:
    """
    Unified signal generator that provides both physiological and standard waveforms.
    
    This node combines two previous generators:
    1. RegistrySyntheticGenerator - for physiological signals (EDA, ECG, RR)
    2. RegistrySignalGenerator - for standard waveforms (sine, square, etc.)
    
    All signals are properly registered with SignalRegistry for visualization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Signal type selection
                "generator_mode": (["physiological", "waveform"], {"default": "physiological"}),
                
                # Core parameters
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "sampling_rate": ("INT", {"default": 100, "min": 1, "max": 1000}),
                
                # Signal characteristics
                "noise_level": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                # Physiological signal options
                "show_eda": ("BOOLEAN", {"default": True}),
                "show_ecg": ("BOOLEAN", {"default": False}),
                "show_rr": ("BOOLEAN", {"default": False}),
                "custom_eda_id": ("STRING", {"default": "EDA"}),
                "custom_ecg_id": ("STRING", {"default": "ECG"}),
                "custom_rr_id": ("STRING", {"default": "RR"}),
                
                # Waveform options
                "waveform_type": (["sine", "square", "triangle", "sawtooth", "noise", "ecg_synthetic", "random"], {"default": "sine"}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "signal_id": ("STRING", {"default": "signal_1"}),
                
                # Common options
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "live_generation": ("BOOLEAN", {"default": True, "label_on": "Live Data", "label_off": "Static Data"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_ids",)
    FUNCTION = "generate_signals"
    CATEGORY = "signal/generators"
    OUTPUT_NODE = False
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"unified_generator_{str(uuid.uuid4())[:8]}"
        
        # Get registry singleton
        self.registry = SignalRegistry.get_instance()
        
        # Create synthetic data generator for physiological signals
        self.physio_generator = SyntheticDataGenerator()
        self.physio_generator.plot = False  # Disable direct plotting
        
        # Thread for live signal generation
        self.generation_thread = None
        self.stop_event = threading.Event()
        self.active_signals = {}  # Track active signals for live updates
        
        logger.info(f"Unified Signal Generator Node {self.node_id} initialized")
    
    def __del__(self):
        """Clean up any running threads when the node is deleted"""
        self.stop_live_generation()
    
    def stop_live_generation(self):
        """Stop any ongoing live signal generation threads"""
        if self.generation_thread and self.generation_thread.is_alive():
            self.stop_event.set()
            self.generation_thread.join(timeout=2.0)  # Wait up to 2 seconds
            logger.info("Live signal generation stopped")
    
    def generate_signals(self, 
                         generator_mode,
                         duration, sampling_rate, 
                         noise_level, amplitude,
                         show_eda=True, show_ecg=False, show_rr=False, 
                         custom_eda_id="EDA", custom_ecg_id="ECG", custom_rr_id="RR",
                         waveform_type="sine", frequency=1.0, signal_id="signal_1",
                         seed=-1, live_generation=True):
        """
        Unified method to generate signals based on the selected mode
        
        Args:
            generator_mode: "physiological" or "waveform"
            duration: Duration in seconds
            sampling_rate: Sample rate (samples per second)
            noise_level: Amount of noise (0.0-1.0)
            amplitude: Signal amplitude multiplier
            show_eda: Whether to generate EDA signal (physiological mode)
            show_ecg: Whether to generate ECG signal (physiological mode)
            show_rr: Whether to generate RR signal (physiological mode)
            custom_eda_id: Custom ID for the EDA signal
            custom_ecg_id: Custom ID for the ECG signal
            custom_rr_id: Custom ID for the RR signal
            waveform_type: Type of waveform to generate (waveform mode)
            frequency: Frequency of the waveform in Hz (waveform mode)
            signal_id: ID for the waveform signal
            seed: Random seed (-1 for random)
            live_generation: Generate signals in real-time
            
        Returns:
            signal_ids: Comma-separated list of generated signal IDs
        """
        # Use specified random seed if provided
        if seed != -1:
            # Ensure seed is within numpy's valid range (0 to 2^32-1)
            valid_seed = abs(seed) % (2**32)
            random.seed(valid_seed)
            np.random.seed(valid_seed)
        
        # Stop any existing generation threads
        self.stop_live_generation()
        
        # Generate signals based on the selected mode
        if generator_mode == "physiological":
            # Generate physiological signals (EDA, ECG, RR)
            signal_ids = self._generate_physiological_signals(
                show_eda, show_ecg, show_rr,
                custom_eda_id, custom_ecg_id, custom_rr_id,
                duration, sampling_rate, noise_level, amplitude
            )
        else:
            # Generate standard waveform signals
            signal_ids = self._generate_waveform_signal(
                waveform_type, frequency, amplitude,
                duration, sampling_rate, noise_level, signal_id
            )
        
        # Start live generation if requested
        if live_generation and signal_ids:
            self._start_live_generation(
                generator_mode, signal_ids,
                duration, sampling_rate, noise_level, amplitude,
                show_eda, show_ecg, show_rr,
                waveform_type, frequency
            )
        
        # Return comma-separated list of signal IDs
        return (",".join(signal_ids),)
    
    def _generate_physiological_signals(self, 
                                       show_eda, show_ecg, show_rr,
                                       custom_eda_id, custom_ecg_id, custom_rr_id,
                                       duration, sampling_rate, noise_level, amplitude):
        """Generate physiological signals using the SyntheticDataGenerator"""
        logger.info(f"Generating physiological signals: EDA={show_eda}, ECG={show_ecg}, RR={show_rr}")
        
        # Make sure at least one signal is enabled
        if not (show_eda or show_ecg or show_rr):
            show_eda = True  # Default to EDA if none selected
        
        # Create mapping of signal types to IDs
        signal_ids = {}
        if show_eda:
            signal_ids["EDA"] = custom_eda_id
        if show_ecg:
            signal_ids["ECG"] = custom_ecg_id
        if show_rr:
            signal_ids["RR"] = custom_rr_id
            
        # Configure the generator
        self.physio_generator.buffer_size = duration
        
        # Generate the data without plotting
        result = self.physio_generator.generate_multi(
            show_eda=show_eda, 
            show_ecg=show_ecg, 
            show_rr=show_rr,
            duration=duration, 
            sampling_rate=sampling_rate, 
            buffer_size=duration,
            plot=False,
            fps=60,
            auto_restart=False, 
            keep_window=False,
            performance_mode=False,
            line_thickness=1,
            enable_downsampling=False
        )
        
        # Extract data depending on the return format
        if len(result) == 5:
            _, _, _, data, _ = result
        elif len(result) == 4:
            _, _, _, data = result
        else:
            logger.error("Unexpected result format from generator")
            return list(signal_ids.values())
            
        # Register each signal with the registry
        active_signals = []
        for signal_type, reg_id in signal_ids.items():
            if signal_type in data and len(data[signal_type]) > 0:
                # Extract x and y values
                x_values = np.array([point[0] for point in data[signal_type]], dtype=np.float32)
                y_values = np.array([point[1] for point in data[signal_type]], dtype=np.float32)
                
                # Apply amplitude scaling and noise
                y_values = y_values * amplitude
                
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * np.std(y_values), len(y_values))
                    y_values = y_values + noise
                
                # Determine color based on signal type
                if signal_type == "EDA":
                    color = (0, 255, 0)  # Green
                elif signal_type == "ECG":
                    color = (255, 0, 0)  # Red
                elif signal_type == "RR":
                    color = (255, 165, 0)  # Orange
                else:
                    color = (0, 0, 255)  # Blue default
                
                # Register with the registry
                self.registry.register_signal(
                    signal_id=reg_id,
                    signal_data=y_values,
                    metadata={
                        'color': color,
                        'source_node': self.node_id,
                        'signal_type': signal_type,
                        'x_values': x_values,
                        'sampling_rate': sampling_rate,
                        'generator_mode': 'physiological'
                    }
                )
                
                # Store for tracking live updates
                self.active_signals[reg_id] = {
                    'type': signal_type,
                    'color': color,
                    'sampling_rate': sampling_rate,
                    'last_x': x_values[-1] if len(x_values) > 0 else 0
                }
                
                active_signals.append(reg_id)
                logger.info(f"Registered {signal_type} signal as '{reg_id}' with {len(y_values)} samples")
        
        return active_signals

    def _generate_ecg_signal(self, t, amplitude, frequency):
        """Generate a synthetic ECG-like signal"""
        # Parameter to control the sharpness of the peak (R wave)
        r_sharpness = 30.0
        
        # Parameters for P and T waves
        p_offset = -0.2
        p_width = 0.1
        p_height = 0.2
        
        t_offset = 0.2
        t_width = 0.15
        t_height = 0.3
        
        # Base frequency for heart rate
        # frequency is the heart rate in Hz (e.g., 1.2 Hz = 72 BPM)
        
        # Normalize the time to the frequency to get phases in [0, 1] range
        phase = (t * frequency) % 1.0
        
        # Initialize the signal with a small baseline
        signal = np.zeros_like(t)
        
        # R peak (sharp spike)
        r_peak = amplitude * np.exp(-r_sharpness * (phase - 0.0)**2)
        
        # P wave (before R peak)
        p_wave = p_height * amplitude * np.exp(-(phase - p_offset)**2 / p_width)
        
        # T wave (after R peak)
        t_wave = t_height * amplitude * np.exp(-(phase - t_offset)**2 / t_width)
        
        # Q and S waves (small negative deflections before and after R)
        q_wave = -0.1 * amplitude * np.exp(-80.0 * (phase - (-0.05))**2)
        s_wave = -0.2 * amplitude * np.exp(-50.0 * (phase - 0.05)**2)
        
        # Combine all components
        signal = r_peak + p_wave + t_wave + q_wave + s_wave
        
        # Add a small baseline to prevent too low values
        signal += 0.1 * amplitude
        
        return signal

    def _generate_waveform_signal(self, 
                                 waveform_type, frequency, amplitude,
                                 duration, sampling_rate, noise_level, signal_id):
        """Generate a waveform signal (sine, square, etc.)"""
        logger.info(f"Generating {waveform_type} waveform with frequency {frequency}Hz")
        
        # Calculate the number of samples needed
        num_samples = int(duration * sampling_rate)
        
        # Generate time values
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate the appropriate waveform
        if waveform_type == "sine":
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            color = (0, 100, 255)  # Blue
        elif waveform_type == "square":
            signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
            color = (255, 100, 0)  # Orange
        elif waveform_type == "triangle":
            signal = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
            color = (0, 200, 100)  # Green
        elif waveform_type == "sawtooth":
            signal = amplitude * ((2 * (frequency * t - np.floor(0.5 + frequency * t))))
            color = (200, 50, 200)  # Purple
        elif waveform_type == "ecg_synthetic":
            signal = self._generate_ecg_signal(t, amplitude, frequency)
            color = (255, 0, 0)  # Red
        elif waveform_type == "noise":
            signal = amplitude * np.random.randn(num_samples)
            color = (150, 150, 150)  # Gray
        elif waveform_type == "random":
            # Random walk with some drift
            steps = np.random.randn(num_samples)
            signal = np.cumsum(steps) * 0.1 * amplitude
            # Normalize to keep within amplitude
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6) * 2 * amplitude - amplitude
            color = (100, 200, 200)  # Teal
        else:
            # Default to sine if unknown type
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            color = (0, 100, 255)  # Blue
            
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * amplitude, len(signal))
            signal = signal + noise
            
        # Register the signal with the registry
        self.registry.register_signal(
            signal_id=signal_id,
            signal_data=signal,
            metadata={
                'color': color,
                'source_node': self.node_id,
                'signal_type': waveform_type,
                'x_values': t,
                'sampling_rate': sampling_rate,
                'frequency': frequency,
                'generator_mode': 'waveform'
            }
        )
        
        # Store for tracking live updates
        self.active_signals[signal_id] = {
            'type': waveform_type,
            'color': color,
            'sampling_rate': sampling_rate,
            'frequency': frequency,
            'last_x': duration
        }
        
        logger.info(f"Registered {waveform_type} signal as '{signal_id}' with {len(signal)} samples")
        
        return [signal_id]
    
    def _process_signal(self, signal, signal_type):
        """Process a signal for visualization (optional)"""
        processed = signal.copy()
        
        # Apply signal-specific processing
        if signal_type == "ECG":
            # Emphasize peaks for ECG
            processed = signal * 1.2
            
        # Add slight phase shift for visualization
        processed = np.roll(processed, len(signal) // 20)
            
        return processed
        
    def _start_live_generation(self, 
                              generator_mode, signal_ids,
                              duration, sampling_rate, noise_level, amplitude,
                              show_eda, show_ecg, show_rr,
                              waveform_type, frequency):
        """Start a thread for live signal generation"""
        if not signal_ids:
            return
            
        # Convert to list if string
        if isinstance(signal_ids, str):
            signal_ids = signal_ids.split(",")
        
        # Reset the stop event
        self.stop_event = threading.Event()
        
        # Start the appropriate generation thread
        if generator_mode == "physiological":
            self.generation_thread = threading.Thread(
                target=self._live_physiological_generation,
                args=(signal_ids, sampling_rate, noise_level, amplitude, show_eda, show_ecg, show_rr),
                daemon=True
            )
        else:
            self.generation_thread = threading.Thread(
                target=self._live_waveform_generation,
                args=(signal_ids[0], waveform_type, frequency, amplitude, sampling_rate, noise_level),
                daemon=True
            )
            
        self.generation_thread.start()
        logger.info(f"Started live generation thread for {len(signal_ids)} signal(s)")
        
    def _live_physiological_generation(self, 
                                      signal_ids, sampling_rate, noise_level, amplitude,
                                      show_eda, show_ecg, show_rr):
        """Generate physiological signals continuously in a background thread"""
        try:
            # Create a mapping of signal IDs to types
            signal_types = {}
            for sig_id in signal_ids:
                for active_id, info in self.active_signals.items():
                    if sig_id == active_id:
                        signal_types[sig_id] = info['type']
                        
            # Loop until stopped
            while not self.stop_event.is_set():
                for sig_id, sig_type in signal_types.items():
                    # Get current signal data
                    signal_data = self.registry.get_signal(sig_id)
                    if signal_data is None:
                        continue
                        
                    metadata = self.registry.get_signal_metadata(sig_id)
                    if metadata is None:
                        continue
                    
                    # Generate a small segment of new data (1/10 second)
                    new_samples = max(1, int(sampling_rate / 10))
                    new_signal = None
                    
                    # Last x value from existing data
                    last_x = metadata.get('x_values', [0])[-1] if 'x_values' in metadata else 0
                    
                    # Generate new data based on signal type
                    if sig_type == "EDA":
                        # EDA: Slow changing signal with small fluctuations
                        base = np.mean(signal_data[-20:]) if len(signal_data) > 20 else 0.5
                        new_signal = base + np.random.normal(0, 0.01, new_samples) + np.linspace(0, 0.01, new_samples)
                    elif sig_type == "ECG":
                        # ECG: Continue the pattern with peaks
                        new_t = np.linspace(last_x, last_x + new_samples/sampling_rate, new_samples)
                        new_signal = self._generate_ecg_signal(new_t, amplitude, 1.2)  # 1.2 Hz = ~72 BPM
                    elif sig_type == "RR":
                        # RR: Respiratory rate - slower sine wave
                        new_t = np.linspace(last_x, last_x + new_samples/sampling_rate, new_samples)
                        new_signal = 0.8 * amplitude * np.sin(2 * np.pi * 0.25 * new_t)
                        
                    if new_signal is not None:
                        # Add noise
                        if noise_level > 0:
                            noise = np.random.normal(0, noise_level * amplitude * 0.1, new_samples)
                            new_signal = new_signal + noise
                        
                        # Append to existing signal
                        updated_signal = np.concatenate([signal_data, new_signal])
                        
                        # Keep a maximum of 10 seconds of data
                        max_samples = 10 * sampling_rate
                        if len(updated_signal) > max_samples:
                            updated_signal = updated_signal[-max_samples:]
                            
                        # Update x values
                        new_x = np.linspace(last_x, last_x + new_samples/sampling_rate, new_samples)
                        if 'x_values' in metadata:
                            x_values = np.concatenate([metadata['x_values'], new_x])
                            if len(x_values) > max_samples:
                                x_values = x_values[-max_samples:]
                        else:
                            x_values = new_x
                            
                        # Update metadata
                        metadata['x_values'] = x_values
                        
                        # Update registry
                        self.registry.register_signal(sig_id, updated_signal, metadata)
                        
                # Sleep before next update
                time.sleep(0.1)  # Update 10 times per second
                
        except Exception as e:
            logger.error(f"Error in live physiological generation: {e}")
            
    def _live_waveform_generation(self, 
                                signal_id, waveform_type, frequency, amplitude, 
                                sampling_rate, noise_level):
        """Generate waveform signals continuously in a background thread"""
        try:
            # Loop until stopped
            while not self.stop_event.is_set():
                # Get current signal data
                signal_data = self.registry.get_signal(signal_id)
                if signal_data is None:
                    continue
                    
                metadata = self.registry.get_signal_metadata(signal_id)
                if metadata is None:
                    continue
                
                # Generate a small segment of new data (1/10 second)
                new_samples = max(1, int(sampling_rate / 10))
                
                # Last x value from existing data
                last_x = metadata.get('x_values', [0])[-1] if 'x_values' in metadata else 0
                new_t = np.linspace(last_x, last_x + new_samples/sampling_rate, new_samples)
                
                # Generate the appropriate waveform for the new segment
                if waveform_type == "sine":
                    new_signal = amplitude * np.sin(2 * np.pi * frequency * new_t)
                elif waveform_type == "square":
                    new_signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * new_t))
                elif waveform_type == "triangle":
                    new_signal = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * new_t))
                elif waveform_type == "sawtooth":
                    new_signal = amplitude * ((2 * (frequency * new_t - np.floor(0.5 + frequency * new_t))))
                elif waveform_type == "ecg_synthetic":
                    new_signal = self._generate_ecg_signal(new_t, amplitude, frequency)
                elif waveform_type == "noise":
                    new_signal = amplitude * np.random.randn(new_samples)
                elif waveform_type == "random":
                    # Random walk - continue from last value
                    last_value = signal_data[-1] if len(signal_data) > 0 else 0
                    steps = np.random.randn(new_samples) * 0.1 * amplitude
                    new_signal = np.cumsum(steps) + last_value
                    # Keep within bounds
                    new_signal = np.clip(new_signal, -amplitude, amplitude)
                else:
                    # Default to sine
                    new_signal = amplitude * np.sin(2 * np.pi * frequency * new_t)
                
                # Add noise if specified
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * amplitude, new_samples)
                    new_signal = new_signal + noise
                    
                # Append to existing signal
                updated_signal = np.concatenate([signal_data, new_signal])
                
                # Keep a maximum of 10 seconds of data
                max_samples = 10 * sampling_rate
                if len(updated_signal) > max_samples:
                    updated_signal = updated_signal[-max_samples:]
                    
                # Update x values
                if 'x_values' in metadata:
                    x_values = np.concatenate([metadata['x_values'], new_t])
                    if len(x_values) > max_samples:
                        x_values = x_values[-max_samples:]
                else:
                    x_values = new_t
                    
                # Update metadata
                metadata['x_values'] = x_values
                
                # Update registry
                self.registry.register_signal(signal_id, updated_signal, metadata)
                
                # Sleep before next update
                time.sleep(0.1)  # Update 10 times per second
                
        except Exception as e:
            logger.error(f"Error in live waveform generation: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "UnifiedSignalGenerator": UnifiedSignalGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedSignalGenerator": "🌊 Signal Generator"
}
