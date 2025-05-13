"""
Unified Signal Generator for PIC-2025.
This node generates either physiological or standard waveform signals based on the selected mode.
Each signal is generated in its own background thread and registered with the SignalRegistry.
"""

import sys
import os
import uuid
import numpy as np
import random
import time
import threading
import logging
from typing import Dict, List, Tuple, Union, Optional
import math
from ...src.registry.signal_registry  import SignalRegistry

# Set up logger
logger = logging.getLogger("ComfyUI")

class UnifiedSignalGenerator:
    """
    Simplified Unified Signal Generator that follows the single signal type per node principle.
    
    Key features:
    - Single signal type per node instance
    - Mode selection between physiological and waveform signals
    - Continuous live data generation in background thread
    - Direct signal registration with the registry system
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Signal identification
                "id": ("STRING", {"default": "signal_1"}),
                
                # Mode selection
                "mode": (["physiological", "waveform"], {"default": "physiological"}),
                
                # Signal type depends on mode - will be conditionally shown 
                "signal_type": (["EDA", "ECG", "RR", "sine", "square", "triangle", "sawtooth", "noise", "random"], {"default": "EDA"}),
                
                # Core parameters
                "sampling_rate": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "noise_level": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("signal_id",)
    FUNCTION = "generate_signal"
    CATEGORY = "signal/generators"
    OUTPUT_NODE = False
    
    def __init__(self):
        # Generate a unique ID for this node instance
        self.node_id = f"signal_gen_{str(uuid.uuid4())[:8]}"
        
        # Get registry singleton
        self.registry = SignalRegistry.get_instance()
        
        # Thread for live signal generation
        self.generation_thread = None
        self.stop_event = threading.Event()
        self.signal_metadata = None  # Store metadata about the current signal
        
        logger.info(f"Signal Generator Node {self.node_id} initialized")
    
    def __del__(self):
        """Clean up any running threads when the node is deleted"""
        self.stop_live_generation()
    
    def stop_live_generation(self):
        """Stop any ongoing live signal generation threads"""
        if self.generation_thread and self.generation_thread.is_alive():
            self.stop_event.set()
            self.generation_thread.join(timeout=2.0)  # Wait up to 2 seconds
            logger.info(f"Live signal generation stopped for node {self.node_id}")
    
    def generate_signal(self, id, mode, signal_type, sampling_rate, amplitude, noise_level):
        """
        Generate a single signal based on the selected mode and type
        
        Args:
            id: Unique identifier for the signal
            mode: "physiological" or "waveform"  
            signal_type: Type of signal to generate
            sampling_rate: Sample rate (samples per second)
            amplitude: Signal amplitude multiplier
            noise_level: Amount of noise (0.0-1.0)
            
        Returns:
            signal_id: The ID of the generated signal
        """
        # Stop any existing generation thread
        self.stop_live_generation()
        
        # Initial data duration in seconds
        duration = 10
        
        # Create the initial signal
        if mode == "physiological":
            # Validate signal type for physiological mode
            if signal_type not in ["EDA", "ECG", "RR"]:
                signal_type = "EDA"  # Default to EDA for physiological mode
                logger.warning(f"Invalid signal type '{signal_type}' for physiological mode, defaulting to EDA")
                
            self._generate_physiological_signal(signal_type, id, duration, sampling_rate, amplitude, noise_level)
        else:
            # Validate signal type for waveform mode
            valid_waveforms = ["sine", "square", "triangle", "sawtooth", "noise", "random"]
            if signal_type not in valid_waveforms:
                signal_type = "sine"  # Default to sine wave for waveform mode
                logger.warning(f"Invalid signal type '{signal_type}' for waveform mode, defaulting to sine")
                
            self._generate_waveform_signal(signal_type, id, duration, sampling_rate, amplitude, noise_level)
        
        # Start continuous live generation in background thread
        self._start_live_generation(mode, signal_type, id, sampling_rate, amplitude, noise_level)
        
        return (id,)
    
    def _generate_physiological_signal(self, signal_type, signal_id, duration, sampling_rate, amplitude, noise_level):
        """Generate a physiological signal (EDA, ECG, or RR)"""
        logger.info(f"Generating {signal_type} physiological signal with ID '{signal_id}'")
        
        # Determine number of samples
        num_samples = int(duration * sampling_rate)
        
        # Create time array
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate the appropriate physiological signal
        if signal_type == "EDA":
            # EDA: Skin conductance with slow drift and occasional responses
            baseline = 2.0 + 0.3 * np.sin(2 * np.pi * 0.005 * t)
            
            # Add skin conductance responses at random intervals
            scr_contribution = np.zeros_like(t)
            
            # Add a few random SCRs
            for _ in range(3):
                scr_time = np.random.uniform(1, duration-3)
                scr_amplitude = np.random.uniform(0.2, 0.8)
                scr_rise = np.random.uniform(1.0, 2.0)
                scr_decay = np.random.uniform(3.0, 7.0)
                
                idx = np.where((t >= scr_time) & (t <= scr_time + scr_rise + scr_decay))[0]
                for i in idx:
                    time_since_onset = t[i] - scr_time
                    if time_since_onset <= scr_rise:
                        # Rising phase (using sine function for smooth onset)
                        normalized_time = time_since_onset / scr_rise
                        scr_contribution[i] = scr_amplitude * np.sin(np.pi * normalized_time / 2)
                    else:
                        # Decay phase (exponential decay)
                        decay_time = time_since_onset - scr_rise
                        decay_factor = np.exp(-decay_time / (scr_decay * 0.5))
                        scr_contribution[i] = scr_amplitude * decay_factor
            
            signal = baseline + scr_contribution
            color = (0, 255, 0)  # Green
            
        elif signal_type == "ECG":
            # ECG: Cardiac signal with characteristic PQRST waves
            heart_rate = 60 + 5 * np.sin(2 * np.pi * 0.05 * t)  # Varying heart rate
            signal = np.zeros_like(t)
            
            for i, time in enumerate(t):
                # Current heart rate in Hz
                hr_hz = heart_rate[i] / 60.0
                
                # Phase within cardiac cycle
                phase = (time * hr_hz) % 1.0
                
                # Generate PQRST complex
                p_wave = 0.15 * np.exp(-((phase - 0.1) ** 2) / 0.002)
                q_wave = -0.1 * np.exp(-((phase - 0.2) ** 2) / 0.0005)
                r_wave = 1.0 * np.exp(-((phase - 0.22) ** 2) / 0.0002)
                s_wave = -0.3 * np.exp(-((phase - 0.24) ** 2) / 0.0005)
                t_wave = 0.3 * np.exp(-((phase - 0.35) ** 2) / 0.003)
                
                signal[i] = p_wave + q_wave + r_wave + s_wave + t_wave
            
            signal = signal * amplitude
            color = (255, 0, 0)  # Red
            
        else:  # RR signal
            # RR: Respiratory signal with realistic breathing pattern
            breathing_rate = 15 + 2 * np.sin(2 * np.pi * 0.01 * t)  # ~15 breaths per minute with variation
            signal = np.zeros_like(t)
            
            for i, time in enumerate(t):
                br_hz = breathing_rate[i] / 60.0
                phase = (time * br_hz) % 1.0
                
                # Asymmetric breathing pattern (inhale is shorter than exhale)
                if phase < 0.4:
                    normalized_phase = phase / 0.4
                    breathing = np.sin(np.pi * normalized_phase / 2)
                else:
                    normalized_phase = (phase - 0.4) / 0.6
                    breathing = 1.0 - normalized_phase
                
                signal[i] = breathing
            
            signal = (signal * amplitude) + 60.0  # Baseline around 60
            color = (255, 165, 0)  # Orange
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
            signal = signal + noise
        
        # Store signal metadata
        self.signal_metadata = {
            'signal_id': signal_id,
            'signal_type': signal_type,
            'mode': 'physiological',
            'color': color,
            'sampling_rate': sampling_rate,
            'last_x': t[-1],
            'amplitude': amplitude,
            'noise_level': noise_level
        }
        
        # Register with the registry
        self.registry.register_signal(
            signal_id=signal_id,
            signal_data=signal,
            metadata={
                'color': color,
                'source_node': self.node_id,
                'signal_type': signal_type, 
                'x_values': t,
                'sampling_rate': sampling_rate,
                'generator_mode': 'physiological'
            }
        )
        
        logger.info(f"Registered {signal_type} signal as '{signal_id}' with {len(signal)} samples")
        return True
    
    def _generate_waveform_signal(self, waveform_type, signal_id, duration, sampling_rate, amplitude, noise_level):
        """Generate a standard waveform signal"""
        logger.info(f"Generating {waveform_type} waveform signal with ID '{signal_id}'")
        
        # Calculate the number of samples needed
        num_samples = int(duration * sampling_rate)
        
        # Generate time values
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Default frequency of 1Hz
        frequency = 1.0
        
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
            
        elif waveform_type == "noise":
            signal = amplitude * np.random.randn(num_samples)
            color = (150, 150, 150)  # Gray
            
        else:  # Random walk
            steps = np.random.randn(num_samples)
            signal = np.cumsum(steps) * 0.1 * amplitude
            # Normalize to keep within amplitude range
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6) * 2 * amplitude - amplitude
            color = (100, 200, 200)  # Teal
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * amplitude, len(signal))
            signal = signal + noise
        
        # Store signal metadata
        self.signal_metadata = {
            'signal_id': signal_id,
            'signal_type': waveform_type,
            'mode': 'waveform',
            'color': color,
            'sampling_rate': sampling_rate,
            'frequency': frequency,
            'last_x': t[-1],
            'amplitude': amplitude,
            'noise_level': noise_level
        }
        
        # Register with the registry
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
        
        logger.info(f"Registered {waveform_type} signal as '{signal_id}' with {len(signal)} samples")
        return True
    
    def _start_live_generation(self, mode, signal_type, signal_id, sampling_rate, amplitude, noise_level):
        """Start a thread for live signal generation"""
        # Reset the stop event
        self.stop_event = threading.Event()
        
        # Start the generation thread
        self.generation_thread = threading.Thread(
            target=self._live_signal_generation,
            args=(mode, signal_type, signal_id, sampling_rate, amplitude, noise_level),
            daemon=True
        )
        
        self.generation_thread.start()
        logger.info(f"Started live generation thread for {signal_type} signal with ID '{signal_id}'")
    
    def _live_signal_generation(self, mode, signal_type, signal_id, sampling_rate, amplitude, noise_level):
        """Generate signal continuously in a background thread"""
        try:
            # Loop until stopped
            while not self.stop_event.is_set():
                # Get current signal data
                signal_data = self.registry.get_signal(signal_id)
                if signal_data is None:
                    logger.warning(f"Signal {signal_id} not found in registry, waiting...")
                    time.sleep(0.5)
                    continue
                    
                metadata = self.registry.get_signal_metadata(signal_id)
                if metadata is None:
                    logger.warning(f"Metadata for signal {signal_id} not found, waiting...")
                    time.sleep(0.5)
                    continue
                
                # Generate a small segment of new data (1/10 second)
                new_samples = max(1, int(sampling_rate / 10))
                
                # Last x value from existing data
                last_x = metadata.get('x_values', [0])[-1] if 'x_values' in metadata else 0
                new_t = np.linspace(last_x, last_x + new_samples/sampling_rate, new_samples)
                
                # Generate new data based on signal type and mode
                new_signal = self._generate_live_segment(mode, signal_type, new_t, amplitude, noise_level, signal_data)
                
                if new_signal is not None:
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
                    self.registry.register_signal(signal_id, updated_signal, metadata)
                
                # Sleep before next update
                time.sleep(0.1)  # Update 10 times per second
                
        except Exception as e:
            logger.error(f"Error in live signal generation: {str(e)}")
    
    def _generate_live_segment(self, mode, signal_type, t, amplitude, noise_level, last_signal):
        """Generate a small segment of signal for live updates"""
        new_signal = None
        
        if mode == "physiological":
            if signal_type == "EDA":
                # Continue from last value with small changes
                last_value = last_signal[-1] if len(last_signal) > 0 else 2.0
                drift = np.random.normal(0, 0.01, len(t))
                trend = np.linspace(0, 0.01 * np.random.choice([-1, 1]), len(t))
                new_signal = last_value + drift + trend
                
            elif signal_type == "ECG":
                # ECG with a fixed heart rate for the segment
                heart_rate = 60 + 5 * np.sin(2 * np.pi * 0.05 * t[0])  # Basic variation
                hr_hz = heart_rate / 60.0
                
                new_signal = np.zeros_like(t)
                for i, time in enumerate(t):
                    phase = (time * hr_hz) % 1.0
                    
                    p_wave = 0.15 * np.exp(-((phase - 0.1) ** 2) / 0.002)
                    q_wave = -0.1 * np.exp(-((phase - 0.2) ** 2) / 0.0005)
                    r_wave = 1.0 * np.exp(-((phase - 0.22) ** 2) / 0.0002)
                    s_wave = -0.3 * np.exp(-((phase - 0.24) ** 2) / 0.0005)
                    t_wave = 0.3 * np.exp(-((phase - 0.35) ** 2) / 0.003)
                    
                    new_signal[i] = p_wave + q_wave + r_wave + s_wave + t_wave
                
                new_signal = new_signal * amplitude
                
            elif signal_type == "RR":
                # Respiratory rate with slow variations
                breathing_rate = 15 + 2 * np.sin(2 * np.pi * 0.01 * t[0])
                br_hz = breathing_rate / 60.0
                
                new_signal = np.zeros_like(t)
                for i, time in enumerate(t):
                    phase = (time * br_hz) % 1.0
                    
                    if phase < 0.4:
                        normalized_phase = phase / 0.4
                        breathing = np.sin(np.pi * normalized_phase / 2)
                    else:
                        normalized_phase = (phase - 0.4) / 0.6
                        breathing = 1.0 - normalized_phase
                    
                    new_signal[i] = breathing
                
                new_signal = (new_signal * amplitude) + 60.0  # Baseline around 60
        
        else:  # Waveform mode
            frequency = 1.0  # Fixed frequency for simplicity
            
            if signal_type == "sine":
                new_signal = amplitude * np.sin(2 * np.pi * frequency * t)
                
            elif signal_type == "square":
                new_signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
                
            elif signal_type == "triangle":
                new_signal = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
                
            elif signal_type == "sawtooth":
                new_signal = amplitude * ((2 * (frequency * t - np.floor(0.5 + frequency * t))))
                
            elif signal_type == "noise":
                new_signal = amplitude * np.random.randn(len(t))
                
            else:  # Random walk
                # Continue from last value
                last_value = last_signal[-1] if len(last_signal) > 0 else 0
                steps = np.random.randn(len(t)) * 0.1 * amplitude
                random_walk = np.cumsum(steps)
                new_signal = random_walk + last_value
                # Keep within amplitude bounds
                new_signal = np.clip(new_signal, -amplitude, amplitude)
        
        # Add noise
        if noise_level > 0 and new_signal is not None:
            noise = np.random.normal(0, noise_level * amplitude * 0.1, len(new_signal))
            new_signal = new_signal + noise
            
        return new_signal

# Node registration
NODE_CLASS_MAPPINGS = {
    "UnifiedSignalGenerator": UnifiedSignalGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedSignalGenerator": "🌊 Signal Generator"
}
