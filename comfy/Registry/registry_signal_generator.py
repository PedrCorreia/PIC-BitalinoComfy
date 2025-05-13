import sys
import os
import uuid
import numpy as np
import torch
import random
import importlib
import math
from typing import Dict, List, Tuple, Union

# Import from registry
from ...src.plot.signal_registry import SignalRegistry

class RegistrySignalGenerator:
    """
    Signal generator that registers signals in the centralized registry.
    This node creates synthetic signals and makes them available to visualization nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_type": (["sine", "square", "triangle", "sawtooth", "noise", "ecg", "random"], {"default": "sine"}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "sample_rate": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "noise_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "signal_id": ("STRING", {"default": "signal_1", "multiline": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("signal_id", "max_value", "samples_count")
    FUNCTION = "generate_signal"
    CATEGORY = "signal/generators"
    OUTPUT_NODE = True
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"registry_generator_{str(uuid.uuid4())[:8]}"
        
        # Get registry singleton
        self.registry = SignalRegistry.get_instance()
        print(f"[Registry Generator] Node {self.node_id} initialized")
    
    def generate_signal(self, signal_type, frequency, amplitude, duration, sample_rate, noise_level, signal_id="signal_1", seed=-1):
        """
        Generate a signal and register it in the registry
        
        Args:
            signal_type: Type of signal to generate
            frequency: Frequency of the signal (Hz)
            amplitude: Amplitude of the signal
            duration: Duration of the signal (seconds)
            sample_rate: Sample rate (samples per second)
            noise_level: Amount of noise to add (0.0-1.0)
            signal_id: ID to use for this signal in the registry
            seed: Random seed (-1 for random)
            
        Returns:
            signal_id: The ID of the registered signal
            max_value: The maximum value in the signal
            samples_count: Number of samples in the signal
        """
        print(f"[Registry Generator] Generating {signal_type} signal with ID: {signal_id}")
          # Set random seed if provided
        if seed != -1:
            # Ensure seed is within numpy's valid range (0 to 2^32-1)
            valid_seed = abs(seed) % (2**32)
            random.seed(valid_seed)
            np.random.seed(valid_seed)
        
        # Calculate number of samples
        num_samples = int(duration * sample_rate)
        
        # Generate time array
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate the base signal
        if signal_type == "sine":
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        elif signal_type == "square":
            signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif signal_type == "triangle":
            signal = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - amplitude
        elif signal_type == "sawtooth":
            signal = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif signal_type == "noise":
            signal = amplitude * np.random.randn(num_samples)
        elif signal_type == "ecg":
            # Simple ECG-like pattern
            signal = self._generate_ecg_signal(t, amplitude, frequency)
        elif signal_type == "random":
            # Random combination of signal types
            choices = ["sine", "square", "triangle", "sawtooth"]
            selected = random.choice(choices)
            if selected == "sine":
                signal = amplitude * np.sin(2 * np.pi * frequency * t)
            elif selected == "square":
                signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
            elif selected == "triangle":
                signal = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - amplitude
            else:  # sawtooth
                signal = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
        else:
            # Default to sine if unknown type
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add noise if requested
        if noise_level > 0:
            noise = np.random.randn(num_samples) * amplitude * noise_level
            signal = signal + noise
        
        # Create processed version of this signal for twin view
        processed_signal = self._process_signal(signal, signal_type)
        
        # Metadata to store with the signal
        metadata = {
            'type': signal_type,
            'frequency': frequency,
            'amplitude': amplitude,
            'duration': duration,
            'sample_rate': sample_rate,
            'noise_level': noise_level,
            'created_by': self.node_id,
            'color': self._get_color_for_signal_type(signal_type)
        }
        
        # Register signal and processed version in the registry
        self.registry.register_signal(signal_id, signal, metadata)
        self.registry.register_signal(f"{signal_id}_processed", processed_signal, {
            **metadata,
            'processed': True,
            'color': (0, 180, 220)  # Blue for processed signals
        })
        
        print(f"[Registry Generator] Signal {signal_id} registered with {len(signal)} samples")
        
        # Return the signal ID and some metadata
        return (signal_id, float(np.max(np.abs(signal))), len(signal))
    
    def _generate_ecg_signal(self, t, amplitude, frequency):
        """Generate a synthetic ECG-like signal"""
        # Period of the ECG cycle
        period = 1.0 / frequency
        
        # Initialize signal
        ecg = np.zeros_like(t)
        
        # Create each beat
        for i in range(int(t[-1] / period) + 1):
            # Time within the current beat
            beat_time = t - i * period
            
            # Only process times within the current beat period
            mask = (beat_time >= 0) & (beat_time < period)
            
            # P wave (atrial depolarization)
            p_center = 0.2 * period
            p_width = 0.08 * period
            ecg += mask * 0.25 * amplitude * np.exp(-(beat_time - p_center)**2 / (2 * p_width**2))
            
            # QRS complex
            q_time = 0.35 * period
            qrs_width = 0.03 * period
            ecg += mask * -0.3 * amplitude * np.exp(-(beat_time - q_time)**2 / (2 * (qrs_width/2)**2))
            
            r_time = 0.4 * period
            ecg += mask * 1.0 * amplitude * np.exp(-(beat_time - r_time)**2 / (2 * qrs_width**2))
            
            s_time = 0.45 * period
            ecg += mask * -0.4 * amplitude * np.exp(-(beat_time - s_time)**2 / (2 * (qrs_width/2)**2))
            
            # T wave (ventricular repolarization)
            t_center = 0.6 * period
            t_width = 0.1 * period
            ecg += mask * 0.35 * amplitude * np.exp(-(beat_time - t_center)**2 / (2 * t_width**2))
        
        return ecg
    
    def _process_signal(self, signal, signal_type):
        """
        Process a signal to demonstrate twin view functionality
        This is a simple example - in real use, more sophisticated processing would be applied
        """
        # Copy signal to avoid modifying the original
        processed = signal.copy()
        
        if signal_type == "noise":
            # For noise, apply smoothing
            kernel_size = min(25, len(signal) // 10)
            if kernel_size % 2 == 0:  # Ensure odd kernel size
                kernel_size += 1
                
            if kernel_size >= 3:
                kernel = np.ones(kernel_size) / kernel_size
                processed = np.convolve(signal, kernel, mode='same')
        else:
            # For other signals, apply filtering to remove high frequencies
            from scipy import signal as sig
            
            # Simple low-pass filter
            b, a = sig.butter(3, 0.2, 'low')
            processed = sig.filtfilt(b, a, signal)
            
            # Add slight phase shift for visualization
            processed = np.roll(processed, len(signal) // 20)
            
        return processed
    
    def _get_color_for_signal_type(self, signal_type):
        """Get a consistent color for a signal type"""
        color_map = {
            "sine": (220, 180, 0),      # Gold
            "square": (220, 80, 80),    # Red
            "triangle": (80, 220, 80),  # Green
            "sawtooth": (220, 120, 20), # Orange
            "noise": (180, 180, 180),   # Gray
            "ecg": (220, 20, 60),       # Crimson
            "random": (180, 100, 220)   # Purple
        }
        
        return color_map.get(signal_type, (220, 180, 0))  # Default to gold

# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "RegistrySignalGenerator": RegistrySignalGenerator
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegistrySignalGenerator": "Registry Signal Generator"
}
