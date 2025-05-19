"""
Signal generator utility for PlotUnit visualization system.

This module provides functions for generating test signals for visualization 
in the PlotUnit system, including sine, square, sawtooth and other waveforms.
"""

import numpy as np
import time

def generate_sine_wave(amplitude=1.0, frequency=1.0, phase=0.0, num_points=1000):
    """
    Generate a sine wave signal.
    
    Args:
        amplitude (float): Amplitude of the sine wave
        frequency (float): Frequency of the sine wave
        phase (float): Phase offset in radians
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Sine wave signal
    """
    t = np.linspace(0, 1.0, num_points)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def generate_square_wave(amplitude=1.0, frequency=1.0, duty_cycle=0.5, num_points=1000):
    """
    Generate a square wave signal.
    
    Args:
        amplitude (float): Amplitude of the square wave
        frequency (float): Frequency of the square wave
        duty_cycle (float): Duty cycle of the square wave (0.0-1.0)
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Square wave signal
    """
    t = np.linspace(0, 1.0, num_points)
    return amplitude * ((t * frequency) % 1.0 < duty_cycle) * 2 - amplitude

def generate_triangle_wave(amplitude=1.0, frequency=1.0, num_points=1000):
    """
    Generate a triangle wave signal.
    
    Args:
        amplitude (float): Amplitude of the triangle wave
        frequency (float): Frequency of the triangle wave
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Triangle wave signal
    """
    t = np.linspace(0, 1.0, num_points)
    return amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))

def generate_sawtooth_wave(amplitude=1.0, frequency=1.0, num_points=1000):
    """
    Generate a sawtooth wave signal.
    
    Args:
        amplitude (float): Amplitude of the sawtooth wave
        frequency (float): Frequency of the sawtooth wave
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Sawtooth wave signal
    """
    t = np.linspace(0, 1.0, num_points)
    return amplitude * ((2 * (frequency * t - np.floor(0.5 + frequency * t))))

def generate_inverted_square_wave(amplitude=1.0, frequency=1.0, duty_cycle=0.5, num_points=1000):
    """
    Generate an inverted square wave signal.
    
    Args:
        amplitude (float): Amplitude of the square wave
        frequency (float): Frequency of the square wave
        duty_cycle (float): Duty cycle of the square wave (0.0-1.0)
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Inverted square wave signal
    """
    return -generate_square_wave(amplitude, frequency, duty_cycle, num_points)

def generate_test_signals():
    """
    Generate a set of test signals for PlotUnit visualization.
    
    Returns:
        dict: Dictionary containing raw and processed test signals
    """
    num_points = 1000
    t = np.linspace(0, 20, num_points)  # 20 seconds buffer
    signals = {
        # Raw signals
        "raw_sine": {"t": t, "v": generate_sine_wave(amplitude=0.8, frequency=2.0, num_points=num_points)},
        "raw_square": {"t": t, "v": generate_square_wave(amplitude=0.9, frequency=1.0, num_points=num_points)},
        # Processed signals
        "inverted_square": {"t": t, "v": generate_inverted_square_wave(amplitude=0.9, frequency=1.0, num_points=num_points)},
        "sawtooth": {"t": t, "v": generate_sawtooth_wave(amplitude=0.7, frequency=2.0, num_points=num_points)},
        "triangle": {"t": t, "v": generate_triangle_wave(amplitude=0.8, frequency=1.5, num_points=num_points)}
    }
    return signals

def update_test_signals(data, delta_t=0.05):
    """
    Update test signals with time evolution.
    
    Args:
        data (dict): Dictionary containing current signals
        delta_t (float): Time step for evolution
        
    Returns:
        dict: Updated signals
    """
    num_points = 1000
    t = np.linspace(0, 20, num_points)  # 20 seconds buffer
    phase_shift = time.time() % (2 * np.pi)
    # Update raw signals
    data["raw_sine"] = {"t": t, "v": generate_sine_wave(amplitude=0.8, frequency=2.0, phase=phase_shift, num_points=num_points)}
    data["raw_square"] = {"t": t, "v": generate_square_wave(amplitude=0.9, frequency=1.0, num_points=num_points)}
    # Update processed signals
    data["inverted_square"] = {"t": t, "v": generate_inverted_square_wave(amplitude=0.9, frequency=1.0, num_points=num_points)}
    data["sawtooth"] = {"t": t, "v": generate_sawtooth_wave(amplitude=0.7, frequency=2.0, num_points=num_points)}
    data["triangle"] = {"t": t, "v": generate_triangle_wave(amplitude=0.8, frequency=1.5, num_points=num_points)}
    return data
