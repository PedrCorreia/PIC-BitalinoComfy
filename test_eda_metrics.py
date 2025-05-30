#!/usr/bin/env python3
"""
Test script to verify EDA node SCR frequency calculation and metrics registry.
"""

import time
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.registry.signal_registry import SignalRegistry
from comfy.phy.eda import EDANode

def generate_synthetic_eda(duration=30, fs=1000):
    """Generate synthetic EDA data with known SCR events"""
    t = np.linspace(0, duration, int(duration * fs))
    
    # Base tonic level (SCL)
    scl_base = 5.0  # μS
    scl_trend = 0.5 * np.sin(0.1 * t)  # slow variations
    
    # Add SCR events (known positions)
    scr_times = [5, 10, 15, 20, 25]  # seconds
    scr_responses = []
    
    for scr_time in scr_times:
        # SCR response: exponential rise and decay
        response_start = int(scr_time * fs)
        response_duration = int(5 * fs)  # 5 second duration
        
        if response_start + response_duration < len(t):
            response_t = t[response_start:response_start + response_duration]
            # Rise phase (0.5s) + decay phase (4.5s)
            rise = np.exp((response_t - scr_time) * 2) * (response_t <= scr_time + 0.5)
            decay = np.exp(-(response_t - scr_time - 0.5) * 0.7) * (response_t > scr_time + 0.5)
            scr_response = 1.5 * (rise + decay)  # 1.5 μS amplitude
            scr_responses.append((response_start, scr_response))
    
    # Combine all components
    eda_signal = scl_base + scl_trend
    for start_idx, response in scr_responses:
        end_idx = min(start_idx + len(response), len(eda_signal))
        eda_signal[start_idx:end_idx] += response[:end_idx-start_idx]
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(eda_signal))
    eda_signal += noise
    
    return t, eda_signal, scr_times

def test_eda_metrics():
    """Test EDA node metrics calculation"""
    print("Testing EDA node SCR frequency metrics...")
    
    # Generate synthetic data
    t, eda_signal, expected_scr_times = generate_synthetic_eda(duration=30, fs=1000)
    print(f"Generated {len(eda_signal)} samples over {t[-1]:.1f} seconds")
    print(f"Expected SCR events at: {expected_scr_times} seconds")
    
    # Register synthetic signal
    registry = SignalRegistry.get_instance()
    signal_data = {
        "t": t.tolist(),
        "v": eda_signal.tolist()
    }
    registry.register_signal("TEST_EDA_INPUT", signal_data, {
        "id": "TEST_EDA_INPUT",
        "type": "synthetic_eda",
        "fs": 1000
    })
    
    # Create EDA node
    eda_node = EDANode()
    
    # Process the signal
    print("Processing EDA signal...")
    scl, sck, signal_id = eda_node.process_eda(
        input_signal_id="TEST_EDA_INPUT",
        show_peaks=True,
        output_signal_id="TEST_EDA_OUTPUT",
        enabled=True
    )
    
    print(f"Initial SCL: {scl:.3f}, SCK: {sck:.3f}")
    
    # Wait for background processing
    print("Waiting for background processing...")
    time.sleep(3)
    
    # Check if metrics are being registered
    print("\nChecking metrics registry...")
    
    # Check SCL metric
    scl_metric = registry.get_signal("SCL_METRIC")
    if scl_metric:
        print(f"SCL_METRIC found: {len(scl_metric.get('v', []))} data points")
        if len(scl_metric.get('v', [])) > 0:
            print(f"Latest SCL value: {scl_metric['v'][-1]:.3f}")
    else:
        print("SCL_METRIC not found")
    
    # Check SCK metric
    sck_metric = registry.get_signal("SCK_METRIC")
    if sck_metric:
        print(f"SCK_METRIC found: {len(sck_metric.get('v', []))} data points")
        if len(sck_metric.get('v', [])) > 0:
            print(f"Latest SCK value: {sck_metric['v'][-1]:.3f}")
    else:
        print("SCK_METRIC not found")
    
    # Check SCR frequency metric
    scr_metric = registry.get_signal("SCR_METRIC")
    if scr_metric:
        print(f"SCR_METRIC found: {len(scr_metric.get('v', []))} data points")
        if len(scr_metric.get('v', [])) > 0:
            print(f"Latest SCR frequency: {scr_metric['v'][-1]:.1f} events/min")
    else:
        print("SCR_METRIC not found")
    
    # Check processed signal
    processed_signal = registry.get_signal("TEST_EDA_OUTPUT")
    if processed_signal:
        print(f"Processed signal found: {len(processed_signal.get('v', []))} data points")
        metadata = registry.get_signal_metadata("TEST_EDA_OUTPUT")
        if metadata:
            print(f"SCR frequency in metadata: {metadata.get('scr_frequency', 'N/A')}")
            print(f"SCR peaks detected: {len(metadata.get('scr_peak_timestamps', []))}")
            print(f"Peak timestamps: {metadata.get('scr_peak_timestamps', [])}")
    else:
        print("Processed signal not found")
    
    # Stop background processing
    if hasattr(eda_node, '_stop_flags') and "TEST_EDA_OUTPUT" in eda_node._stop_flags:
        eda_node._stop_flags["TEST_EDA_OUTPUT"][0] = True
        print("Stopped background processing")
    
    return True

if __name__ == "__main__":
    test_eda_metrics()
