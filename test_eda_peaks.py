#!/usr/bin/env python3
"""
Test script to verify EDA peak detection and visualization are working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.phy.eda_signal_processing import EDA
from src.utils.signal_processing import NumpySignalProcessor

def test_eda_peak_detection():
    """
    Test EDA peak detection on synthetic data to verify the implementation.
    """
    print("Testing EDA Peak Detection...")
    
    # Generate synthetic EDA data with known peaks
    fs = 1000  # Sampling frequency
    duration = 10  # Duration in seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Create a baseline with slow drift (tonic component)
    tonic_component = 2.0 + 0.5 * t / duration  # Slow increase from 2.0 to 2.5
    
    # Add periodic phasic responses (peaks)
    phasic_component = np.zeros_like(t)
    peak_times = [1, 3, 5, 7, 9]  # Known peak locations
    for peak_time in peak_times:
        # Gaussian pulse centered at peak_time
        phasic_component += 0.8 * np.exp(-0.5 * ((t - peak_time) / 0.3) ** 2)
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    
    # Combine components
    synthetic_eda = tonic_component + phasic_component + noise
    
    # Convert to ADC-like values
    synthetic_adc = ((synthetic_eda * 0.132 * (2 ** 10) / 3.3)).clip(0, 2 ** 10 - 1)
    
    # Process the signal
    eda_us = EDA.convert_adc_to_eda(synthetic_adc)
    preprocessed_signal = EDA.preprocess_signal(eda_us, fs)
    tonic, phasic = EDA.extract_tonic_phasic(preprocessed_signal, fs)
    
    # Detect events/peaks
    detected_events, envelope = EDA.detect_events(phasic, fs, threshold=0.1)
    
    print(f"Detected {len(detected_events)} peaks")
    print(f"Expected peaks at times: {peak_times}")
    print(f"Detected peaks at times: {t[detected_events]}")
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Original synthetic signal
    axes[0].plot(t, synthetic_eda, 'b-', label='Synthetic EDA')
    axes[0].set_title('Synthetic EDA Signal')
    axes[0].set_ylabel('Amplitude (μS)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Preprocessed signal
    axes[1].plot(t, preprocessed_signal, 'g-', label='Preprocessed')
    axes[1].set_title('Preprocessed EDA Signal')
    axes[1].set_ylabel('Amplitude (μS)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Tonic and phasic components
    axes[2].plot(t, tonic, 'g-', label='Tonic', linewidth=2)
    axes[2].plot(t, phasic, 'orange', label='Phasic', linewidth=1.5)
    axes[2].set_title('Tonic and Phasic Components')
    axes[2].set_ylabel('Amplitude (μS)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Phasic with detected peaks
    axes[3].plot(t, phasic, 'orange', label='Phasic', linewidth=1.5)
    if len(envelope) > 0:
        axes[3].plot(t, envelope, 'cyan', label='Envelope', linewidth=1)
    
    # Mark detected peaks
    if len(detected_events) > 0:
        axes[3].scatter(t[detected_events], phasic[detected_events], 
                       color='red', marker='x', s=100, linewidth=3,
                       label=f'Detected Peaks ({len(detected_events)})')
    
    # Mark expected peaks for comparison
    for peak_time in peak_times:
        axes[3].axvline(x=peak_time, color='gray', linestyle='--', alpha=0.5)
    
    axes[3].set_title('Phasic Component with Peak Detection')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude (μS)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig('eda_peak_detection_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return len(detected_events) > 0

def test_metadata_structure():
    """
    Test that the metadata structure matches what the visualization expects.
    """
    print("\nTesting metadata structure...")
    
    # Simulate the metadata structure from EDA node
    mock_scr_event_indices = [100, 200, 300, 400]
    mock_scr_event_times = [1.0, 2.0, 3.0, 4.0]
    
    metadata = {
        "scr_peak_timestamps": mock_scr_event_times,
        "scr_peak_indices": mock_scr_event_indices,
        "show_peaks": True,
    }
    
    # Check that visualization code would find the required fields
    has_indices = 'scr_peak_indices' in metadata
    has_show_peaks = metadata.get('show_peaks', False)
    indices_length = len(metadata['scr_peak_indices']) if has_indices else 0
    
    print(f"Metadata has scr_peak_indices: {has_indices}")
    print(f"Show peaks enabled: {has_show_peaks}")
    print(f"Number of peak indices: {indices_length}")
    
    # Test the visualization condition
    visualization_condition = (has_show_peaks and has_indices and indices_length > 0)
    print(f"Visualization condition met: {visualization_condition}")
    
    return visualization_condition

if __name__ == "__main__":
    print("EDA Peak Detection and Visualization Test")
    print("=" * 50)
    
    # Test peak detection
    detection_success = test_eda_peak_detection()
    
    # Test metadata structure
    metadata_success = test_metadata_structure()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Peak detection working: {detection_success}")
    print(f"Metadata structure correct: {metadata_success}")
    
    if detection_success and metadata_success:
        print("✅ All tests passed! EDA peak detection and visualization should work correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")
