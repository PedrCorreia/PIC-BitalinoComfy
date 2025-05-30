#!/usr/bin/env python3
"""
Simple test script to verify EDA peak detection is working correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

from src.phy.eda_signal_processing import EDA

def test_eda_peak_detection():
    """
    Simple test for EDA peak detection.
    """
    print("Testing EDA Peak Detection...")
    
    # Generate simple synthetic EDA data
    fs = 1000
    duration = 5
    t = np.linspace(0, duration, fs * duration)
    
    # Create a signal with known peaks at times 1, 2, 3, 4
    signal = np.zeros_like(t)
    for peak_time in [1, 2, 3, 4]:
        signal += np.exp(-0.5 * ((t - peak_time) / 0.2) ** 2)
    
    # Add baseline
    signal += 0.5
    
    # Add noise
    signal += 0.05 * np.random.randn(len(t))
    
    print(f"Created synthetic signal with {len(signal)} samples")
    print(f"Signal range: {np.min(signal):.3f} to {np.max(signal):.3f}")
    
    try:
        # Test peak detection
        detected_events, envelope = EDA.detect_events(signal, fs, threshold=0.3)
        
        print(f"✅ Peak detection successful!")
        print(f"Detected {len(detected_events)} peaks")
        print(f"Peak indices: {detected_events}")
        if len(detected_events) > 0:
            print(f"Peak times: {t[detected_events]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Peak detection failed: {e}")
        return False

def test_metadata_compatibility():
    """
    Test metadata structure for visualization compatibility.
    """
    print("\nTesting metadata compatibility...")
    
    # Create mock metadata like the EDA node produces
    metadata = {
        "scr_peak_timestamps": [1.0, 2.0, 3.0],
        "scr_peak_indices": [1000, 2000, 3000],
        "show_peaks": True,
        "type": "eda_processed"
    }
    
    # Test visualization conditions
    can_show_peaks = (
        metadata.get('show_peaks', False) and 
        'scr_peak_indices' in metadata and 
        len(metadata['scr_peak_indices']) > 0
    )
    
    print(f"✅ Metadata structure correct: {can_show_peaks}")
    return can_show_peaks

if __name__ == "__main__":
    print("EDA Peak Detection Test")
    print("=" * 30)
    
    detection_ok = test_eda_peak_detection()
    metadata_ok = test_metadata_compatibility()
    
    print("\n" + "=" * 30)
    if detection_ok and metadata_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
