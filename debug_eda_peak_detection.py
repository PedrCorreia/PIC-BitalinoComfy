#!/usr/bin/env python3
"""
Debug script for EDA peak detection using matplotlib visualization.
This script tests and visualizes the EDA peak detection algorithm to verify it's working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

from src.phy.eda_signal_processing import EDA
from src.utils.signal_processing import NumpySignalProcessor

def create_synthetic_eda_signal(fs=1000, duration=20):
    """
    Create a synthetic EDA signal with known peaks for testing.
    
    Parameters:
    - fs: Sampling frequency in Hz
    - duration: Duration in seconds
    
    Returns:
    - t: Time vector
    - eda_signal: Synthetic EDA signal
    - true_peak_times: Known peak locations for validation
    """
    t = np.linspace(0, duration, fs * duration)
    
    # Create baseline tonic component (slowly increasing)
    tonic_baseline = 2.0 + 0.3 * t / duration
    
    # Add some low-frequency drift
    tonic_drift = 0.2 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz drift
    
    # Create phasic responses (SCRs) at known times
    true_peak_times = [2, 5, 8, 11, 14, 17]  # Known peak locations
    phasic_responses = np.zeros_like(t)
    
    for peak_time in true_peak_times:
        # Create realistic SCR shape (fast rise, slow decay)
        response_mask = t >= peak_time
        if np.any(response_mask):
            # Exponential rise and decay
            time_from_peak = t[response_mask] - peak_time
            # SCR shape: quick rise (0.5s) then slow decay (3s)
            amplitude = np.random.uniform(0.3, 0.8)  # Random amplitude
            rise_time = 0.5
            decay_time = 3.0
            
            scr_shape = amplitude * np.exp(-time_from_peak / decay_time) * (1 - np.exp(-time_from_peak / rise_time))
            phasic_responses[response_mask] += scr_shape
    
    # Add realistic noise
    noise = np.random.normal(0, 0.05, len(t))
    
    # Combine all components
    eda_signal = tonic_baseline + tonic_drift + phasic_responses + noise
    
    return t, eda_signal, true_peak_times

def debug_peak_detection():
    """
    Main debugging function for EDA peak detection.
    """
    print("EDA Peak Detection Debug Script")
    print("=" * 50)
    
    # Create synthetic signal
    t, eda_signal, true_peak_times = create_synthetic_eda_signal()
    fs = 1000
    
    print(f"Created synthetic EDA signal:")
    print(f"  Duration: {len(t)/fs:.1f} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  True peak times: {true_peak_times}")
    print(f"  Signal range: {np.min(eda_signal):.3f} to {np.max(eda_signal):.3f} μS")
    
    # Process the signal step by step
    print("\nProcessing signal...")
    
    # 1. Preprocess the signal
    preprocessed = EDA.preprocess_signal(eda_signal, fs)
    print(f"  Preprocessed signal range: {np.min(preprocessed):.3f} to {np.max(preprocessed):.3f}")
    
    # 2. Extract tonic and phasic components
    tonic, phasic = EDA.extract_tonic_phasic(preprocessed, fs)
    print(f"  Tonic range: {np.min(tonic):.3f} to {np.max(tonic):.3f}")
    print(f"  Phasic range: {np.min(phasic):.3f} to {np.max(phasic):.3f}")
    
    # 3. Test different thresholds for peak detection
    thresholds = [0.01, 0.05, 0.1, 0.2]
    detection_results = {}
    
    for threshold in thresholds:
        try:
            detected_events, envelope = EDA.detect_events(phasic, fs, threshold=threshold)
            detected_times = t[detected_events] if len(detected_events) > 0 else []
            detection_results[threshold] = {
                'events': detected_events,
                'times': detected_times,
                'envelope': envelope,
                'count': len(detected_events)
            }
            print(f"  Threshold {threshold:.2f}: {len(detected_events)} peaks at times {detected_times}")
        except Exception as e:
            print(f"  Threshold {threshold:.2f}: ERROR - {e}")
            detection_results[threshold] = {'events': [], 'times': [], 'envelope': np.zeros_like(phasic), 'count': 0}
    
    # Choose best threshold (closest to expected number of peaks)
    expected_peaks = len(true_peak_times)
    best_threshold = min(thresholds, key=lambda th: abs(detection_results[th]['count'] - expected_peaks))
    best_result = detection_results[best_threshold]
    
    print(f"\nBest threshold: {best_threshold:.2f} (detected {best_result['count']} peaks, expected {expected_peaks})")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.suptitle('EDA Peak Detection Debug Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Original signal with true peaks
    axes[0].plot(t, eda_signal, 'b-', linewidth=1, label='Original EDA Signal')
    for peak_time in true_peak_times:
        axes[0].axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0].set_title('Original EDA Signal with True Peak Locations')
    axes[0].set_ylabel('Amplitude (μS)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Preprocessed signal
    axes[1].plot(t, preprocessed, 'g-', linewidth=1, label='Preprocessed Signal')
    axes[1].set_title('Preprocessed EDA Signal')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
      # Plot 3: Tonic and Phasic components with peaks
    axes[2].plot(t, tonic, 'g-', linewidth=2, label='Tonic Component')
    axes[2].plot(t, phasic, 'orange', linewidth=1.5, label='Phasic Component')
      # Show detected peaks on phasic component with prominent markers
    if len(best_result['events']) > 0:
        # Draw large red X markers for peaks
        axes[2].scatter(best_result['times'], phasic[best_result['events']], 
                       color='red', marker='x', s=200, linewidth=4,
                       label=f'Detected Peaks ({best_result["count"]})', zorder=10)
        
        # Add vertical lines from x-axis to peak points for clarity
        for i, peak_idx in enumerate(best_result['events']):
            axes[2].axvline(x=best_result['times'][i], color='red', alpha=0.3, 
                           linestyle='-', linewidth=2, zorder=5)
            # Add small circles at the peak points for additional visibility
            axes[2].scatter(best_result['times'][i], phasic[peak_idx], 
                           color='white', marker='o', s=50, linewidth=2,
                           edgecolor='red', zorder=11)
    
    # Mark true peak locations for comparison
    for peak_time in true_peak_times:
        axes[2].axvline(x=peak_time, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    axes[2].set_title('Tonic and Phasic Components with Peak Detection')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
      # Plot 4: Peak detection results for different thresholds
    axes[3].plot(t, phasic, 'orange', linewidth=1.5, label='Phasic Component', alpha=0.7)
    
    colors = ['red', 'blue', 'green', 'purple']
    markers = ['X', 'o', 's', '^']  # Changed first marker to capital X for prominence
    sizes = [150, 80, 80, 80]  # Larger size for best threshold (red)
    
    for i, threshold in enumerate(thresholds):
        result = detection_results[threshold]
        if len(result['events']) > 0:
            axes[3].scatter(result['times'], phasic[result['events']], 
                          color=colors[i], marker=markers[i], s=sizes[i], 
                          linewidth=2,
                          label=f'Threshold {threshold:.2f} ({result["count"]} peaks)',
                          zorder=5)
            
            # Highlight the best threshold result with additional emphasis
            if threshold == best_threshold:
                axes[3].scatter(result['times'], phasic[result['events']], 
                              facecolors='none', edgecolors='black', 
                              marker='o', s=200, linewidth=3, zorder=6)
    
    axes[3].set_title('Peak Detection Results for Different Thresholds')
    axes[3].set_ylabel('Phasic Amplitude')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
      # Plot 5: Best result with envelope - Enhanced peak visualization
    axes[4].plot(t, phasic, 'orange', linewidth=2, label='Phasic Component')
    if len(best_result['envelope']) > 0:
        axes[4].plot(t, best_result['envelope'], 'cyan', linewidth=1.5, alpha=0.8, label='Validation Envelope')
    
    # Mark detected peaks with prominent styling
    if len(best_result['events']) > 0:
        # Large red X markers
        axes[4].scatter(best_result['times'], phasic[best_result['events']], 
                       color='red', marker='X', s=200, linewidth=4,
                       label=f'Detected Peaks ({best_result["count"]})', zorder=10)
        
        # Add white circles with red borders for extra visibility
        axes[4].scatter(best_result['times'], phasic[best_result['events']], 
                       color='white', marker='o', s=100, linewidth=2,
                       edgecolor='red', zorder=11)
        
        # Add vertical stems to make peaks easier to identify
        for i, peak_idx in enumerate(best_result['events']):
            axes[4].plot([best_result['times'][i], best_result['times'][i]], 
                        [0, phasic[peak_idx]], 'red', alpha=0.5, linewidth=2, zorder=8)
    
    # Mark true peaks for comparison
    for peak_time in true_peak_times:
        axes[4].axvline(x=peak_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[4].set_title(f'Final Peak Detection (Threshold: {best_threshold:.2f})')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Phasic Amplitude')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'eda_peak_detection_debug.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Performance analysis
    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    if len(best_result['events']) > 0:
        detected_times = best_result['times']
        
        # Calculate detection accuracy
        matches = 0
        tolerance = 1.0  # 1 second tolerance
        
        for true_time in true_peak_times:
            # Check if any detected peak is within tolerance
            distances = [abs(det_time - true_time) for det_time in detected_times]
            if distances and min(distances) <= tolerance:
                matches += 1
        
        precision = matches / len(detected_times) if len(detected_times) > 0 else 0
        recall = matches / len(true_peak_times) if len(true_peak_times) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"True peaks: {len(true_peak_times)}")
        print(f"Detected peaks: {len(detected_times)}")
        print(f"Matched peaks: {matches}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1_score:.3f}")
        
        if f1_score > 0.8:
            print("✅ EXCELLENT: Peak detection is working very well!")
        elif f1_score > 0.6:
            print("✅ GOOD: Peak detection is working adequately.")
        elif f1_score > 0.4:
            print("⚠️  FAIR: Peak detection needs improvement.")
        else:
            print("❌ POOR: Peak detection has significant issues.")
    else:
        print("❌ CRITICAL: No peaks detected!")
    
    return True

def test_real_signal_processing():
    """
    Test with a more realistic EDA signal pattern.
    """
    print("\n" + "=" * 50)
    print("TESTING WITH REALISTIC SIGNAL PATTERNS")
    print("=" * 50)
    
    # Create more realistic signal with varying amplitudes and frequencies
    fs = 1000
    duration = 30
    t = np.linspace(0, duration, fs * duration)
    
    # Baseline with realistic EDA characteristics
    baseline = 2.5 + 0.1 * np.sin(2 * np.pi * 0.01 * t)  # Very slow baseline changes
    
    # Add stress response pattern (multiple SCRs in clusters)
    stress_periods = [(5, 8), (15, 18), (25, 28)]  # Stress periods
    signal = baseline.copy()
    
    for start_time, end_time in stress_periods:
        # Multiple SCRs during stress
        scr_times = np.arange(start_time, end_time, 1.5)  # SCR every 1.5 seconds
        for scr_time in scr_times:
            if scr_time < end_time:
                mask = t >= scr_time
                if np.any(mask):
                    time_from_scr = t[mask] - scr_time
                    amplitude = np.random.uniform(0.2, 0.6)
                    scr = amplitude * np.exp(-time_from_scr / 2.5) * (1 - np.exp(-time_from_scr / 0.3))
                    signal[mask] += scr
    
    # Add noise
    signal += np.random.normal(0, 0.03, len(signal))
    
    # Process the signal
    preprocessed = EDA.preprocess_signal(signal, fs)
    tonic, phasic = EDA.extract_tonic_phasic(preprocessed, fs)
    detected_events, envelope = EDA.detect_events(phasic, fs, threshold=0.1)
    
    print(f"Realistic signal analysis:")
    print(f"  Signal duration: {duration}s")
    print(f"  Detected {len(detected_events)} SCR events")
    print(f"  Event rate: {len(detected_events)/duration*60:.1f} events/minute")
    
    # Quick visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, 'b-', linewidth=1)
    plt.title('Realistic EDA Signal')
    plt.ylabel('Amplitude (μS)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, tonic, 'g-', linewidth=2, label='Tonic')
    plt.plot(t, phasic, 'orange', linewidth=1, label='Phasic')
    plt.title('Tonic and Phasic Components')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, phasic, 'orange', linewidth=1, label='Phasic')
    if len(detected_events) > 0:
        plt.scatter(t[detected_events], phasic[detected_events], 
                   color='red', marker='x', s=60, linewidth=2, 
                   label=f'Detected SCRs ({len(detected_events)})')
    plt.title('Peak Detection Results')
    plt.xlabel('Time (s)')
    plt.ylabel('Phasic Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_eda_test.png', dpi=150, bbox_inches='tight')
    print(f"Realistic signal test saved as: realistic_eda_test.png")
    plt.show()

if __name__ == "__main__":
    try:
        print("Starting EDA Peak Detection Debug Analysis...")
        success = debug_peak_detection()
        
        if success:
            test_real_signal_processing()
            print("\n✅ Debug analysis completed successfully!")
        else:
            print("\n❌ Debug analysis failed!")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
