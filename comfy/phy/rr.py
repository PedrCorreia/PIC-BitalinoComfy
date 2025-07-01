from collections import deque
import numpy as np
import threading
import time
from ...src.test_src.rr_signal_processing import RR
from ...src.registry.signal_registry import SignalRegistry
from ...src.utils.signal_processing import NumpySignalProcessor
from ...src.utils.utils import Arousal

class RRNode:
    """
    Node for processing RR (Respiratory Rate) signals.
    Provides visualization-ready data and calculates respiration rate (breaths/min) using detected peaks.
    This is the ONLY node whose output should be used for RR_METRIC in MetricsView.
    """
    
    # Track background processing threads
    _processing_threads = {}
    _stop_flags = {}
#================================================================================================================================== Input Types ============================================================================================================================
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the RR node.
        - signal_deque: Deque of (timestamp, value) pairs.
        - viz_buffer_size: Buffer size for visualization data.
        - feature_buffer_size: Buffer size for feature extraction.
        - print_rr: Print RR value if True.
        """
        return {
            "required": {
                "input_signal_id": ("STRING", {"default": ""}),
                "show_peaks": ("BOOLEAN", {"default": True}),
                "output_signal_id": ("STRING", {"default": "RR_PROCESSED"}),
                "enabled": ("BOOLEAN", {"default": True})
            }
        }
        

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("Respiration_Rate", "Is_Peak", "Signal_ID","Arousal")
    FUNCTION = "process_rr"
    CATEGORY = "Pedro_PIC/🔬 Bio-Processing"    
    
    @staticmethod
    def validate_breathing_events(signal, fs):
        """
        Validates breathing events by finding alternating peaks (inspirations) and dips (expirations).
        - Finds peaks in the original signal.
        - Finds dips by inverting the signal.
        - Applies adaptive thresholds for better sensitivity.
        - Merges and sorts peaks and dips, ensuring an alternating sequence.
        - Returns only the validated peak indices.
        """
        if not isinstance(signal, np.ndarray) or signal.size < 2:
            return np.array([], dtype=int)
            
        # Calculate signal statistics for adaptive thresholds
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_range = np.max(signal) - np.min(signal)
        
        # Determine if the signal has low amplitude (needs more sensitive detection)
        is_low_amplitude = signal_range < 0.5  # Adjust this threshold as needed
        
        # 1. Find peaks (inspirations) with adaptive threshold
        # Lower threshold (0.15 instead of 0.2) for more sensitivity
        # For low amplitude signals, use even lower threshold (0.1)
        if is_low_amplitude:
            peak_thresh = 0.1 * np.max(signal) if np.max(signal) > 0 else 0
            # Alternative threshold based on standard deviation
            std_peak_thresh = signal_mean + 1.0 * signal_std  # Lower factor (1.0) for more sensitivity
            # Use the minimum of the two thresholds for maximum sensitivity
            peak_thresh = min(peak_thresh, std_peak_thresh) if np.max(signal) > 0 else std_peak_thresh
        else:
            peak_thresh = 0.15 * np.max(signal) if np.max(signal) > 0 else 0
            
        # More sensitive distance parameter (0.3s instead of 0.5s)
        peak_distance = int(fs * 0.3)  # Minimum distance between peaks (0.3 seconds)
        
        # Use the window parameter instead of distance for minimum peak separation
        # Window parameter defines the size of the sliding window for peak detection
        peaks = NumpySignalProcessor.find_peaks(signal, fs=fs, threshold=peak_thresh, window=peak_distance)

        # 2. Find dips (expirations) by inverting the signal
        inverted_signal = -signal
        # Threshold for dips is based on the max of the *inverted* signal
        # Lower threshold (0.05 instead of 0.1) for more sensitivity
        if is_low_amplitude:
            dip_thresh = 0.05 * np.max(inverted_signal) if np.max(inverted_signal) > 0 else 0
            # Alternative threshold based on standard deviation
            std_dip_thresh = -signal_mean + 1.0 * signal_std
            # Use the minimum of the two thresholds for maximum sensitivity
            dip_thresh = min(dip_thresh, std_dip_thresh) if np.max(inverted_signal) > 0 else std_dip_thresh
        else:
            dip_thresh = 0.08 * np.max(inverted_signal) if np.max(inverted_signal) > 0 else 0
            
        # Use the window parameter for dips detection as well
        dips = NumpySignalProcessor.find_peaks(inverted_signal, fs=fs, threshold=dip_thresh, window=peak_distance)

        if not isinstance(peaks, np.ndarray):
            peaks = np.array([], dtype=int)
        if not isinstance(dips, np.ndarray):
            dips = np.array([], dtype=int)

        if peaks.size == 0 and dips.size == 0:
            return np.array([], dtype=int)

        # 3. Merge and sort events
        # Store as (index, type), where 1=peak, -1=dip
        events = []
        if peaks.size > 0:
            events.extend([(p, 1) for p in peaks])
        if dips.size > 0:
            events.extend([(d, -1) for d in dips])
        
        # Sort by index (timestamp)
        events.sort(key=lambda x: x[0])

        # 4. Validate sequence with improved pattern recognition
        if not events:
            return np.array([], dtype=int)
            
        # Improved validation logic for more sensitivity
        # This section handles breathing pattern recognition
        
        # First pass: Basic alternating sequence validation
        validated_events = [events[0]]
        for i in range(1, len(events)):
            # Only add if the type is different from the previous one
            if events[i][1] != validated_events[-1][1]:
                validated_events.append(events[i])
        
        # Add additional peaks if they meet spacing criteria
        # This helps catch events that don't perfectly alternate
        valid_event_indices = set(idx for idx, _ in validated_events)
        min_event_spacing = int(fs * 1.5)  # Minimum spacing between events (1.5 seconds)
        
        # Second pass: Check for isolated peaks that might have been missed
        for i, (idx, event_type) in enumerate(events):
            # Skip events already validated
            if idx in valid_event_indices:
                continue
                
            # Check if this is a peak
            if event_type == 1:
                # Check if it's sufficiently far from existing events
                is_isolated = all(abs(idx - existing_idx) > min_event_spacing for existing_idx, _ in validated_events)
                
                # If it's an isolated peak, add it
                if is_isolated:
                    validated_events.append((idx, event_type))
                    valid_event_indices.add(idx)
        
        # Sort all events by index again after adding isolated peaks
        validated_events.sort(key=lambda x: x[0])
        
        # 5. Return only the indices of the validated peaks
        validated_peaks = np.array([idx for idx, event_type in validated_events if event_type == 1], dtype=int)
        
        return validated_peaks

    @classmethod
    def IS_CHANGED(elf, input_signal_id, show_peaks, output_signal_id, enabled):
        return float("NaN")

    def __init__(self):
        self._recent_peak_times = []
        self._last_rr = 0.0
        self._last_is_peak = False
        self.arousal= 0.5
#================================================================================================================================== Background Processing ============================================================================================================================
    def _background_process(self, input_signal_id, show_peaks, stop_flag, output_signal_id):
        #--- Initialize signal registry and buffers ---
        registry = SignalRegistry.get_instance()
        fs = 1000  # RR typical sampling frequency
        viz_window_sec = 30  # Changed to 30 seconds as requested
        viz_buffer_size = fs * viz_window_sec
        
        # Feature buffer should preserve 30 seconds but accounting for downsampling
        feature_window_sec = 30  # Also 30 seconds for features
        # Calculate decimation factor first to determine actual feature buffer size
        nyquist_fs = fs / 2
        max_frequency_interest = 1000  # RR rarely above 1 Hz (60 bpm)
        decimation_factor = max(1, int(nyquist_fs / max_frequency_interest))
        use_decimation = decimation_factor > 1
        
        # Feature buffer size accounts for downsampling - preserves 30 seconds of downsampled data
        effective_fs_after_decimation = fs // decimation_factor if use_decimation else fs
        feature_buffer_size = effective_fs_after_decimation * feature_window_sec + effective_fs_after_decimation  # +1 second safety margin
        
        feature_values_deque = deque(maxlen=feature_buffer_size)
        feature_timestamps_deque = deque(maxlen=feature_buffer_size)
        viz_values_deque = deque(maxlen=viz_buffer_size)
        viz_timestamps_deque = deque(maxlen=viz_buffer_size)
        max_peaks_to_average = 10
        last_registry_data_hash = None
        last_process_time = time.time()
        processing_interval = 0.2  # 5 Hz processing rate (reduced for better performance)
        start_time = None
        # --- Filtering parameters for RR, adapted for decimation if needed ---

        # RR metrics buffer: preserve 30 seconds of RR data 
        # For RR metrics, we need much fewer samples - typical RR is 10-30 breaths/min
        # So for 30 seconds, we need max ~15 RR values, but use buffer for safety
        rr_metrics_window_sec = 30.0    # 30 seconds for RR metrics
        rr_metrics_update_rate = 3.0    # Approximately 3 Hz for RR updates (conservative)
        rr_metrics_buffer_size = int(rr_metrics_window_sec * rr_metrics_update_rate)  
        rr_metrics_deque = deque(maxlen=rr_metrics_buffer_size)

        while not stop_flag[0]:
            current_time = time.time()
            if current_time - last_process_time < processing_interval:
                continue  # Skip this iteration, don't sleep
            last_process_time = current_time
            signal_data = registry.get_signal(input_signal_id)
            #print(f"[RR_NODE_DEBUG] input_signal_id: {input_signal_id}, signal_data available: {signal_data is not None}") # DEBUG PRINT
            if not signal_data or "t" not in signal_data or "v" not in signal_data:
                continue  # Skip this iteration, don't sleep
            timestamps = np.array(signal_data["t"])
            values = np.array(signal_data["v"])
            #print(f"[RR_NODE_DEBUG] Timestamps len: {len(timestamps)}, Values len: {len(values)}") # DEBUG PRINT
            if len(timestamps) < 2 or len(values) < 2:
                continue  # Skip this iteration, don't sleep
                continue
            # --- Retrieve start_time from metadata if available (for peak logic only) ---
            meta = None
            if hasattr(registry, 'get_signal_metadata'):
                meta = registry.get_signal_metadata(input_signal_id)
            elif isinstance(signal_data, dict) and 'meta' in signal_data:
                meta = signal_data['meta']
            meta_start_time = meta['start_time'] if meta and 'start_time' in meta else None
            if start_time is None and len(feature_timestamps_deque) > 0:
                start_time = time.time() - feature_timestamps_deque[0]
            # Process all new data - simplified logic for rotating deques
            # When deques are full, they automatically rotate old data out
            if use_decimation:
                if len(timestamps) > 1:
                    decimated_values = NumpySignalProcessor.robust_decimate(values, decimation_factor)
                    decimated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=len(decimated_values)) if len(timestamps) > 1 else timestamps
                    
                    # For rotating deques, we need to check if we have genuinely new data
                    # Compare with the last few timestamps to avoid duplicate processing
                    if len(viz_timestamps_deque) > 0:
                        last_processed_time = viz_timestamps_deque[-1]
                        # Only add data that's newer than what we last processed
                        new_mask = decimated_timestamps > last_processed_time
                        if np.any(new_mask):
                            new_decimated_timestamps = decimated_timestamps[new_mask]
                            new_decimated_values = decimated_values[new_mask]
                            viz_timestamps_deque.extend(new_decimated_timestamps)
                            viz_values_deque.extend(new_decimated_values)
                            feature_timestamps_deque.extend(new_decimated_timestamps)
                            feature_values_deque.extend(new_decimated_values)
                    else:
                        # First time processing - add all data
                        viz_timestamps_deque.extend(decimated_timestamps)
                        viz_values_deque.extend(decimated_values)
                        feature_timestamps_deque.extend(decimated_timestamps)
                        feature_values_deque.extend(decimated_values)
            else:
                # For rotating deques, check for new data based on timestamps
                if len(viz_timestamps_deque) > 0:
                    last_processed_time = viz_timestamps_deque[-1]
                    # Only add data that's newer than what we last processed
                    new_mask = timestamps > last_processed_time
                    if np.any(new_mask):
                        new_timestamps = timestamps[new_mask]
                        new_values = values[new_mask]
                        viz_timestamps_deque.extend(new_timestamps)
                        viz_values_deque.extend(new_values)
                        feature_timestamps_deque.extend(new_timestamps)
                        feature_values_deque.extend(new_values)
                else:
                    # First time processing - add all data
                    viz_timestamps_deque.extend(timestamps)
                    viz_values_deque.extend(values)
                    feature_timestamps_deque.extend(timestamps)
                    feature_values_deque.extend(values)
            
            viz_timestamps = np.array(viz_timestamps_deque)
            viz_values = np.array(viz_values_deque)
            feature_timestamps = np.array(feature_timestamps_deque)
            feature_values = np.array(feature_values_deque)
            # Adjust effective sampling rate if decimation was applied
            effective_fs = fs
            if use_decimation:
                effective_fs = fs / decimation_factor
            # Pre-process RR with bandpass filter adapted for RR (0.1-1 Hz)
            lowcut = 0.1
            highcut = 1
            filtered_rr = NumpySignalProcessor.bandpass_filter(
                feature_values,
                lowcut=lowcut,
                highcut=highcut,
                fs=effective_fs
            )
            
            # --- Use the new validation method to find peaks ---
            detected_peaks = RRNode.validate_breathing_events(filtered_rr, fs=effective_fs)
            
            # Validate peaks with lag-based edge avoidance (similar to ECG)
            peaks = detected_peaks
            avg_rr = 0.0
            # Use signal timestamp domain, not system time
            current_signal_time = feature_timestamps[-1] if len(feature_timestamps) > 0 else 0.0
            
            if isinstance(peaks, np.ndarray) and len(peaks) > 0 and len(feature_timestamps) > 0:
                peak_times = feature_timestamps[peaks]
                latest_peak_time = peak_times[-1]
                add_peak = False
                if not self._recent_peak_times or latest_peak_time != self._recent_peak_times[-1]:
                    if self._recent_peak_times:
                        last_time = self._recent_peak_times[-1]
                        if (latest_peak_time - last_time) >= 2.0:  # 2.0s min breath interval (realistic for breathing)
                            add_peak = True
                    else:
                        add_peak = True
                    if add_peak:
                        self._recent_peak_times.append(latest_peak_time)
                        
                # TIME-BASED FILTERING: Only keep peaks from last 30 seconds (signal time)
                time_window = 30.0  # 30 seconds
                cutoff_signal_time = current_signal_time - time_window
                self._recent_peak_times = [t for t in self._recent_peak_times if t >= cutoff_signal_time]
                
                # Calculate RR only if we have recent peaks (within 30 seconds of signal time)
                if len(self._recent_peak_times) >= 1:
                    # Check if most recent peak is recent enough (signal time domain)
                    most_recent_peak = max(self._recent_peak_times)
                    signal_time_since_last_peak = current_signal_time - most_recent_peak
                    
                    if signal_time_since_last_peak <= 10.0:  # If no peak in last 10 seconds of signal time, RR = 0
                        # FIXED FORMULA: Use fixed time window for consistent calculation
                        # For calm breathing with fewer peaks, we still use the full 30s window
                        actual_time_window = min(time_window, current_signal_time - min(self._recent_peak_times))
                        # Ensure we have a reasonable time window (at least 10s for reliable calculation)
                        if actual_time_window >= 10.0:
                            peak_count = len(self._recent_peak_times)
                            
                            # Use FIXED time window approach: peaks per fixed time window
                            breaths_per_second = peak_count / actual_time_window
                            avg_rr = breaths_per_second * 60.0  # Convert to breaths per minute
                        else:
                            avg_rr = 0.0  # Not enough time window for reliable calculation
                    else:
                        avg_rr = 0.0  # No recent breathing detected
                else:
                    avg_rr = 0.0  # Not enough peaks for calculation
            #print(f"[RR_NODE_DEBUG] avg_rr: {avg_rr}") # DEBUG PRINT
            # Robust is_peak logic (timing + newness)
            is_peak, latest_peak_time = RR.is_peak(
                filtered_rr, feature_timestamps, fs, start_time=meta_start_time, rr=avg_rr, used_peaks=self._recent_peak_times[:-1] if len(self._recent_peak_times) > 1 else []
            )
            self._last_rr = avg_rr
            self._last_is_peak = is_peak
            # Update arousal based on RR - handle edge cases better
            if avg_rr <= 0:
                self.arousal = 0.0  # No breathing = no arousal (sleep/apnea)
            else:
                self.arousal = Arousal.rr_arousal(avg_rr)

            # --- RR Metric Registration ---
            current_metric_timestamp = feature_timestamps[-1] if len(feature_timestamps) > 0 else time.time()
            # Ensure avg_rr is a float and finite, default to 0.0 if not
            rr_metric_value = float(avg_rr) if avg_rr is not None and np.isfinite(avg_rr) else 0.0
            
            rr_metrics_deque.append((current_metric_timestamp, rr_metric_value))
            
            rr_metric_t = [x[0] for x in rr_metrics_deque]
            rr_metric_v = [x[1] for x in rr_metrics_deque]

            rr_metric_data = {
                "t": rr_metric_t,
                "v": rr_metric_v,
                "last": rr_metric_value
            }
            #print(f"[RR_NODE_DEBUG] RR_METRIC data: t_len={len(rr_metric_t)}, v_len={len(rr_metric_v)}, last={rr_metric_value}") # DEBUG PRINT
            # Use metrics_registry (which is SignalRegistry) for metrics
            metrics_registry = SignalRegistry.get_instance() # Ensure we use the correct registry instance
            metrics_registry.register_signal("RR_METRIC", rr_metric_data, {
                "id": "RR_METRIC",
                "type": "rr_metrics", # Consistent type for metrics
                "label": "Respiration Rate (bpm)",
                "source": output_signal_id,
                "scope": "global_metric"
            })
            # --- End RR Metric Registration ---

            # Visualization window
            if len(viz_timestamps) > 0:
                window_max = viz_timestamps[-1]
                window_min = max(window_max - viz_window_sec, viz_timestamps[0])
                window_mask = (viz_timestamps >= window_min) & (viz_timestamps <= window_max)
                viz_timestamps_window = viz_timestamps[window_mask]
                viz_values_window = viz_values[window_mask]
            else:
                viz_timestamps_window = viz_timestamps
                viz_values_window = viz_values


            # Vectorized peak mapping to window
            peak_timestamps_in_window = []
            if show_peaks and len(peaks) > 0 and len(feature_timestamps) > 0 and len(viz_timestamps_window) > 0:
                peak_times = feature_timestamps[peaks]
                in_window_mask = (peak_times >= window_min) & (peak_times <= window_max)
                peak_timestamps_in_window = peak_times[in_window_mask].tolist()
            # Prepare filtered RR for visualization (match window)
            filtered_viz_rr = np.zeros_like(viz_values_window)
            if len(filtered_rr) > 0 and len(viz_values_window) > 0:
                if len(window_mask) == len(filtered_rr):
                    filtered_viz_rr = filtered_rr[window_mask]
                else:
                    n_window = len(viz_values_window)
                    filtered_window = filtered_rr[-n_window:] if len(filtered_rr) > n_window else filtered_rr
                    if len(filtered_window) < n_window:
                        filtered_viz_rr[-len(filtered_window):] = filtered_window
                    else:
                        filtered_viz_rr = filtered_window
                viz_min, viz_max = np.min(viz_values_window), np.max(viz_values_window)
                filtered_min, filtered_max = np.min(filtered_viz_rr), np.max(filtered_viz_rr)
                if filtered_max != filtered_min and viz_max != viz_min:
                    filtered_viz_rr = (filtered_viz_rr - filtered_min) / (filtered_max - filtered_min) * (viz_max - viz_min) + viz_min

  

            # Hash for registry update optimization - include deque length and range to detect rotation
            data_hash = hash((
                len(viz_timestamps_window),  # Length changes when deque rotates
                tuple(viz_timestamps_window[-5:]) if len(viz_timestamps_window) >= 5 else tuple(viz_timestamps_window),
                tuple(viz_timestamps_window[:3]) if len(viz_timestamps_window) >= 3 else tuple(),  # First few values change on rotation
                tuple(filtered_viz_rr[-5:]) if len(filtered_viz_rr) >= 5 else tuple(filtered_viz_rr),
                tuple(peak_timestamps_in_window[-3:] if peak_timestamps_in_window else []),
                avg_rr,
                viz_timestamps_window[0] if len(viz_timestamps_window) > 0 else 0,  # First timestamp changes on rotation
                viz_timestamps_window[-1] if len(viz_timestamps_window) > 0 else 0  # Last timestamp always changes
            ))
            # if data_hash == last_registry_data_hash: # Reverted debug change
            # time.sleep(0.01) # Reverted debug change
            # continue # Reverted debug change
            # last_registry_data_hash = time.time() # Reverted debug change
            if data_hash == last_registry_data_hash: # Original logic
                continue  # Skip when no data available
            last_registry_data_hash = data_hash


            metadata = {
                "id": output_signal_id,
                "type": "processed",  # CHANGED from "rr_processed"
                "rr": avg_rr,
                "arousal": self.arousal,  # Add arousal value for metrics view
                "color": "#55F4FF",
                "show_peaks": show_peaks,
                "peak_marker": "o",
                "peak_timestamps": peak_timestamps_in_window
            }
            processed_signal_data = {
                "t": viz_timestamps_window.tolist(),
                "v": filtered_viz_rr.tolist()
            }
            #print(f"[RR_NODE_DEBUG] RR_PROCESSED data: t_len={len(processed_signal_data['t'])}, v_len={len(processed_signal_data['v'])}, metadata: {metadata}") # DEBUG PRINT
            registry.register_signal(output_signal_id, processed_signal_data, metadata)
            
            # Register arousal as a separate metric for the metrics view
            # metrics_registry = registry  # Use the same registry # This was potentially problematic if registry was PlotRegistry
            
            # Ensure we have a valid arousal value
            arousal_value = 0.5  # Default middle value
            if self.arousal is not None and isinstance(self.arousal, (float, int)) and np.isfinite(self.arousal):
                arousal_value = float(self.arousal)
                
            # Create timestamp - use the latest timestamp from the visualization window or current time
            current_timestamp = viz_timestamps_window[-1] if len(viz_timestamps_window) > 0 else time.time()
            
            # Prepare arousal metric data with both timeseries and 'last' value for easy access
            arousal_metrics_data = {
                "t": [current_timestamp],
                "v": [arousal_value],
                "last": arousal_value  # Add last value for easy access in metrics view
            }
            
            # Register with the metrics registry
            metrics_registry.register_signal("RR_AROUSAL_METRIC", arousal_metrics_data, {
                "id": "RR_AROUSAL_METRIC",
                "type": "arousal_metrics",
                "label": "RR Arousal",
                "source": output_signal_id,
                "scope": "global_metric",
                "arousal_value": float(arousal_value)  # Explicitly add arousal value as a float in metadata
            })
            continue  # Continue processing loop without delay
#================================================================================================================================== Processing Output ============================================================================================================================
    def process_rr(self, input_signal_id="", show_peaks=True, output_signal_id="RR_PROCESSED", enabled=True):
        if not enabled:
            return 0.0, False, output_signal_id, 0.5  # Include default arousal value of 0.5
        signal_id = output_signal_id
        if signal_id in self._processing_threads and self._processing_threads[signal_id].is_alive():
            avg_rr = getattr(self, '_last_rr', 0.0)
            is_peak = getattr(self, '_last_is_peak', False)
            arousal = getattr(self, 'arousal', 0.5)  # Get current arousal value or default to 0.5
            return avg_rr, is_peak, output_signal_id, arousal
        if signal_id in self._stop_flags:
            self._stop_flags[signal_id][0] = True
        stop_flag = [False]
        self._stop_flags[signal_id] = stop_flag
        thread = threading.Thread(
            target=self._background_process,
            args=(input_signal_id, show_peaks, stop_flag, output_signal_id),
            daemon=True
        )
        self._processing_threads[signal_id] = thread
        thread.start()
        return self._last_rr, self._last_is_peak, output_signal_id, self.arousal
        
    def __del__(self):
        """Clean up background threads when node is deleted"""
        for stop_flag in self._stop_flags.values():
            stop_flag[0] = True
        self._processing_threads.clear()
        self._stop_flags.clear()

NODE_CLASS_MAPPINGS = {
    "RRNode": RRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RRNode": "🌬️ RR Processor (Registry)"
}

