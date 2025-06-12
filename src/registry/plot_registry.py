import numpy as np
import threading
import logging
import time
from collections import OrderedDict
import sys
import builtins

# Configure logger
logger = logging.getLogger('PlotRegistry')

class PlotRegistry:
    """
    A dedicated registry for the PlotUnit to track signals and visualization connections.
    This acts as the bridge between signal sources and visualization.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        # Use builtins to store the singleton globally
        if not hasattr(builtins, '_PIC25_PLOT_REGISTRY_SINGLETON'):
            setattr(builtins, '_PIC25_PLOT_REGISTRY_SINGLETON', cls())
        return getattr(builtins, '_PIC25_PLOT_REGISTRY_SINGLETON')
    
    def __init__(self):
        """Initialize the registry with empty containers"""
        self.signals = OrderedDict()  # Signal data keyed by ID
        self.metadata = {}           # Signal metadata keyed by ID
        self.connections = {}        # Track which nodes connect to which signals
        self.visualized_signals = set()  # Track signals actually being visualized
        self.creation_time = time.time()
        self.connected_nodes = 0     # Count of connected nodes
        
        # Thread-safety lock for registry operations
        self.registry_lock = threading.Lock()

    def register_signal(self, signal_id, signal_data, metadata=None):
        """
        Register a signal in the registry, optimized to avoid unnecessary copies and metadata updates.
        Also, if the signal contains metrics (e.g., HR, RR, SCL, SCK), register/update corresponding metrics signals for MetricsView.
        """
        with self.registry_lock:
            # --- Existing registration logic ---
            # Accept dict with 't' and 'v' as-is (new format)
            if isinstance(signal_data, dict) and 't' in signal_data and 'v' in signal_data:
                # Only copy if the object is not already the same as last
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    #print(f"[DEBUG] PlotRegistry: Updating signal data for {signal_id}")
                    self.signals[signal_id] = signal_data
            # ADDED: Handle other dictionaries (e.g. complex processed data like EDA)
            elif isinstance(signal_data, dict):
                prev = self.signals.get(signal_id)
                if prev is not signal_data: 
                    self.signals[signal_id] = signal_data # Store as dict
            # Existing logic for tuple, array, etc.
            elif isinstance(signal_data, tuple) and len(signal_data) == 2:
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    self.signals[signal_id] = signal_data
            elif not isinstance(signal_data, np.ndarray): # Catches lists and other convertibles, but not dicts anymore
                arr = np.array(signal_data)
                prev = self.signals.get(signal_id)
                # Comparing numpy arrays with 'is' is not reliable if they are new objects.
                # np.array_equal could be used, but for now, this matches existing style.
                if prev is not arr: # This might lead to more updates than necessary if arr is always new.
                    self.signals[signal_id] = arr
            else: # signal_data is already an np.ndarray
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    self.signals[signal_id] = signal_data
            
            # Store metadata if provided (this logic applies after signal_data is stored)
            if metadata is not None: # Check if metadata was passed in this call
                prev_meta = self.metadata.get(signal_id)
                if prev_meta != metadata:
                    # Make sure we preserve the 'over' flag if it was previously set but not in new metadata
                    if prev_meta and 'over' in prev_meta and prev_meta['over'] and ('over' not in metadata or not metadata.get('over')): # check metadata.get('over')
                        metadata['over'] = True
                    self.metadata[signal_id] = metadata
            elif signal_id not in self.metadata: # If no metadata provided in call AND no metadata exists yet for this signal_id
                self.metadata[signal_id] = {
                    'color': self._generate_color_from_id(signal_id),
                    'created': time.time()
                }
            # --- Passive metrics registration ---
            metrics_to_check = [
                ("hr", "HR_METRIC"),
                ("rr", "RR_METRIC"),
                ("scl", "SCL_METRIC"),
                ("sck", "SCK_METRIC"),
            ]
            meta_src = metadata or (signal_data.get('meta') if isinstance(signal_data, dict) else None)
            # If this is a time series with 't' and 'v', and the metric is present, register as a time series
            for metric_key, metric_id in metrics_to_check:
                # If this signal is itself a metric time series, skip (avoid recursion)
                if signal_id == metric_id:
                    continue
                # If this is a time series and meta has the metric, append to metric time series
                if (
                    isinstance(signal_data, dict)
                    and 't' in signal_data and 'v' in signal_data
                    and meta_src and metric_key in meta_src
                ):
                    t_now = float(signal_data['t'][-1]) if len(signal_data['t']) else time.time()
                    val = meta_src[metric_key]
                    prev = self.signals.get(metric_id)
                    if prev and isinstance(prev, dict) and 't' in prev and 'v' in prev:
                        prev['t'].append(t_now)
                        prev['v'].append(val)
                        if len(prev['t']) > 1000:
                            prev['t'] = prev['t'][-1000:]
                            prev['v'] = prev['v'][-1000:]
                        self.signals[metric_id] = prev
                    else:
                        self.signals[metric_id] = {'t': [t_now], 'v': [val]}
                    if metric_id not in self.metadata:
                        self.metadata[metric_id] = {
                            'id': metric_id,
                            'type': 'metric',
                            'source': signal_id,
                            'created': t_now
                        }
                    #print(f"[PlotRegistry][metrics] Registered/updated {metric_id}: t={t_now}, val={val}, keys={list(self.signals[metric_id].keys())}")
                # If this is a metric value (not a time series), register as a single-point time series
                elif meta_src and metric_key in meta_src:
                    t_now = time.time()
                    val = meta_src[metric_key]
                    self.signals[metric_id] = {'t': [t_now], 'v': [val]}
                    if metric_id not in self.metadata:
                        self.metadata[metric_id] = {
                            'id': metric_id,
                            'type': 'metric',
                            'source': signal_id,
                            'created': t_now
                        }
                    #print(f"[PlotRegistry][metrics] Registered/updated {metric_id}: t={t_now}, val={val}, keys={list(self.signals[metric_id].keys())}")
    
    def connect_node_to_signal(self, node_id, signal_id):
        """
        Connect a visualization node to a specific signal.
        
        Args:
            node_id (str): Unique identifier for the node
            signal_id (str): Identifier of the signal to connect to
        """
        with self.registry_lock:
            # Validate input parameters
            if not isinstance(signal_id, str):
                logger.error(f"Invalid signal_id type: {type(signal_id)}. Expected string.")
                return False
            
            # Make sure the signal exists
            if signal_id not in self.signals:
                logger.warning(f"Attempted to connect node {node_id} to non-existent signal {signal_id}")
                return False
            
            # Register the connection
            if node_id not in self.connections:
                self.connections[node_id] = set()
                self.connected_nodes += 1
            
            self.connections[node_id].add(signal_id)
            self.visualized_signals.add(signal_id)
            
            logger.debug(f"Node {node_id} connected to signal {signal_id}")
            return True
    
    def disconnect_node(self, node_id):
        """
        Disconnect a node from all its signals.
        
        Args:
            node_id (str): Identifier of the node to disconnect
        """
        with self.registry_lock:
            if node_id in self.connections:
                # Remove all connections for this node
                del self.connections[node_id]
                self.connected_nodes -= 1
                
                # Update visualized signals
                self._update_visualized_signals()
                
                logger.debug(f"Node {node_id} disconnected from all signals")
                return True
            return False
    
    def get_signal(self, signal_id):
        """Get a signal by its ID"""
        with self.registry_lock:
            if signal_id in self.signals:
                return self.signals[signal_id]
            return None
    
    def get_signal_metadata(self, signal_id):
        """Get metadata for a signal"""
        with self.registry_lock:
            if signal_id in self.metadata:
                return self.metadata[signal_id]
            return None
    def get_all_signals(self):
        """Get all signal IDs in the registry"""
        with self.registry_lock:
            return list(self.signals.keys())
            
    def get_all_signal_ids(self):
        """Get all signal IDs in the registry (alias for get_all_signals)"""
        return self.get_all_signals()
    
    def get_visualized_signals(self):
        """Get IDs of signals that are actively visualized"""
        with self.registry_lock:
            return list(self.visualized_signals)
    
    def clear_signals(self):
        """Clear all signals from the registry"""
        with self.registry_lock:
            self.signals.clear()
            self.metadata.clear()
            self.visualized_signals.clear()
            # Don't clear connections - just the signal data
            logger.info("All signals cleared from registry")
    
    def reset(self):
        """Reset the entire registry"""
        with self.registry_lock:
            self.signals.clear()
            self.metadata.clear()
            self.connections.clear()
            self.visualized_signals.clear()
            self.connected_nodes = 0
            self.creation_time = time.time()
            logger.info("PlotRegistry completely reset")
    
    def _update_visualized_signals(self):
        """Update the set of visualized signals based on connections"""
        active_signals = set()
        for node_id, signals in self.connections.items():
            active_signals.update(signals)
        self.visualized_signals = active_signals
    
    def _generate_color_from_id(self, signal_id):
        """Generate a consistent color based on signal ID"""
        import hashlib
        hash_val = int(hashlib.md5(signal_id.encode()).hexdigest(), 16)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        # Make colors brighter for better visibility
        color = (min(r + 100, 255), min(g + 100, 255), min(b + 100, 255))
        return color
    
    def get_signals_by_type(self, window_sec, signal_type, debug=False):
        """
        Fetch signals from the registry by type ('raw' or 'processed'), returning a list of dicts:
        [{ 'id': ..., 't': ..., 'v': ..., 'meta': ... }, ...]\r\n        """
        import numpy as np
        all_signal_ids = self.get_all_signal_ids()
        ids = []
        for sid in all_signal_ids:
            meta = self.get_signal_metadata(sid)
            meta_type = meta.get('type') if meta else None
            if signal_type == 'raw':
                if (meta_type == 'raw' or
                    (meta_type is None and ('RAW' in sid or 'ECG' in sid or 'EDA' in sid or '_RAW' in sid.upper()))): # Added _RAW check
                    ids.append(sid)
            elif signal_type == 'processed':
                if (meta_type == 'processed' or 
                    (meta_type and meta_type.endswith('_processed')) or
                    (meta_type is None and ('PROC' in sid.upper() or 'PROCESSED' in sid.upper() or 'WAVE' in sid.upper() or '_PROC' in sid.upper()))): # Added _PROC and made checks case-insensitive
                    ids.append(sid)
        
        signals_out = [] # Renamed from 'signals' to avoid conflict with module
        for sid in ids:
            data = self.get_signal(sid)
            current_meta_from_registry = self.get_signal_metadata(sid) # Original metadata for this signal
            
            if data is None:
                if debug: logger.warning(f"[PlotRegistry.get_signals_by_type] No data for signal ID: {sid}")
                continue

            t_signal, v_signal, processed_meta = None, None, current_meta_from_registry

            # Try to parse data into t_signal, v_signal, and update processed_meta if needed
            if isinstance(data, dict) and 't' in data and 'v' in data: # Standard dict format
                t_signal = np.array(data['t'])
                v_signal = np.array(data['v'])
                if 'meta' in data and isinstance(data['meta'], dict): # If data dict itself contains meta, prefer it
                    processed_meta = data['meta']
            elif isinstance(data, dict) and 't' in data: # Handles other dicts like processed EDA
                t_signal = np.array(data['t'])
                
                # Enrich metadata with the original data dictionary
                enriched_meta = processed_meta.copy() if processed_meta else {}
                enriched_meta['_original_data_dict'] = data
                processed_meta = enriched_meta
                
                # Attempt to find a primary 'v' component for standardization
                if 'phasic_norm' in data: # EDA specific
                    v_signal = np.array(data['phasic_norm'])
                elif 'tonic_norm' in data: # EDA specific fallback
                    v_signal = np.array(data['tonic_norm'])
                # Add other 'elif key in data:' checks here if other dict types need a primary 'v'
                else:
                    v_signal = np.array([]) # Default to empty 'v' if no primary component found
            
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2: # Numpy (N,2) array
                t_signal, v_signal = data[:, 0], data[:, 1]
            
            elif isinstance(data, tuple) and len(data) == 2: # (timestamps, values) tuple
                try:
                    t_signal, v_signal = np.array(data[0]), np.array(data[1])
                except Exception as e:
                    if debug: logger.error(f"[PlotRegistry.get_signals_by_type] Error converting tuple signal {sid} to numpy arrays: {e}")
                    continue
            
            elif isinstance(data, (list, np.ndarray)): # 1D list or array (treat as values, synthesize t)
                if isinstance(data, np.ndarray) and data.ndim == 0:
                    if debug: logger.warning(f"[PlotRegistry.get_signals_by_type] Signal {sid} data is a 0-d numpy array (value: {data.item()}). Cannot synthesize 't'. Skipping.")
                    continue
                
                v_signal = np.array(data)
                if len(v_signal) == 0:
                    if debug: logger.warning(f"[PlotRegistry.get_signals_by_type] Signal {sid} data is an empty list/array. Skipping.")
                    continue
                
                sr = processed_meta.get('sampling_rate', 100) if processed_meta else 100
                t_signal = np.arange(len(v_signal)) / sr
            
            else:
                if debug: logger.warning(f"[PlotRegistry.get_signals_by_type] Signal {sid} has unknown data format: {type(data)}. Skipping.")
                continue

            # If parsing failed to produce t_signal, skip
            if t_signal is None:
                if debug: logger.warning(f"[PlotRegistry.get_signals_by_type] Signal {sid} could not be parsed into t_signal. Skipping.")
                continue
            
            # Ensure v_signal is an ndarray if it's None (e.g. if t_signal was from dict but no v_keys matched)
            if v_signal is None:
                v_signal = np.array([])

            # Synchronize lengths of t_signal and v_signal if v_signal is not empty
            # (it can be deliberately empty for dicts without a primary 'v' like EDA)
            if v_signal.size > 0:
                min_len_sync = min(len(t_signal), len(v_signal))
                if len(t_signal) != min_len_sync: t_signal = t_signal[:min_len_sync]
                if len(v_signal) != min_len_sync: v_signal = v_signal[:min_len_sync]
            
            # Store unwindowed versions for potential fallback if windowing makes signal too short
            t_unwindowed, v_unwindowed = t_signal.copy(), v_signal.copy()

            # Apply windowing if window_sec is specified
            if window_sec is not None and len(t_signal) > 1:
                t0 = t_signal[-1] - window_sec
                mask = t_signal >= t0
                
                t_signal_windowed = t_signal[mask]
                v_signal_windowed = v_signal[mask] if v_signal.size > 0 and len(mask) == len(v_signal) else (np.array([]) if v_signal.size == 0 else v_signal[mask])


                # If windowing results in too few points, try to take last 2 from original (if available)
                if len(t_signal_windowed) < 2 and len(t_unwindowed) >= 2:
                    t_final = t_unwindowed[-2:]
                    v_final = v_unwindowed[-2:] if v_unwindowed.size > 0 and len(v_unwindowed) >=2 else (np.array([]) if v_unwindowed.size == 0 else v_unwindowed[-min(2, len(v_unwindowed)):])

                else:
                    t_final = t_signal_windowed
                    v_final = v_signal_windowed

                if len(t_final) > 0:
                    t_final = t_final - t_final[0] # Normalize time to start from 0
                else: # t_final is empty, v_final should also be empty
                    v_final = np.array([])
            else:
                # No windowing or signal too short for windowing, use original (or length-synced) t_signal, v_signal
                t_final, v_final = t_signal, v_signal
            
            signals_out.append({'id': sid, 't': t_final, 'v': v_final, 'meta': processed_meta})
        
        return signals_out
