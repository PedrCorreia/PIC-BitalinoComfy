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
            print(f"[DEBUG] PlotRegistry: Registering signal {signal_id}, metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            if metadata and 'over' in metadata:
                print(f"[DEBUG] PlotRegistry: Signal {signal_id} has over={metadata['over']}")
            if metadata and 'phasic_norm' in metadata and 'tonic_norm' in metadata:
                print(f"[DEBUG] PlotRegistry: Signal {signal_id} has phasic/tonic components")
                
            # --- Existing registration logic ---
            # Accept dict with 't' and 'v' as-is (new format)
            if isinstance(signal_data, dict) and 't' in signal_data and 'v' in signal_data:
                # Only copy if the object is not already the same as last
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    print(f"[DEBUG] PlotRegistry: Updating signal data for {signal_id}")
                    self.signals[signal_id] = signal_data
                # Only update meta if changed
                if metadata is not None:
                    prev_meta = self.metadata.get(signal_id)
                    if prev_meta != metadata:
                        print(f"[DEBUG] PlotRegistry: Updating metadata for {signal_id}")
                        # Make sure we preserve the 'over' flag if it was previously set but not in new metadata
                        if prev_meta and 'over' in prev_meta and prev_meta['over'] and metadata and 'over' not in metadata:
                            print(f"[DEBUG] PlotRegistry: Preserving over=True flag from previous metadata")
                            metadata['over'] = True
                        self.metadata[signal_id] = metadata
            # Existing logic for tuple, array, etc
            elif isinstance(signal_data, tuple) and len(signal_data) == 2:
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    self.signals[signal_id] = signal_data
            elif not isinstance(signal_data, np.ndarray):
                arr = np.array(signal_data)
                prev = self.signals.get(signal_id)
                if prev is not arr:
                    self.signals[signal_id] = arr
            else:
                prev = self.signals.get(signal_id)
                if prev is not signal_data:
                    self.signals[signal_id] = signal_data
            # Store metadata if provided
            if metadata:
                prev_meta = self.metadata.get(signal_id)
                if prev_meta != metadata:
                    self.metadata[signal_id] = metadata
            elif signal_id not in self.metadata:
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
        [{ 'id': ..., 't': ..., 'v': ..., 'meta': ... }, ...]
        """
        import numpy as np
        all_signal_ids = self.get_all_signal_ids()
        # --- Improved type detection: use metadata if available, fallback to ID heuristics ---
        ids = []
        for sid in all_signal_ids:
            meta = self.get_signal_metadata(sid)
            meta_type = meta.get('type') if meta else None
            if signal_type == 'raw':
                if (meta_type == 'raw' or
                    (meta_type is None and ('RAW' in sid or 'ECG' in sid or 'EDA' in sid))):
                    ids.append(sid)
            elif signal_type == 'processed':
                if (meta_type == 'processed' or
                    (meta_type is None and ('PROC' in sid or 'PROCESSED' in sid or 'WAVE' in sid))):
                    ids.append(sid)
        signals = []
        for sid in ids:
            data = self.get_signal(sid)
            meta = self.get_signal_metadata(sid)
            if data is None:
                continue
            # --- Patch: robustly handle all generator/legacy formats ---
            t = v = None
            # Handle new format: dict with 't', 'v', 'meta'
            if isinstance(data, dict) and 't' in data and 'v' in data:
                t = np.array(data['t'])
                v = np.array(data['v'])
                # If meta is already in the data dict, use it; otherwise use the one from the registry
                if 'meta' in data:
                    meta = data['meta']
            # Handle numpy array of shape (N, 2)
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
                t, v = data[:, 0], data[:, 1]
            # Handle (timestamps, values) tuple
            elif isinstance(data, tuple) and len(data) == 2:
                t, v = np.array(data[0]), np.array(data[1])
            # Handle 1D array or list: treat as values, synthesize t
            elif isinstance(data, (list, np.ndarray)):
                v = np.array(data)
                sr = meta.get('sampling_rate', 100) if meta else 100
                t = np.arange(len(v)) / sr
            else:
                continue  # skip unknown format
            # --- Ensure t and v are the same length before windowing ---
            min_len = min(len(t), len(v))
            t = t[:min_len]
            v = v[:min_len]
            # --- Only apply windowing if window_sec is not None ---
            if window_sec is not None and len(t) > 1:
                t0 = t[-1] - window_sec
                mask = t >= t0
                if len(mask) == len(t):
                    t, v = t[mask], v[mask]
                if len(t) < 2:
                    t, v = t[-2:], v[-2:]
                t = t - t[0]
                signals.append({'id': sid, 't': t, 'v': v, 'meta': meta})
            else:
                # No windowing, just return full t and v
                signals.append({'id': sid, 't': t, 'v': v, 'meta': meta})
        return signals
