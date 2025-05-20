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
        Register a signal in the registry.
        
        Args:
            signal_id (str): Unique identifier for the signal
            signal_data (array or tuple): The signal data (numpy array, list, or (timestamps, values) tuple)
            metadata (dict, optional): Metadata for the signal (color, etc)
        """
        with self.registry_lock:
            # --- Accept dict with 't' and 'v' as-is (new format) ---
            if isinstance(signal_data, dict) and 't' in signal_data and 'v' in signal_data:
                # Always store meta in the signal dict for UI compatibility
                if metadata is not None:
                    signal_data = dict(signal_data)  # copy
                    signal_data['meta'] = metadata
                self.signals[signal_id] = signal_data
                logger.debug(f"Signal '{signal_id}' registered as dict with keys {list(signal_data.keys())}")
            # --- Existing logic for tuple, array, etc ---
            elif isinstance(signal_data, tuple) and len(signal_data) == 2:
                self.signals[signal_id] = signal_data
                logger.debug(f"Signal '{signal_id}' registered as tuple with lengths {[len(x) for x in self.signals[signal_id]]}")
            elif not isinstance(signal_data, np.ndarray):
                signal_data = np.array(signal_data)
                self.signals[signal_id] = signal_data
                logger.debug(f"Signal '{signal_id}' registered with shape {self.signals[signal_id].shape}")
            else:
                self.signals[signal_id] = signal_data
                logger.debug(f"Signal '{signal_id}' registered with shape {self.signals[signal_id].shape}")
            
            # Store metadata if provided
            if metadata:
                self.metadata[signal_id] = metadata
            elif signal_id not in self.metadata:
                # Default metadata if none exists
                self.metadata[signal_id] = {
                    'color': self._generate_color_from_id(signal_id),
                    'created': time.time()
                }
                
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
