"""
Live Signal Adapter

This module provides a bridge between SignalRegistry and PlotRegistry,
enabling live signal visualization with proper thread safety.

Part of the PIC-2025 unified signal architecture:
SignalGenerators → SignalRegistry → PlotRegistry → Visualization System
"""

import threading
import time
import logging
import numpy as np
from enum import Enum

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveSignalAdapter')

class SignalAdapterState(Enum):
    """States for the LiveSignalAdapter thread."""
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2

class LiveSignalAdapter:
    """
    Adapter to bridge SignalRegistry and PlotRegistry with thread safety.
    
    This class manages the signal flow between the two registries,
    ensuring proper thread management and signal routing based on
    the currently selected view mode.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of LiveSignalAdapter."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the LiveSignalAdapter."""
        if LiveSignalAdapter._instance is not None:
            raise RuntimeError("Use LiveSignalAdapter.get_instance() to get the singleton instance")
        
        self.signal_registry = None
        self.plot_registry = None
        self.current_view_mode = None
        self.thread = None
        self.state = SignalAdapterState.STOPPED
        self.state_lock = threading.Lock()
        self.adapter_config = {
            'update_interval': 0.1,  # Update interval in seconds
            'max_signal_length': 1000,  # Maximum number of points to keep
            'auto_reconnect': True,  # Automatically reconnect if disconnected
        }
        
        logger.info("LiveSignalAdapter instance created")
    
    def connect_registries(self, signal_registry, plot_registry):
        """
        Connect to the signal and plot registries.
        
        Args:
            signal_registry: The SignalRegistry instance
            plot_registry: The PlotRegistry instance
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        logger.info("Connecting to registries")
        
        if signal_registry is None or plot_registry is None:
            logger.error("Cannot connect to None registries")
            return False
        
        self.signal_registry = signal_registry
        self.plot_registry = plot_registry
        
        logger.info("Connected to registries")
        return True
    
    def start(self, view_mode=None):
        """
        Start the adapter thread.
        
        Args:
            view_mode: Initial view mode to use
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self.state_lock:
            if self.state == SignalAdapterState.RUNNING:
                logger.warning("Adapter already running")
                return True
            
            if self.signal_registry is None or self.plot_registry is None:
                logger.error("Cannot start adapter without connected registries")
                return False
            
            self.current_view_mode = view_mode
            self.state = SignalAdapterState.RUNNING
        
        self.thread = threading.Thread(target=self._adapter_thread, daemon=True)
        self.thread.start()
        
        logger.info(f"Signal adapter thread started with view mode: {view_mode}")
        return True
    
    def stop(self):
        """
        Stop the adapter thread gracefully.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        with self.state_lock:
            if self.state == SignalAdapterState.STOPPED:
                return True
            
            self.state = SignalAdapterState.STOPPED
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Adapter thread did not stop within timeout")
                return False
        
        logger.info("Signal adapter thread stopped")
        return True
    
    def pause(self):
        """
        Pause signal updates temporarily.
        
        Returns:
            bool: True if paused successfully, False otherwise
        """
        with self.state_lock:
            if self.state == SignalAdapterState.RUNNING:
                self.state = SignalAdapterState.PAUSED
                logger.info("Signal adapter paused")
                return True
            return False
    
    def resume(self):
        """
        Resume signal updates from a paused state.
        
        Returns:
            bool: True if resumed successfully, False otherwise
        """
        with self.state_lock:
            if self.state == SignalAdapterState.PAUSED:
                self.state = SignalAdapterState.RUNNING
                logger.info("Signal adapter resumed")
                return True
            return False
    
    def set_view_mode(self, view_mode):
        """
        Change the current view mode.
        
        This method safely changes the view mode by briefly pausing
        the adapter thread to ensure thread safety during tab switches.
        
        Args:
            view_mode: The new view mode to use
            
        Returns:
            bool: True if changed successfully, False otherwise
        """
        # Pause the adapter to ensure thread safety during view mode change
        was_running = False
        with self.state_lock:
            if self.state == SignalAdapterState.RUNNING:
                was_running = True
                self.state = SignalAdapterState.PAUSED
        
        # Short sleep to allow current cycle to finish
        time.sleep(0.1)
        
        # Update view mode
        logger.info(f"Changing view mode to: {view_mode}")
        self.current_view_mode = view_mode
        
        # Resume if it was running
        if was_running:
            with self.state_lock:
                self.state = SignalAdapterState.RUNNING
        
        return True
    
    def _adapter_thread(self):
        """Main thread function for the adapter."""
        logger.info("Adapter thread starting")
        
        try:
            while True:
                # Check if we should stop
                with self.state_lock:
                    if self.state == SignalAdapterState.STOPPED:
                        break
                    elif self.state == SignalAdapterState.PAUSED:
                        pass  # Do nothing while paused
                    elif self.state == SignalAdapterState.RUNNING:
                        # Only run the update when in RUNNING state
                        self._update_signals()
                
                # Sleep between updates
                time.sleep(self.adapter_config['update_interval'])
                
        except Exception as e:
            logger.error(f"Error in adapter thread: {e}")
        
        logger.info("Adapter thread exiting")
    
    def _update_signals(self):
        """Update signals from SignalRegistry to PlotRegistry based on current view mode."""
        if self.signal_registry is None or self.plot_registry is None:
            return
        
        try:
            # Get signals from the registry
            registry_signals = self._get_registry_signals()
            
            if not registry_signals:
                return
            
            # Filter signals based on current view mode
            filtered_signals = self._filter_signals_for_view_mode(registry_signals)
            
            # Update the plot registry with the filtered signals
            if hasattr(self.plot_registry, 'data_lock'):
                with self.plot_registry.data_lock:
                    if hasattr(self.plot_registry, 'data'):
                        # Clear existing data if this is PlotUnit
                        self.plot_registry.data.clear()
                    
                    # Update signals
                    self._apply_signals_to_registry(filtered_signals)
            else:
                # No data_lock available, use alternative approach
                self._apply_signals_to_registry(filtered_signals)
            
        except Exception as e:
            logger.error(f"Error updating signals: {e}")
    
    def _apply_signals_to_registry(self, signals):
        """Apply signals to the plot registry using appropriate method."""
        if not signals:
            return
            
        for signal_id, signal_data in signals.items():
            try:
                # Skip None or empty data
                if signal_data is None or (hasattr(signal_data, 'size') and signal_data.size == 0):
                    logger.debug(f"Skipping empty signal: {signal_id}")
                    continue
                
                # Get metadata if available
                metadata = self._get_signal_metadata(signal_id)
                
                # Ensure we have signal data as numpy array
                if not isinstance(signal_data, np.ndarray):
                    try:
                        signal_data = np.array(signal_data)
                        logger.debug(f"Converted {signal_id} data to numpy array")
                    except Exception as conv_err:
                        logger.warning(f"Could not convert signal {signal_id} to numpy array: {conv_err}")
                
                # Register with PlotRegistry based on available interface
                if hasattr(self.plot_registry, 'register_signal'):
                    self.plot_registry.register_signal(signal_id, signal_data, metadata)
                # If this is PlotUnit, directly update its data dictionary
                elif hasattr(self.plot_registry, 'data'):
                    self.plot_registry.data[signal_id] = np.copy(signal_data)
                # If this is a simple dictionary-like object
                elif hasattr(self.plot_registry, 'signals'):
                    self.plot_registry.signals[signal_id] = np.copy(signal_data)
                else:
                    logger.warning(f"No suitable interface found to register signal {signal_id}")
                    
            except Exception as e:
                logger.error(f"Error updating signal {signal_id}: {e}", exc_info=True)
    
    def _get_registry_signals(self):
        """Get all signals from the signal registry."""
        signals = {}
        
        try:
            # If registry has signals attribute (dictionary), use it directly
            if hasattr(self.signal_registry, 'signals'):
                if hasattr(self.signal_registry, 'lock'):
                    with self.signal_registry.lock:
                        signals = self.signal_registry.signals.copy()
                else:
                    signals = self.signal_registry.signals.copy()
            
            # If registry has get_all_signals method, use that
            elif hasattr(self.signal_registry, 'get_all_signals') and callable(self.signal_registry.get_all_signals):
                signals = self.signal_registry.get_all_signals()
            
            # Otherwise try to access individual signals
            elif hasattr(self.signal_registry, 'get_signal') and callable(self.signal_registry.get_signal):
                # Try to get a list of signal IDs somehow
                signal_ids = []
                if hasattr(self.signal_registry, 'get_signal_ids') and callable(self.signal_registry.get_signal_ids):
                    signal_ids = self.signal_registry.get_signal_ids()
                elif hasattr(self.signal_registry, 'signals') and isinstance(self.signal_registry.signals, dict):
                    signal_ids = list(self.signal_registry.signals.keys())
                
                for signal_id in signal_ids:
                    signal_data = self.signal_registry.get_signal(signal_id)
                    if signal_data is not None:
                        signals[signal_id] = signal_data
            
        except Exception as e:
            logger.error(f"Error getting signals from registry: {e}")
        
        return signals
    
    def _get_signal_metadata(self, signal_id):
        """Get metadata for a signal from the signal registry."""
        try:
            if hasattr(self.signal_registry, 'get_signal_metadata') and callable(self.signal_registry.get_signal_metadata):
                return self.signal_registry.get_signal_metadata(signal_id)
            
            # Try alternative methods
            if hasattr(self.signal_registry, 'metadata') and isinstance(self.signal_registry.metadata, dict):
                if signal_id in self.signal_registry.metadata:
                    return self.signal_registry.metadata[signal_id]
                
            # Default metadata if none available
            return {
                'color': (180, 180, 180),
                'source': 'unknown',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for signal {signal_id}: {e}")
            return {
                'color': (180, 180, 180),
                'source': 'unknown',
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _filter_signals_for_view_mode(self, signals):
        """
        Filter signals based on the current view mode.
        
        Args:
            signals (dict): Dictionary of signal_id -> signal_data
            
        Returns:
            dict: Filtered dictionary of signals appropriate for the current view mode
        """
        if not signals or self.current_view_mode is None:
            return signals
        
        filtered = {}
        
        try:
            # Define filtering strategies for different view modes
            if hasattr(self.current_view_mode, 'name'):
                view_name = self.current_view_mode.name
            else:
                # If the view_mode isn't an enum with a name attribute, convert to string
                view_name = str(self.current_view_mode)
            
            # Filter based on view mode
            if "RAW" in view_name:
                # For RAW view, only show signals with "RAW" in the name
                for signal_id, signal_data in signals.items():
                    if isinstance(signal_id, str) and "RAW" in signal_id:
                        filtered[signal_id] = signal_data
            
            elif "PROCESSED" in view_name:
                # For PROCESSED view, only show signals with "PROCESSED" or "PROC" in the name
                for signal_id, signal_data in signals.items():
                    if isinstance(signal_id, str) and ("PROCESSED" in signal_id or "PROC" in signal_id):
                        filtered[signal_id] = signal_data
            
            elif "TWIN" in view_name:
                # For TWIN view, include all signals
                filtered = signals.copy()
            
            elif "SETTINGS" in view_name:
                # For SETTINGS view, don't include any signals
                pass
            
            else:
                # For unknown view modes, include all signals
                filtered = signals.copy()
                
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            # Return all signals as fallback
            filtered = signals.copy()
        
        return filtered