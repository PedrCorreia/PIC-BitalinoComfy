"""
Integration layer between PlotRegistry and PlotUnit.
This module handles the connection between the registry and visualization systems.
"""

import logging
import numpy as np
import threading
import time
from .plot_registry import PlotRegistry

# Set up logger
logger = logging.getLogger('PlotRegistryIntegration')

class PlotRegistryIntegration:
    """
    Integration layer between PlotRegistry and PlotUnit.
    
    This class:
    1. Observes the PlotRegistry for changes
    2. Translates registry signals into PlotUnit visualization data
    3. Handles connection tracking between nodes and signals
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the integration layer"""
        if PlotRegistryIntegration._instance is not None:
            raise RuntimeError("Use PlotRegistryIntegration.get_instance() instead")
            
        # Get the registry
        self.registry = PlotRegistry.get_instance()
        self.plot_unit = None  # Will be set when connected
        
        # Keep track of signals being visualized
        self._visualized_signals = set()
        self._node_connections = {}  # node_id -> set of signal_ids
        
        # Background thread for checking registry updates
        self._update_thread = None
        self._stop_event = threading.Event()
        self._update_interval = 0.5  # Check for updates every 500ms
        self._is_running = False
        
        logger.info("PlotRegistryIntegration initialized")
    
    def connect_plot_unit(self, plot_unit):
        """
        Connect a PlotUnit instance to this integration
        
        Args:
            plot_unit: The PlotUnit instance to connect
        """
        self.plot_unit = plot_unit
        logger.info("Connected PlotUnit to registry integration")
        
        # Start background update thread if not already running
        self._start_update_thread()
    
    def _start_update_thread(self):
        """Start background thread to check for registry updates"""
        if self._is_running:
            return
            
        if self._update_thread is not None and self._update_thread.is_alive():
            self._stop_event.set()
            self._update_thread.join(timeout=1.0)
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._registry_update_loop,
            daemon=True
        )
        self._update_thread.start()
        self._is_running = True
        logger.info("Started registry update thread")
    
    def _registry_update_loop(self):
        """Background thread loop to check for registry updates"""
        logger.info("Registry update thread started")
        
        while not self._stop_event.is_set():
            try:
                self._process_registry_updates()
            except Exception as e:
                logger.error(f"Error processing registry updates: {e}")
            
            # Wait for next update interval or until stopped
            self._stop_event.wait(self._update_interval)
        
        logger.info("Registry update thread stopped")
        self._is_running = False
    
    def _process_registry_updates(self):
        """Process updates from registry to PlotUnit"""
        if self.plot_unit is None:
            return
            
        # Get signals that should be visualized
        visualized_signals = set(self.registry.get_visualized_signals())
        
        # Check for new signals to visualize
        for signal_id in visualized_signals:
            if signal_id not in self._visualized_signals:
                # New signal to visualize
                self._add_signal_to_plot_unit(signal_id)
                self._visualized_signals.add(signal_id)
        
        # Update all current signals
        for signal_id in self._visualized_signals:
            self._update_signal_in_plot_unit(signal_id)
    
    def _add_signal_to_plot_unit(self, signal_id):
        """Add a signal from registry to PlotUnit"""
        signal_data = self.registry.get_signal(signal_id)
        metadata = self.registry.get_signal_metadata(signal_id)
        
        if signal_data is None:
            return
            
        # Convert to numpy if needed
        if hasattr(signal_data, 'numpy'):
            signal_data = signal_data.numpy()
        
        # Extract color from metadata if available
        color = None
        if metadata and 'color' in metadata:
            color = metadata['color']
        
        # Send to PlotUnit
        self.plot_unit.add_signal_data(signal_data, name=signal_id, color=color)
        logger.info(f"Added signal '{signal_id}' to PlotUnit")
    
    def _update_signal_in_plot_unit(self, signal_id):
        """Update an existing signal in PlotUnit"""
        signal_data = self.registry.get_signal(signal_id)
        metadata = self.registry.get_signal_metadata(signal_id)
        
        if signal_data is None:
            return
            
        # Convert to numpy if needed
        if hasattr(signal_data, 'numpy'):
            signal_data = signal_data.numpy()
        
        # Extract color from metadata if available
        color = None
        if metadata and 'color' in metadata:
            color = metadata['color']
        
        # Send to PlotUnit
        self.plot_unit.add_signal_data(signal_data, name=signal_id, color=color)
    
    def register_node(self, node_id):
        """
        Register a visualization node with the integration
        
        Args:
            node_id: ID of the node to register
        """
        if node_id not in self._node_connections:
            self._node_connections[node_id] = set()
            if self.plot_unit:
                self.plot_unit.increment_connected_nodes()
            logger.info(f"Node '{node_id}' registered with integration")
    
    def connect_node_to_signal(self, node_id, signal_id):
        """
        Connect a node to a signal for visualization
        
        Args:
            node_id: ID of the node
            signal_id: ID of the signal to connect to
        """
        # Register node if needed
        if node_id not in self._node_connections:
            self.register_node(node_id)
        
        # Add signal to node's connections
        self._node_connections[node_id].add(signal_id)
        
        # Tell registry about connection
        self.registry.connect_node_to_signal(node_id, signal_id)
        
        logger.info(f"Node '{node_id}' connected to signal '{signal_id}'")
    
    def disconnect_node(self, node_id):
        """
        Disconnect a node from all signals
        
        Args:
            node_id: ID of the node to disconnect
        """
        if node_id in self._node_connections:
            # Remove from local tracking
            del self._node_connections[node_id]
            
            # Tell registry about disconnection
            self.registry.disconnect_node(node_id)
            
            if self.plot_unit:
                self.plot_unit.decrement_connected_nodes()
                
            logger.info(f"Node '{node_id}' disconnected from all signals")
    
    def reset(self):
        """Reset the integration and registry"""
        # Reset the registry
        self.registry.reset()
        
        # Clear local tracking
        self._visualized_signals.clear()
        self._node_connections.clear()
        
        # Reset PlotUnit if connected
        if self.plot_unit:
            self.plot_unit.clear_plots()
            self.plot_unit.settings['connected_nodes'] = 0
            
        logger.info("PlotRegistryIntegration reset")

    def shutdown(self):
        """Shutdown the integration layer"""
        if self._is_running:
            self._stop_event.set()
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=1.0)
            self._is_running = False
        logger.info("PlotRegistryIntegration shut down")
