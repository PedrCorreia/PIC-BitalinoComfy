import numpy as np
import torch
import uuid
import time
import threading
import logging
from ...src.plot.plot_unit import PlotUnit
from ...src.registry.plot_registry import PlotRegistry
from ...src.registry.plot_registry_integration import PlotRegistryIntegration
from ...src.registry.signal_registry import SignalRegistry

# Configure logger
logger = logging.getLogger('PlotUnitNode')

class PlotUnitNode:
    """
    A standalone visualization hub that displays signals from the registry in a persistent window.
    
    This node follows a background-monitoring approach:
    1. Connects to registries once at initialization
    2. Continuously monitors for signals without requiring ComfyUI execution
    3. Displays signals as they become available in real-time
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # We need minimal inputs since this will run in the background
        return {}
    
    RETURN_TYPES = ()  # No outputs
    OUTPUT_NODE = True
    FUNCTION = "run_visualization_hub"
    CATEGORY = "Pedro_PIC/🌊 Signal Registry"
    
    def __init__(self):
        # Generate a unique ID for this node
        self.node_id = f"plot_unit_{str(uuid.uuid4())[:8]}"
        
        # Get singleton instances
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        self.plot_registry = PlotRegistry.get_instance()
        self.signal_registry = SignalRegistry.get_instance()
        self.integration = PlotRegistryIntegration.get_instance()
        
        # Connect plot unit to integration
        self.integration.connect_plot_unit(self.plot_unit)
        
        # Register this node with the integration
        self.integration.register_node(self.node_id)
        
        # Add required methods if they don't exist
        if not hasattr(self.plot_unit, 'update'):
            self.plot_unit.update = self._update_fallback
        if not hasattr(self.plot_unit, 'clear_plots'):
            self.plot_unit.clear_plots = self._clear_plots_fallback
            
        # Background monitoring
        self._stop_monitor = threading.Event()
        self._monitor_thread = None
        self._last_update = time.time()
        self._update_interval = 0.1  # 100ms update interval
        self._monitoring = False
        
        # Start monitoring thread in the background
        self._start_monitor_thread()
        
        logger.info(f"[Plot Unit] Node {self.node_id} initialized with background monitoring")
        
    def __del__(self):
        """Clean up when the node is deleted"""
        try:
            # Stop background thread
            self._stop_monitor.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
                
            # Disconnect node from integration
            self.integration.disconnect_node(self.node_id)
            logger.info(f"[Plot Unit] Node {self.node_id} disconnected")
        except:
            # This might fail during shutdown, so we'll just ignore errors
            pass
            
    def _update_fallback(self):
        """Fallback method if PlotUnit doesn't have an update method"""
        logger.info("[Plot Unit] Using fallback update method")
        # Try to push a message to the event queue if it exists
        if hasattr(self.plot_unit, 'event_queue'):
            self.plot_unit.event_queue.put({
                'type': 'refresh',
                'timestamp': time.time()
            })
    
    def _clear_plots_fallback(self):
        """Fallback method if PlotUnit doesn't have a clear_plots method"""
        logger.info("[Plot Unit] Using fallback clear_plots method")
        # Try to clear data directly if possible
        if hasattr(self.plot_unit, 'data') and hasattr(self.plot_unit, 'data_lock'):
            with self.plot_unit.data_lock:
                for key in list(self.plot_unit.data.keys()):
                    self.plot_unit.data[key] = np.zeros(100)
    
    def _start_monitor_thread(self):
        """Start the background monitoring thread"""
        if self._monitoring:
            return
            
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_signals,
            daemon=True,
            name="PlotUnitMonitor"
        )
        self._monitor_thread.start()
        self._monitoring = True
        logger.info("[Plot Unit] Background monitoring started")
    
    def _monitor_signals(self):
        """Background thread to monitor signals and update visualization"""
        logger.info("[Plot Unit] Monitor thread started")
        
        while not self._stop_monitor.is_set():
            try:
                # Only update if enough time has passed
                now = time.time()
                if now - self._last_update >= self._update_interval:
                    # Connect signals from signal registry to plot registry
                    self._connect_signals()
                    
                    # Update visualization
                    self.plot_unit.update()
                    
                    # Update timestamp
                    self._last_update = now
            except Exception as e:
                logger.error(f"[Plot Unit] Error in monitor thread: {e}")
            
            # Sleep a short time
            time.sleep(0.05)
        
        logger.info("[Plot Unit] Monitor thread stopped")
        self._monitoring = False
    
    def _connect_signals(self):
        """Connect signals from SignalRegistry to PlotRegistry"""
        try:
            # Get all signals from SignalRegistry
            signal_ids = self.signal_registry.get_all_signals()
            
            # Process each signal
            for signal_id in signal_ids:
                # Check if this is already in PlotRegistry
                if self.plot_registry.get_signal(signal_id) is None:
                    # Get signal data from SignalRegistry
                    signal_data = self.signal_registry.get_signal(signal_id)
                    metadata = self.signal_registry.get_signal_metadata(signal_id)
                    
                    if signal_data is not None:
                        # Add to PlotRegistry
                        self.plot_registry.register_signal(signal_id, signal_data, metadata)
                        
                        # Connect this node to the signal
                        self.plot_registry.connect_node_to_signal(self.node_id, signal_id)
                        logger.info(f"[Plot Unit] Connected signal: {signal_id}")
                else:
                    # Update existing signal in PlotRegistry
                    signal_data = self.signal_registry.get_signal(signal_id)
                    metadata = self.signal_registry.get_signal_metadata(signal_id)
                    
                    if signal_data is not None:
                        # Update in PlotRegistry
                        self.plot_registry.register_signal(signal_id, signal_data, metadata)
        except Exception as e:
            logger.error(f"[Plot Unit] Error connecting signals: {e}")
    
    def run_visualization_hub(self):
        """
        Run the visualization hub.
        This function is minimal since the actual work happens in the background thread.
        """
        # Make sure the monitor thread is running
        if not self._monitoring or not self._monitor_thread.is_alive():
            self._start_monitor_thread()
            
        # Simply return since actual visualization happens in background
        return ()

# Node registration
NODE_CLASS_MAPPINGS = {
    "PlotUnitNode": PlotUnitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlotUnitNode": "📊 Signal Visualization Hub"
}
