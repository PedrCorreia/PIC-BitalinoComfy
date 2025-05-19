#!/usr/bin/env python
"""
Plot Registry Adapter for PIC-2025
- Contains only adapter logic for connecting the registry to the UI/PlotUnit.
- Handles view mode, filtering, and provides signal IDs/data for the UI.
- No signal generation logic.
"""
import threading
import time
from src.registry.plot_registry import PlotRegistry


class PlotUnitRegistryAdapter:
    """
    Adapter that connects the PlotUnit visualization with the PlotRegistry.
    Handles only view mode, filtering, and data access for the UI.
    """
    def __init__(self, plot_unit=None):
        self.plot_registry = PlotRegistry.get_instance()
        self.plot_unit = plot_unit
        self.running = False
        self.thread = None
        self.last_update_time = time.time()
        self.blink_state = False
        self.blink_timer = 0
        self.signal_count = 0
        self.current_view_mode = None
        self.active_signal_types = {'raw', 'processed'}
        self.latency_values = []
        self.latency_window_size = 20
        self.connected_signals = set()
        self._node_connections = {}
        print("PlotUnitRegistryAdapter initialized")

    def connect(self):
        """Connect the PlotUnit to the PlotRegistry and start monitoring."""
        if self.running:
            print("Adapter already running")
            return
        if self.plot_unit is None:
            print("ERROR: No PlotUnit instance available")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_registry)
        self.thread.daemon = True
        self.thread.start()
        print("PlotUnitRegistryAdapter started - connecting PlotUnit to registry")

    def disconnect(self):
        """Disconnect the adapter and stop monitoring."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("PlotUnitRegistryAdapter stopped")

    def set_view_mode(self, view_mode):
        self.current_view_mode = view_mode
        print(f"Adapter: set_view_mode called with {view_mode}")

    def register_node(self, node_id):
        if node_id not in self._node_connections:
            self._node_connections[node_id] = set()
            if self.plot_unit and hasattr(self.plot_unit, 'increment_connected_nodes'):
                self.plot_unit.increment_connected_nodes()
            print(f"Node '{node_id}' registered with adapter")

    def connect_node_to_signal(self, node_id, signal_id):
        if not isinstance(signal_id, str):
            print(f"Attempted to connect node {node_id} to non-string signal ID: {signal_id} (type: {type(signal_id)})")
            return False
        if node_id not in self._node_connections:
            self.register_node(node_id)
        self._node_connections[node_id].add(signal_id)
        result = self.plot_registry.connect_node_to_signal(node_id, signal_id)
        if result:
            print(f"Node '{node_id}' connected to signal '{signal_id}'")
        return result

    def disconnect_node(self, node_id):
        if node_id in self._node_connections:
            signals = list(self._node_connections[node_id])
            for signal_id in signals:
                self.plot_registry.disconnect_node(node_id)
            del self._node_connections[node_id]
            if self.plot_unit and hasattr(self.plot_unit, 'decrement_connected_nodes'):
                self.plot_unit.decrement_connected_nodes()
            print(f"Node '{node_id}' disconnected from adapter")

    def get_connected_nodes(self):
        return list(self._node_connections.keys())

    def reset(self):
        self.plot_registry.reset()
        self._node_connections.clear()
        if self.plot_unit and hasattr(self.plot_unit, 'clear_plots'):
            self.plot_unit.clear_plots()
            if hasattr(self.plot_unit, 'settings'):
                self.plot_unit.settings['connected_nodes'] = 0
        print("PlotUnitRegistryAdapter reset")

    def shutdown(self):
        self.disconnect()
        print("PlotUnitRegistryAdapter shut down")

    def _monitor_registry(self):
        """
        Monitor the registry for updates and print signal IDs for debugging.
        This method runs in a separate thread.
        """
        print("Monitoring registry for updates...")
        while self.running:
            current_time = time.time()
            if current_time - self.last_update_time > 1.0:
                self.last_update_time = current_time
                all_signal_ids = self.plot_registry.get_all_signal_ids()
                self.signal_count = len(all_signal_ids)
                print(f"[Adapter] Signals in registry: {all_signal_ids}")
            time.sleep(0.1)
