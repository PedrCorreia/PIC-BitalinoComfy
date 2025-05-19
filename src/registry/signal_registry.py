"""
SignalRegistry: Central registry for all generated signals.
Singleton pattern for global access.
"""
import threading
import builtins

class SignalRegistry:
    _lock = threading.Lock()

    def __init__(self):
        self._signals = {}

    @classmethod
    def get_instance(cls):
        # Use builtins to store the singleton globally
        if not hasattr(builtins, '_PIC25_SIGNAL_REGISTRY_SINGLETON'):
            setattr(builtins, '_PIC25_SIGNAL_REGISTRY_SINGLETON', cls())
        return getattr(builtins, '_PIC25_SIGNAL_REGISTRY_SINGLETON')

    def register_signal(self, signal_id, signal_obj, metadata=None):
        self._signals[signal_id] = {
            'data': signal_obj,
            'metadata': metadata or {}
        }

    def get_signal(self, signal_id):
        entry = self._signals.get(signal_id)
        return entry['data'] if entry else None

    def get_metadata(self, signal_id):
        entry = self._signals.get(signal_id)
        return entry['metadata'] if entry else None

    def set_metadata(self, signal_id, metadata):
        if signal_id in self._signals:
            self._signals[signal_id]['metadata'] = metadata

    def get_all_signals(self):
        return {k: v['data'] for k, v in self._signals.items()}

    def get_all_metadata(self):
        return {k: v['metadata'] for k, v in self._signals.items()}

    def remove_signal(self, signal_id):
        if signal_id in self._signals:
            del self._signals[signal_id]
