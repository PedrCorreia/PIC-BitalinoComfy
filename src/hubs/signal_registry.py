import numpy as np
import time

class SignalRegistry:
    _instance = None

    @staticmethod
    def get_instance():
        if SignalRegistry._instance is None:
            SignalRegistry._instance = SignalRegistry()
            print("[DEBUG-REGISTRY] Created new SignalRegistry instance")
        return SignalRegistry._instance

    @staticmethod
    def reset():
        """Reset the registry - useful for debugging"""
        if SignalRegistry._instance is not None:
            print("[DEBUG-REGISTRY] Resetting registry")
            old_count = len(SignalRegistry._instance.signals)
            SignalRegistry._instance.signals = {}
            print(f"[DEBUG-REGISTRY] Registry reset complete. Cleared {old_count} signals.")
        else:
            print("[DEBUG-REGISTRY] No registry instance to reset")
            # Create new instance if it doesn't exist
            SignalRegistry.get_instance()

    def __init__(self):
        self.signals = {}
        self.creation_time = time.time()
        print("[DEBUG-REGISTRY] Signal registry initialized with empty signals dict")

    def register_signal(self, signal_id, signal_tensor):
        """Register a signal with its ID"""
        self.signals[signal_id] = signal_tensor
        print(f"[DEBUG-REGISTRY] Signal '{signal_id}' registered in pool")
        print(f"[DEBUG-REGISTRY] Current registry size: {len(self.signals)} signals")
        print(f"[DEBUG-REGISTRY] Available signal IDs: {list(self.signals.keys())}")

    def get_signal(self, signal_id):
        """Get a signal by its ID"""
        print(f"[DEBUG-REGISTRY] Looking up signal ID: '{signal_id}'")
        print(f"[DEBUG-REGISTRY] Available IDs: {list(self.signals.keys())}")

        if signal_id in self.signals:
            signal = self.signals[signal_id]
            print(f"[DEBUG-REGISTRY] Found signal '{signal_id}' in pool, shape: {signal.shape}")
            return signal
        print(f"[WARNING-REGISTRY] Signal '{signal_id}' not found in pool")
        return None

    def get_all_signals(self):
        """Return a dictionary of all registered signals"""
        return self.signals.copy()