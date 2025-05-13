import threading
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Any
from src.registry.signal_registry import SignalRegistry

class RegistrySender:
    """
    Class for sending signals to the registry for visualization.
    Acts as a bridge between generators and the visualization system.
    """
    
    def __init__(self):
        """Initialize the RegistrySender"""
        # Get the registry instance
        self.registry = SignalRegistry.get_instance()
        print("[DEBUG-REGISTRY-SENDER] Registry sender initialized")
    
    def send_signal(self, signal_id: str, signal_data: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Send a signal to the registry
        
        Args:
            signal_id: Unique identifier for the signal
            signal_data: The signal data (tensor, numpy array, list)
            metadata: Optional metadata to store with the signal
        
        Returns:
            bool: True if signal was successfully registered
        """
        try:
            # Convert data to tensor if it's not already
            if isinstance(signal_data, np.ndarray):
                signal_tensor = torch.from_numpy(signal_data)
            elif isinstance(signal_data, list):
                # Try to convert to tensor, handling both 1D lists and list of tuples
                if len(signal_data) > 0 and isinstance(signal_data[0], tuple):
                    # For time-series data passed as [(time1, value1), (time2, value2), ...]
                    times, values = zip(*signal_data)
                    signal_tensor = torch.tensor(values).float()
                else:
                    # For simple value lists
                    signal_tensor = torch.tensor(signal_data).float()
            elif isinstance(signal_data, torch.Tensor):
                signal_tensor = signal_data
            else:
                print(f"[WARNING-REGISTRY-SENDER] Unsupported data type: {type(signal_data)}")
                return False
            
            # Create a package with tensor and metadata if provided
            if metadata:
                signal_package = {
                    'tensor': signal_tensor,
                    'metadata': metadata
                }
                self.registry.register_signal(signal_id, signal_package)
            else:
                # Register directly if no metadata
                self.registry.register_signal(signal_id, signal_tensor)
                
            print(f"[DEBUG-REGISTRY-SENDER] Signal '{signal_id}' sent to registry successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR-REGISTRY-SENDER] Failed to send signal '{signal_id}' to registry: {e}")
            return False
    
    def send_multi_signals(self, signals: Dict[str, Any], metadata: Optional[Dict[str, Dict]] = None) -> Dict[str, bool]:
        """
        Send multiple signals to the registry at once
        
        Args:
            signals: Dictionary mapping signal_id -> signal_data
            metadata: Optional dictionary mapping signal_id -> metadata
        
        Returns:
            Dict[str, bool]: Dictionary of signal_id -> success status
        """
        results = {}
        for signal_id, signal_data in signals.items():
            signal_metadata = metadata.get(signal_id) if metadata else None
            results[signal_id] = self.send_signal(signal_id, signal_data, signal_metadata)
        
        return results
