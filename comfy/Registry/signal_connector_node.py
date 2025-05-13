import torch
import numpy as np
from ...src.plot.plot_unit import PlotUnit

class SignalConnectorNode:
    """
    Node that connects signals to the PlotUnit visualization hub
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal_id": ("STRING", {"default": "default_signal"}),
                "signal_type": (["P", "F"], {"default": "P"}),
                "connect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "signal": ("SIGNAL", ),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "connect_signal"
    CATEGORY = "Pedro_PIC/🧰 Tools"
    OUTPUT_NODE = True
    def __init__(self):
        # Initialize connection to the visualization hub
        self.plot_unit = PlotUnit.get_instance()
        self.plot_unit.start()
        self.connected_signals = {}
    
    def connect_signal(self, signal_id, signal_type, connect, signal=None):
        """Connect or disconnect a signal from the visualization hub"""
        status_msg = ""
        
        # Handle connect/disconnect logic
        if connect:
            if signal is None:
                status_msg = f"Error: No signal provided for {signal_id}"
            else:
                # Convert tensor to numpy if needed
                if isinstance(signal, torch.Tensor):
                    if len(signal.shape) > 1:
                        # Take the first row/channel if multidimensional
                        signal_data = signal[0].flatten().cpu().numpy()
                    else:
                        signal_data = signal.cpu().numpy()
                else:
                    signal_data = np.array(signal)
                
                # Determine color based on signal type
                color = (0, 180, 220) if signal_type == "F" else (220, 180, 0)
                
                # Send to visualization hub
                self.plot_unit.add_signal_data(signal_data, signal_id, signal_type, color)
                self.connected_signals[signal_id] = signal_type
                
                # Increment connected nodes counter
                self.plot_unit.increment_connected_nodes()
                
                status_msg = f"Signal {signal_id} connected as {signal_type} type"
        else:
            # Disconnect signal if it was previously connected
            if signal_id in self.connected_signals:
                self.plot_unit.remove_signal(signal_id)
                del self.connected_signals[signal_id]
                
                # Decrement connected nodes counter
                self.plot_unit.decrement_connected_nodes()
                
                status_msg = f"Signal {signal_id} disconnected"
            else:
                status_msg = f"Signal {signal_id} was not connected"
        
        return (status_msg,)
