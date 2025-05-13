import sys
import os
import logging
import time
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LoggerNode:
    """
    A simple logging node for ComfyUI that logs messages and signal data.
    Useful for debugging workflows, particularly with signal processing chains.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {"default": "Log message", "multiline": True})
            },
            "optional": {
                "level": (["INFO", "DEBUG", "WARNING", "ERROR"], {"default": "INFO"}),
                "signal_id": ("STRING", {"default": ""}),
                "signal_data": ("SIGNAL", ),
                "include_timestamp": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "log_message"
    CATEGORY = "signal/diagnostics"
    OUTPUT_NODE = True
    
    def __init__(self):
        # Configure logger
        self.logger = logging.getLogger('SignalLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Try to add file handler
            try:
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(
                    os.path.join(log_dir, f'signal_log_{time.strftime("%Y%m%d_%H%M%S")}.log')
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Log file created at {log_dir}")
            except Exception as e:
                self.logger.warning(f"Could not create log file: {e}")
        
        self.logger.info("Logger Node initialized")
    
    def log_message(self, message, level="INFO", signal_id="", signal_data=None, include_timestamp=True):
        """Log a message with optional signal data information"""
        log_method = getattr(self.logger, level.lower())
        
        # Format message with timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            message = f"[{timestamp}] {message}"
        
        # Add signal information if provided
        if signal_id:
            message = f"{message} | Signal ID: {signal_id}"
            
            if signal_data is not None:
                # Add basic info about the signal data
                try:
                    if hasattr(signal_data, 'shape'):
                        message = f"{message}, shape: {signal_data.shape}"
                    elif hasattr(signal_data, '__len__'):
                        message = f"{message}, length: {len(signal_data)}"
                    
                    # Add datatype info
                    message = f"{message}, type: {type(signal_data).__name__}"
                    
                    # Check for min/max if numeric data
                    if hasattr(signal_data, 'min') and hasattr(signal_data, 'max'):
                        try:
                            min_val = signal_data.min()
                            max_val = signal_data.max()
                            message = f"{message}, range: [{min_val:.2f}, {max_val:.2f}]"
                        except:
                            pass
                except Exception as e:
                    message = f"{message} (Error analyzing data: {str(e)})"
        
        # Log the message
        log_method(message)
        
        return (message,)

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoggerNode": LoggerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoggerNode": "Signal Logger"
}
