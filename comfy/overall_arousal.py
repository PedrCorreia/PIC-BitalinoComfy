import numpy as np
import time
from ..src.registry.signal_registry import SignalRegistry
from ..src.utils.utils import Arousal

class OverallArousalNode:
    """
    ComfyUI Node that outputs overall arousal value calculated from all available arousal metrics.
    No inputs required - calculates from registry data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "hr_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "rr_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "scl_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "scr_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("Overall_Arousal",)
    FUNCTION = "get_overall_arousal"
    CATEGORY = "Pedro_PIC/ Arousal"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def __init__(self):
        self.registry = SignalRegistry.get_instance()
        self._last_overall_arousal = 0.5  # Default middle value


    def _get_arousal_level_description(self, arousal_value):
        """Convert arousal value to descriptive level"""
        if arousal_value < 0.2:
            return "sleep"
        elif arousal_value < 0.4:
            return "relaxed" 
        elif arousal_value < 0.6:
            return "normal"
        elif arousal_value < 0.8:
            return "aroused"
        else:
            return "stressed"

    def _get_metric_value(self, metric_id):
        """Get the latest value from a metric signal"""
        signal_data = self.registry.get_signal(metric_id)
        if signal_data:
            if "last" in signal_data and signal_data["last"] is not None:
                return float(signal_data["last"])
            elif "v" in signal_data and len(signal_data["v"]) > 0:
                return float(signal_data["v"][-1])
        return None

    def _calculate_weighted_overall_arousal(self, hr_weight=1.0, rr_weight=1.0, scl_weight=1.0, scr_weight=1.0):
        """Calculate weighted overall arousal using custom weights for each metric"""
        try:
            # Get raw metric values from the processing nodes
            hr = self._get_metric_value("HR_METRIC")  # From ECG processing
            rr = self._get_metric_value("RR_METRIC")  # From RR processing  
            scl = self._get_metric_value("SCL_METRIC")  # From EDA processing (tonic)
            scr = self._get_metric_value("SCR_METRIC")  # From EDA processing (phasic)
            
            print(f"[OverallArousal] Raw metrics - HR: {hr}, RR: {rr}, SCL: {scl}, SCR: {scr}")
            print(f"[OverallArousal] Weights - HR: {hr_weight}, RR: {rr_weight}, SCL: {scl_weight}, SCR: {scr_weight}")
            
            current_time = time.time()
            
            # Calculate individual arousal values
            weighted_scores = []
            total_weight = 0.0
            
            rr_arousal = None
            ecg_arousal = None  
            eda_arousal = None
            
            if rr is not None and rr_weight > 0:
                rr_arousal = Arousal.rr_arousal(rr)
                self._register_arousal_metric("RR_AROUSAL_METRIC", rr_arousal, "RR Arousal", current_time)
                weighted_scores.append(rr_arousal * rr_weight)
                total_weight += rr_weight
                
            if hr is not None and hr_weight > 0:
                ecg_arousal = Arousal.hr_arousal(hr)
                self._register_arousal_metric("ECG_AROUSAL_METRIC", ecg_arousal, "ECG Arousal", current_time)
                weighted_scores.append(ecg_arousal * hr_weight)
                total_weight += hr_weight
                
            if scl is not None and scl_weight > 0:
                eda_arousal = Arousal.scl_arousal(scl)
                self._register_arousal_metric("EDA_AROUSAL_METRIC", eda_arousal, "EDA Arousal", current_time)
                weighted_scores.append(eda_arousal * scl_weight)
                total_weight += scl_weight
            elif scr is not None and scr_weight > 0:
                # Fallback to SCR if SCL not available
                eda_arousal = Arousal.scr_arousal(scr)
                self._register_arousal_metric("EDA_AROUSAL_METRIC", eda_arousal, "EDA Arousal", current_time)
                weighted_scores.append(eda_arousal * scr_weight)
                total_weight += scr_weight
            
            # Calculate weighted overall arousal
            if weighted_scores and total_weight > 0:
                overall_arousal = sum(weighted_scores) / total_weight
            else:
                overall_arousal = 0.5  # Default when no data available
            
            print(f"[OverallArousal] Individual arousal - RR: {rr_arousal}, ECG: {ecg_arousal}, EDA: {eda_arousal}")
            print(f"[OverallArousal] Weighted overall arousal: {overall_arousal}")
            
            return float(overall_arousal)

        except Exception as e:
            print(f"Error calculating weighted arousal metrics: {e}")
            return 0.5

    def _register_arousal_metric(self, metric_id, arousal_value, label, timestamp):
        """Register an individual arousal metric in the registry"""
        if arousal_value is not None:
            arousal_data = {
                "t": [timestamp],
                "v": [float(arousal_value)],
                "last": float(arousal_value)
            }
            
            self.registry.register_signal(metric_id, arousal_data, {
                "id": metric_id,
                "type": "arousal_metrics",
                "label": label,
                "arousal_value": float(arousal_value),
                "arousal_level": self._get_arousal_level_description(arousal_value)
            })

    def get_overall_arousal(self, enabled=True, hr_weight=1.0, rr_weight=1.0, scl_weight=1.0, scr_weight=1.0):
        """Main function called by ComfyUI"""
        if not enabled:
            return (0.5,)  # Return default middle value when disabled
            
        try:
            # Calculate all arousal metrics (individual + weighted overall) and register them
            overall_arousal = self._calculate_weighted_overall_arousal(hr_weight, rr_weight, scl_weight, scr_weight)
            
            # Store last value for consistency
            self._last_overall_arousal = overall_arousal
            
            # Register the overall arousal metric with weight information in metadata
            current_time = time.time()
            overall_data = {
                "t": [current_time],
                "v": [float(overall_arousal)],
                "last": float(overall_arousal)
            }
            
            self.registry.register_signal("OVERALL_AROUSAL_METRIC", overall_data, {
                "id": "OVERALL_AROUSAL_METRIC",
                "type": "arousal_metrics",
                "label": "Overall Arousal",
                "arousal_value": float(overall_arousal),
                "arousal_level": self._get_arousal_level_description(overall_arousal),
                "weights": {
                    "hr_weight": hr_weight,
                    "rr_weight": rr_weight, 
                    "scl_weight": scl_weight,
                    "scr_weight": scr_weight
                }
            })
            
            return (overall_arousal,)
            
        except Exception as e:
            print(f"Error in OverallArousalNode: {e}")
            return (self._last_overall_arousal,)





# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OverallArousalNode": OverallArousalNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverallArousalNode": "🎯 Overall Arousal Metric"
}