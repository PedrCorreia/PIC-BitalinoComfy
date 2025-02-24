from .periodic_signal import SineNode, CosineNode, NoiseNode,PeriodicNode
from .tools import PlotNode,PerSumNode2, PerSumNode3, PerSumNode4
from .processing_tools import FFTNode, LowPassFilterNode, HighPassFilterNode, BandPassFilterNode, FilterNode

NODE_CLASS_MAPPINGS = {
    "Periodic Signals/Sin": SineNode,
    "Periodic Signals/Cos": CosineNode,
    "Periodic Signals/Noise": NoiseNode,
    "Periodic Signals/Periodic": PeriodicNode,
    "Periodic Signals/PeriodicSum2": PerSumNode2,
    "Periodic Signals/PeriodicSum3": PerSumNode3,
    "Periodic Signals/PeriodicSum4": PerSumNode4,
    "Tools/Plotter": PlotNode,
    "Tools/FFT": FFTNode,
    "Tools/LowPassFilter": LowPassFilterNode,
    "Tools/HighPassFilter": HighPassFilterNode,
    "Tools/BandPassFilter": BandPassFilterNode,
    "Tools/Filter": FilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Periodic Signals/Sin": "📈 Sin",
    "Periodic Signals/Cos": "📈 Cos",
     "Periodic Signals/Periodic": "📈 Pure Periodic Signal",
    "Periodic Signals/PeriodicSum2": "➕ Periodic Sum 2",
    "Periodic Signals/PeriodicSum3": "➕ Periodic Sum 3",
    "Periodic Signals/PeriodicSum4": "➕ Periodic Sum 4",
    "Periodic Signals/Noise": "🔊 Noise",
    "Tools/Plotter": "📊 Plotter",
    "Tools/FFT": "🔍 FFT",
    "Tools/LowPassFilter": "🔍 Low Pass Filter",
    "Tools/HighPassFilter": "🔍 High Pass Filter",
    "Tools/BandPassFilter": "🔍 Band Pass Filter",
    "Tools/Filter": "🔍 Filter",
}

