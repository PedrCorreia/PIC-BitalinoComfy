from .periodic_signal import SineNode, CosineNode, PerSumNode2, PerSumNode3, PerSumNode4
from .tools import PlotNode
from .processing_tools import FFTNode, LowPassFilterNode, HighPassFilterNode, BandPassFilterNode

NODE_CLASS_MAPPINGS = {
    "Periodic Signals/Sin": SineNode,
    "Periodic Signals/Cos": CosineNode,
    "Periodic Signals/PeriodicSum2": PerSumNode2,
    "Periodic Signals/PeriodicSum3": PerSumNode3,
    "Periodic Signals/PeriodicSum4": PerSumNode4,
    "Tools/Plotter": PlotNode,
    "Tools/FFT": FFTNode,
    "Tools/LowPassFilter": LowPassFilterNode,
    "Tools/HighPassFilter": HighPassFilterNode,
    "Tools/BandPassFilter": BandPassFilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Periodic Signals/Sin": "📈 Sin",
    "Periodic Signals/Cos": "📈 Cos",
    "Periodic Signals/PeriodicSum2": "➕ Periodic Sum 2",
    "Periodic Signals/PeriodicSum3": "➕ Periodic Sum 3",
    "Periodic Signals/PeriodicSum4": "➕ Periodic Sum 4",
    "Tools/Plotter": "📊 Plotter",
    "Tools/FFT": "🔍 FFT",
    "Tools/LowPassFilter": "🔍 Low Pass Filter",
    "Tools/HighPassFilter": "🔍 High Pass Filter",
    "Tools/BandPassFilter": "🔍 Band Pass Filter",
}

