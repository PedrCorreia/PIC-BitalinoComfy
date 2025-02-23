from .periodic_signal import SineNode, CosineNode, PerSumNode2, PerSumNode3, PerSumNode4
from .tools import TensorBifurcation, PlotNode
from .processing_tools import FFTNode, LowPassFilterNode, HighPassFilterNode

NODE_CLASS_MAPPINGS = {
    "Periodic Signals/SineWaveGenerator": SineNode,
    "Periodic Signals/CosineWaveGenerator": CosineNode,
    "Periodic Signals/PeriodicSum2": PerSumNode2,
    "Periodic Signals/PeriodicSum3": PerSumNode3,
    "Periodic Signals/PeriodicSum4": PerSumNode4,
    "Tools/TensorBifurcation": TensorBifurcation,
    "Tools/SineWavePlotter": PlotNode,
    "Tools/FFT": FFTNode,
    "Tools/LowPassFilter": LowPassFilterNode,
    "Tools/HighPassFilter": HighPassFilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Periodic Signals/SineWaveGenerator": "📈 Sine Wave Generator",
    "Periodic Signals/CosineWaveGenerator": "📉 Cosine Wave Generator",
    "Periodic Signals/PeriodicSum2": "➕ Periodic Sum 2",
    "Periodic Signals/PeriodicSum3": "➕ Periodic Sum 3",
    "Periodic Signals/PeriodicSum4": "➕ Periodic Sum 4",
    "Tools/SineWavePlotter": "📊 Plotter",
    "Tools/FFT": "🔍 FFT",
    "Tools/LowPassFilter": "🔍 Low Pass Filter",
    "Tools/HighPassFilter": "🔍 High Pass Filter",
}

