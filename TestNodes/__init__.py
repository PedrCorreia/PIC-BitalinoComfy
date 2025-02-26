from .periodic_signal import SineNode, CosineNode, NoiseNode, PeriodicNode
from .tools import PlotNode, PerSumNode2, PerSumNode3, PerSumNode4, SavePlot, PreviewPlot
from .processing_tools import FFTNode, LowPassFilterNode, HighPassFilterNode, BandPassFilterNode, FilterNode


NODE_CLASS_MAPPINGS = {
    "PIC/Obsolete/Modulated Signals/Sin": SineNode,
    "PIC/Obsolete/Modulated Signals/Cos": CosineNode,
    "PIC/Active/Modulated Signals/Noise": NoiseNode,
    "PIC/Active/Modulated Signals/Periodic": PeriodicNode,
    "PIC/Active/Tools/PeriodicSum2": PerSumNode2,
    "PIC/Active/Tools/PeriodicSum3": PerSumNode3,
    "PIC/Active/Tools/PeriodicSum4": PerSumNode4,
    "PIC/Active/Tools/Plotter": PlotNode,
    "PIC/Active/Tools/SavePlot": SavePlot,
    "PIC/Active/Tools/PreviewPlot": PreviewPlot,
    "PIC/Active/Basic Signal Processing/FFT": FFTNode,
    "PIC/Obsolete/Basic Signal Processing/LowPassFilter": LowPassFilterNode,
    "PIC/Obsolete/Basic Signal Processing/HighPassFilter": HighPassFilterNode,
    "PIC/Obsolete/Basic Signal Processing/BandPassFilter": BandPassFilterNode,
    "PIC/Active/Basic Signal Processing/Filter": FilterNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIC/Obsolete/Modulated Signals/Sin": "📈 Sin",
    "PIC/Obsolete/Modulated Signals/Cos": "📈 Cos",
    "PIC/Active/Modulated Signals/Noise": "🔊 Noise",
    "PIC/Active/Modulated Signals/Periodic": "📈 Pure Periodic Signal",
    "PIC/Active/Tools/PeriodicSum2": "➕ Periodic Sum 2",
    "PIC/Active/Tools/PeriodicSum3": "➕ Periodic Sum 3",
    "PIC/Active/Tools/PeriodicSum4": "➕ Periodic Sum 4",
    "PIC/Active/Tools/Plotter": "📊 Plotter",
    "PIC/Active/Tools/SavePlot": "💾 Save Plot",
    "PIC/Active/Tools/PreviewPlot": "👁️ Preview Plot",
    "PIC/Active/Basic Signal Processing/FFT": "🔍 FFT",
    "PIC/Obsolete/Basic Signal Processing/LowPassFilter": "🔍 Low Pass Filter",
    "PIC/Obsolete/Basic Signal Processing/HighPassFilter": "🔍 High Pass Filter",
    "PIC/Obsolete/Basic Signal Processing/BandPassFilter": "🔍 Band Pass Filter",
    "PIC/Active/Basic Signal Processing/Filter": "🔍 Filter",

}

