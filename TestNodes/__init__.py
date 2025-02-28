from .periodic_signal import SineNode, CosineNode, NoiseNode, PeriodicNode
from .tools import PlotNode, PerSumNode2, PerSumNode3, PerSumNode4, SavePlot, PreviewPlot, SavePlotCustom, PreviewPlotCustom
from .processing_tools import FFTNode, LowPassFilterNode, HighPassFilterNode, BandPassFilterNode, FilterNode, PreviewFFTNode, FrequencySamplingNode, SaveFFT, SaveFFTCustom, PreviewFFTCustom, SavePlot_H, PreviewPlot_H,DiscreteTransferFunctionNode

NODE_CLASS_MAPPINGS = {
    "PIC/Obsolete/Modulated Signals/Sin": SineNode,
    "PIC/Obsolete/Modulated Signals/Cos": CosineNode,
    "PIC/Active/Modulated Signals/Noise": NoiseNode,
    "PIC/Active/Modulated Signals/Periodic": PeriodicNode,
    "PIC/Active/Tools/Sum2": PerSumNode2,
    "PIC/Active/Tools/Sum3": PerSumNode3,
    "PIC/Active/Tools/Sum4": PerSumNode4,
    "PIC/Active/Tools/Plotter": PlotNode,
    "PIC/Active/Tools/SavePlot": SavePlot,
    "PIC/Active/Tools/PreviewPlot": PreviewPlot,
    "PIC/Active/Tools/SavePlotC": SavePlotCustom,
    "PIC/Active/Tools/PreviewPlotC": PreviewPlotCustom,
    "PIC/Active/Basic Signal Processing/FFT": FFTNode,
    "PIC/Obsolete/Basic Signal Processing/LowPassFilter": LowPassFilterNode,
    "PIC/Obsolete/Basic Signal Processing/HighPassFilter": HighPassFilterNode,
    "PIC/Obsolete/Basic Signal Processing/BandPassFilter": BandPassFilterNode,
    "PIC/Active/Basic Signal Processing/Filter": FilterNode,
    "PIC/Active/Basic Signal Processing/SaveFFT": SaveFFT,
    "PIC/Active/Basic Signal Processing/PreviewFFT": PreviewFFTNode,
    "PIC/Active/Basic Signal Processing/SaveFFTCustom": SaveFFTCustom,
    "PIC/Active/Basic Signal Processing/PreviewFFTCustom": PreviewFFTCustom,
    "PIC/Obsolete/Basic Signal Processing/DiscreteTransferFunction": DiscreteTransferFunctionNode,
    "PIC/Active/Basic Signal Processing/FrequencySamplingNode": FrequencySamplingNode,
    "PIC/Active/Basic Signal Processing/SavePlot_H": SavePlot_H,
    "PIC/Active/Basic Signal Processing/PreviewPlot_H": PreviewPlot_H,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIC/Obsolete/Modulated Signals/Sin": "📈 Sin",
    "PIC/Obsolete/Modulated Signals/Cos": "📈 Cos",
    "PIC/Active/Modulated Signals/Noise": "🔊 Noise",
    "PIC/Active/Modulated Signals/Periodic": "📈 Pure Periodic Signal",
    "PIC/Active/Tools/Sum2": "➕ Sum 2",
    "PIC/Active/Tools/Sum3": "➕ Sum 3",
    "PIC/Active/Tools/Sum4": "➕ Sum 4",
    "PIC/Active/Tools/Plotter": "📊 Plotter",
    "PIC/Active/Tools/SavePlot": "💾 Save Plot",
    "PIC/Active/Tools/PreviewPlot": "👁️ Preview Plot",
    "PIC/Active/Tools/SavePlotC": "💾 Save Plot Custom",
    "PIC/Active/Tools/PreviewPlotC": "👁️ Preview Plot Custom",
    "PIC/Active/Basic Signal Processing/FFT": "🔍 FFT",
    "PIC/Obsolete/Basic Signal Processing/LowPassFilter": "🔍 Low Pass Filter",
    "PIC/Obsolete/Basic Signal Processing/HighPassFilter": "🔍 High Pass Filter",
    "PIC/Obsolete/Basic Signal Processing/BandPassFilter": "🔍 Band Pass Filter",
    "PIC/Active/Basic Signal Processing/Filter": "🔍 Filter",
    "PIC/Active/Basic Signal Processing/SaveFFT": "💾 Save FFT",
    "PIC/Active/Basic Signal Processing/PreviewFFT": "👁️ Preview FFT",
    "PIC/Active/Basic Signal Processing/SaveFFTCustom": "💾 Save FFT Custom",
    "PIC/Active/Basic Signal Processing/PreviewFFTCustom": "👁️ Preview FFT Custom",
    "PIC/Obsolete/Basic Signal Processing/DiscreteTransferFunction": "🔍 Discrete Transfer Function",
    "PIC/Active/Basic Signal Processing/FrequencySamplingNode": "🔍 Frequency Sampling Node",
    "PIC/Active/Basic Signal Processing/SavePlot_H": "💾 Save Plot Transfer Function",
    "PIC/Active/Basic Signal Processing/PreviewPlot_H": "👁️ Preview Plot Transfer Function",
}