from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg
from collections import deque
from ..src.plot import Plot
from ..src.signal_processing import NumpySignalProcessor


class PyQtPlotNode:
    """
    Node for PyQt-based plotting (live or static).
    """
    DEFAULT_HEIGHT = 400
    DEFAULT_WIDTH = 700

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signals": ("LIST",),
                "mode": ("STRING", {"default": "static", "choices": ["live", "static"]})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "plot"
    CATEGORY = "Visualization"

    def plot(self, signals, mode="static"):
        if not isinstance(signals, list):
            raise ValueError("Expected a list of signal streams.")
        
        app = QApplication.instance() or QApplication([])
        win = pg.GraphicsLayoutWidget(show=True)
        n_channels = len(signals)

        for i, signal in enumerate(signals):
            plot = win.addPlot(row=i, col=0, title=f"Signal {i+1}")
            timestamps, values = zip(*signal)
            plot.plot(timestamps, values, pen=pg.mkPen(color=(255, 255, 0), width=1.2))
            plot.showGrid(x=True, y=True, alpha=0.3)

        if mode == "live":
            def update():
                for i, signal in enumerate(signals):
                    timestamps, values = zip(*signal)
                    win.getItem(i, 0).plot(timestamps, values, clear=True)

            timer = pg.QtCore.QTimer()
            timer.timeout.connect(update)
            timer.start(30)

        app.exec()

NODE_CLASS_MAPPINGS = {
    "PyQtPlotNode": PyQtPlotNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyQtPlotNode": "📊 PyQt Plot"
}
