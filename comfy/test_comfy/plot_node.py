import threading
import numpy as np

class NodeKO:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default":"Hey Hey!"}),
                "text2": ("STRING", {"forceInput": True}),				
            },
        }

    RETURN_TYPES = ("STRING","INT", "FLOAT", 'LATENT', "CONDITIONING", "IMAGE", "MODEL")
    RETURN_NAMES = ("TxtO", "IntO", "FloatO", "Latent output. Really cool, huh?", "A condition" , "Our image." , "Mo mo modell!!!")

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "nodeKO"

    def test(self):
        return ()

class PygamePlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("LIST", {}),
                "y": ("LIST", {}),
                "as_points": ("BOOLEAN", {"default": False}),
            }
        }
    CATEGORY = "Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True

    def run(self, x, y, as_points):
        def plot_with_pygame(x, y, as_points):
            import pygame
            w, h = 640, 480
            pygame.init()
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Pygame Plot")
            screen.fill((0,0,0))
            x = np.array(x)
            y = np.array(y)
            x_norm = ((x - x.min()) / (x.max() - x.min()) * (w - 40) + 20).astype(int)
            y_norm = (h - ((y - y.min()) / (y.max() - y.min()) * (h - 40) + 20)).astype(int)
            pygame.draw.line(screen, (255,255,255), (20, h-20), (w-20, h-20), 2)
            pygame.draw.line(screen, (255,255,255), (20, h-20), (20, 20), 2)
            if as_points:
                for i in range(len(x)):
                    pygame.draw.circle(screen, (0,0,255), (x_norm[i], y_norm[i]), 2)
            else:
                pts = list(zip(x_norm, y_norm))
                if len(pts) > 1:
                    pygame.draw.lines(screen, (0,0,255), False, pts, 2)
            pygame.display.flip()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            pygame.quit()
        threading.Thread(target=plot_with_pygame, args=(x, y, as_points), daemon=True).start()
        return ()

class PyQtGraphPlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("LIST", {}),
                "y": ("LIST", {}),
                "as_points": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True
    def run(self, x, y, as_points):
        def plot_with_pyqtgraph(x, y, as_points):
            import sys
            from PyQt6.QtWidgets import QApplication
            import pyqtgraph as pg
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                created_app = True
            else:
                created_app = False
            win = pg.plot(title="PyQtGraph Plot")
            if as_points:
                win.plot(x, y, pen=None, symbol='o', symbolBrush='b')
            else:
                win.plot(x, y, pen=pg.mkPen('b', width=2))
            win.show()
            if created_app:
                app.exec()
        threading.Thread(target=plot_with_pyqtgraph, args=(x, y, as_points), daemon=True).start()
        return ()

class OpenCVPlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("LIST", {}),
                "y": ("LIST", {}),
                "as_points": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Plot"
    RETURN_TYPES = ()
    FUNCTION = "plot"
    OUTPUT_NODE = True
    def run(self, x, y, as_points):
        def plot_with_opencv(x, y, as_points):
            import cv2
            w, h = 640, 480
            img = np.zeros((h, w, 3), dtype=np.uint8)
            x = np.array(x)
            y = np.array(y)
            x_norm = ((x - x.min()) / (x.max() - x.min()) * (w - 40) + 20).astype(int)
            y_norm = (h - ((y - y.min()) / (y.max() - y.min()) * (h - 40) + 20)).astype(int)
            cv2.line(img, (20, h-20), (w-20, h-20), (255,255,255), 2)
            cv2.line(img, (20, h-20), (20, 20), (255,255,255), 2)
            if as_points:
                for i in range(len(x)):
                    cv2.circle(img, (x_norm[i], y_norm[i]), 2, (255,0,0), -1)
            else:
                pts = np.column_stack((x_norm, y_norm))
                cv2.polylines(img, [pts], isClosed=False, color=(255,0,0), thickness=2)
            cv2.imshow("OpenCV Plot", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        threading.Thread(target=plot_with_opencv, args=(x, y, as_points), daemon=True).start()
        return ()

NODE_CLASS_MAPPINGS = {
    "My KO Node": NodeKO,
    "PygamePlotNode": PygamePlotNode,
    "PyQtGraphPlotNode": PyQtGraphPlotNode,
    "OpenCVPlotNode": OpenCVPlotNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstNode": "My First Node",
    "PygamePlotNode": "Pygame Plot Node",
    "PyQtGraphPlotNode": "PyQtGraph Plot Node",
    "OpenCVPlotNode": "OpenCV Plot Node",
}