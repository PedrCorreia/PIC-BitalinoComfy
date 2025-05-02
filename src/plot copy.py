import sys
import os
import cv2
import numpy as np
import cupy as cp  # Use CuPy for efficient GPU array operations
import torch  # Use PyTorch for optimized STFT
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor  # For threading
from queue import Queue  # For thread-safe data sharing
import time  # For precise timing
import pycuda.driver as cuda  # Use PyCUDA for GPU memory management
import pycuda.autoinit  # Automatically initialize PyCUDA

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalprocessing import SignalProcessing


class MixedPlot:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = deque(maxlen=1000)
        self.grid = False
        self.show_legend = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.peak_indices = []
        self.stft_data = None

        # Create a persistent Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        # Configure the Matplotlib plot
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(self.grid)
        self.line, = self.ax.plot([], [], color="blue", label="Signal")
        self.peak_line, = self.ax.plot([], [], 'ro', label="Peaks")
        self.legend = self.ax.legend()

    def toggle_legend(self):
        self.show_legend = not self.show_legend
        self.legend.set_visible(self.show_legend)

    def add_data(self, timestamp, value):
        self.data.append((timestamp, value))

    def set_peaks(self, peaks):
        if not self.data:
            return
        timestamps, values = zip(*self.data)
        self.peak_indices = [(timestamps[idx], values[idx]) for idx in peaks if 0 <= idx < len(timestamps)]

    def set_stft_data(self, stft_data):
        self.stft_data = stft_data

    def update_plot(self, show_peaks=False):
        if not self.data:
            return

        # Extract timestamps and values
        timestamps, values = zip(*self.data)

        # Update Matplotlib line data
        self.line.set_data(timestamps, values)
        self.ax.set_xlim(timestamps[0], timestamps[-1])

        # Dynamically calculate y-axis limits based on mean and standard deviation
        mean_value = np.mean(values)
        std_value = np.std(values)
        self.ax.set_ylim(mean_value - 3 * std_value, mean_value + 3 * std_value)

        # Update peaks if enabled
        if show_peaks and self.peak_indices:
            peak_timestamps, peak_values = zip(*self.peak_indices)
            self.peak_line.set_data(peak_timestamps, peak_values)
        else:
            self.peak_line.set_data([], [])

        # Redraw the canvas
        self.canvas.draw()

    def draw(self, img, x0, y0, x1, y1, show_peaks=False, show_stft=False):
        # Dynamically adjust the Matplotlib figure size to match the OpenCV window area
        self.fig.set_size_inches((x1 - x0) / 100, (y1 - y0) / 100)

        # Update the Matplotlib plot
        self.update_plot(show_peaks=show_peaks)

        # Ensure the canvas is initialized before rendering
        self.canvas.draw()

        # Render the plot to an image
        buf = self.canvas.buffer_rgba()
        plot_image = np.asarray(buf)
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
        plot_image = cv2.resize(plot_image, (x1 - x0, y1 - y0))
        img[y0:y1, x0:x1] = plot_image

        # Optionally draw STFT using Matplotlib
        if show_stft and self.stft_data:
            f, t, Zxx = self.stft_data
            fig, ax = plt.subplots(figsize=((x1 - x0) / 100, (y1 - y0) / 100), dpi=100)
            canvas = FigureCanvas(fig)
            # Dynamically adjust the y-axis (frequency) to show only relevant data
            max_frequency = np.max(f)
            ax.pcolormesh(t, f, Zxx, vmin=0, shading='gouraud')
            ax.set_ylim(0, max_frequency)  # Set y-axis to the maximum frequency
            ax.set_xlabel("Time [sec]")
            ax.set_ylabel("Frequency [Hz]")
            canvas.draw()
            stft_image = np.asarray(canvas.buffer_rgba())
            stft_image = cv2.cvtColor(stft_image, cv2.COLOR_RGB2BGR)
            stft_image = cv2.resize(stft_image, (x1 - x0, y1 - y0))
            img[y0:y1, x0:x1] = stft_image
            plt.close(fig)

        return img


class CombinedPlot:
    def __init__(self, width=1200, height=600, grid=False):
        self.width = width
        self.height = height
        self.mixed_plot = MixedPlot(width, height)
        self.show_peaks = False
        self.show_stft = False
        self.grid_on = grid
        self.executor = ThreadPoolExecutor(max_workers=2)  # Thread pool for background tasks

    def toggle_grid(self):
        self.grid_on = not self.grid_on
        self.mixed_plot.grid = self.grid_on
        self.mixed_plot.ax.grid(self.grid_on)

    def toggle_peaks(self):
        self.show_peaks = not self.show_peaks

    def toggle_stft(self):
        self.show_stft = not self.show_stft

    def toggle_legend(self):
        self.mixed_plot.toggle_legend()

    def draw(self, window_title="Plot Window"):
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Define plot area
        x0, y0, x1, y1 = 50, 50, self.width - 50, self.height - 100
        img = self.mixed_plot.draw(img, x0, y0, x1, y1, show_peaks=self.show_peaks, show_stft=self.show_stft)

        # Draw buttons
        button_bar_y = self.height - 40
        cv2.rectangle(img, (0, button_bar_y), (self.width, self.height), (220, 220, 220), -1)

        button_width = 100
        button_height = 20
        spacing = 10
        buttons = [
            ("Grid", self.grid_on),
            ("Peaks", self.show_peaks),
            ("STFT", self.show_stft),
            ("Legend", self.mixed_plot.show_legend),
        ]
        for i, (label, active) in enumerate(buttons):
            x = self.width - (len(buttons) - i) * (button_width + spacing)
            y = button_bar_y + 10
            color = (0, 0, 0) if active else (150, 150, 150)
            cv2.rectangle(img, (x, y), (x + button_width, y + button_height), color, -1)
            cv2.putText(img, label, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(img, window_title, (self.width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return img

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_bar_y = self.height - 40
            button_width = 100
            button_height = 20
            spacing = 10
            buttons = [
                ("Grid", self.toggle_grid),
                ("Peaks", self.toggle_peaks),
                ("STFT", self.toggle_stft),
                ("Legend", self.toggle_legend),
            ]
            for i, (_, action) in enumerate(buttons):
                bx = self.width - (len(buttons) - i) * (button_width + spacing)
                by = button_bar_y + 10
                if bx <= x <= bx + button_width and by <= y <= by + button_height:
                    action()

    def show(self, window_name="Plot Window"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.handle_mouse_event, param=None)

    def update(self, window_name="Plot Window"):
        img = self.draw(window_title=window_name)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return False
        return True


if __name__ == "__main__":
    plot = CombinedPlot(grid=True)
    sampling_rate = 1000
    signal_duration = 2.5  # Display 2.5 seconds of data
    signal_processor = SignalProcessing(signal=deque(maxlen=int(signal_duration * sampling_rate)))
    data_queue = Queue(maxsize=10)  # Thread-safe queue for live data

    plot.show(window_name="Plot Window")

    def generate_data():
        current_time = 0.0
        while True:
            start_time = time.time()
            for _ in range(sampling_rate):  # Generate 1 second of data
                current_time += 1 / sampling_rate
                # Oscillate frequency between 10 Hz and 20 Hz
                frequency = 10 + 10 * np.sin(2 * np.pi * 0.1 * current_time)  # Oscillates at 0.1 Hz
                value = 30 + 30 * np.sin(2 * np.pi * frequency * current_time)

                data_queue.put((current_time, value))  # Add data to the queue

            # Ensure real-time pacing
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 1 - elapsed_time))  # Sleep to maintain 1-second intervals

    def process_data():
        hann_window = torch.hann_window(256, device="cuda")  # Use a Hann window to reduce spectral leakage
        signal_buffer = deque(maxlen=256)  # Fixed-size buffer for STFT computation
        while True:
            try:
                # Process all available data in the queue
                while not data_queue.empty():
                    timestamp, value = data_queue.get()
                    signal_processor.add_sample(value)
                    plot.mixed_plot.add_data(timestamp, value)
                    signal_buffer.append(value)  # Add value to the fixed-size buffer

                # Compute peaks if enabled
                if plot.show_peaks:
                    peaks = signal_processor.detect_peaks(threshold=0.5, smooth_window=11, normalize=True)
                    plot.mixed_plot.set_peaks(peaks)

                # Compute STFT on the latest data if enabled
                if plot.show_stft and len(signal_buffer) == 256:
                    # Use the fixed-size buffer for STFT computation
                    signal_tensor = torch.tensor(cp.asarray(signal_buffer), dtype=torch.float32).to("cuda")
                    stft_result = torch.stft(
                        signal_tensor,
                        n_fft=256,
                        hop_length=128,
                        window=hann_window,
                        return_complex=True
                    )
                    # Extract only positive frequencies
                    f = torch.fft.rfftfreq(256, d=1 / sampling_rate).cpu().numpy()  # Positive frequencies only
                    t = np.arange(stft_result.shape[1]) * (128 / sampling_rate)
                    Zxx = stft_result.abs().cpu().numpy()[:len(f), :]  # Match positive frequencies
                    plot.mixed_plot.set_stft_data((f, t, Zxx))

                # Sleep briefly to allow for live updates
                time.sleep(0.01)  # 100 Hz update rate for STFT
            except Exception as e:
                print(f"Error in processing data: {e}")

    # Run data generation and processing in separate threads
    plot.executor.submit(generate_data)
    plot.executor.submit(process_data)

    try:
        while True:
            if not plot.update(window_name="Plot Window"):
                break
            time.sleep(0.01)  # Reduce update frequency to 10 Hz
    except KeyboardInterrupt:
        print("Exiting gracefully...")
        cv2.destroyAllWindows()
