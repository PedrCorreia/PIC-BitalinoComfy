import sys
import os
import cv2
import numpy as np
import torch  # Import PyTorch for STFT computations
from collections import deque
from scipy.signal import butter
from concurrent.futures import ThreadPoolExecutor  # For parallel processing

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalprocessing import SignalProcessing  # Use your SignalProcessing class


class DataPlot:
    def __init__(self, width, height, x_window_size=10, grid=False):
        self.width = width
        self.height = height
        self.x_window_size = x_window_size
        self.grid = grid
        self.data = deque(maxlen=1000)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.x_label = "Time (s)"
        self.y_label = "Amplitude"
        self.peak_indices = []  # Initialize peak indices

    def add_data(self, timestamp, value):
        self.data.append((timestamp, value))

    def set_peaks(self, peaks, delay_compensation=0):
        """
        Set the timestamps of detected peaks to be displayed, synchronized with the moving window.
        :param peaks: List of indices of detected peaks.
        :param delay_compensation: Number of samples to shift peaks for delay compensation.
        """
        timestamps, _ = zip(*self.data)
        t_min, t_max = timestamps[0], timestamps[-1]
        self.peak_indices = [
            timestamps[max(0, min(idx - delay_compensation, len(timestamps) - 1))]
            for idx in peaks
            if 0 <= idx < len(timestamps) and t_min <= timestamps[idx] <= t_max
        ]

    def draw(self, img, x0, y0, x1, y1):
        if not self.data:
            return img

        w, h = x1 - x0, y1 - y0

        # Draw grid box border
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

        # Draw grid
        if self.grid:
            for x in range(x0, x1, 50):
                cv2.line(img, (x, y0), (x, y1), (200, 200, 200), 1)
            for y in range(y0, y1, 50):
                cv2.line(img, (x0, y), (x1, y), (200, 200, 200), 1)

        # Extract timestamps and values
        timestamps, values = zip(*self.data)

        # Determine X and Y ranges dynamically
        t_min, t_max = timestamps[0], timestamps[-1]
        v_min, v_max = min(values), max(values)

        # Precompute scaling factors
        t_range = max(t_max - t_min, 1e-6)
        v_range = max(v_max - v_min, 1e-6)

        # Plot data
        prev_x, prev_y = None, None
        for t, v in self.data:
            x = x0 + int((t - t_min) / t_range * w)
            y = y1 - int((v - v_min) / v_range * h)
            if prev_x is not None and prev_y is not None:
                cv2.line(img, (prev_x, prev_y), (x, y), (0, 0, 255), 1)
            prev_x, prev_y = x, y

        # Draw peaks if enabled
        for t_peak in self.peak_indices:
            if t_min <= t_peak <= t_max:
                x_peak = x0 + int((t_peak - t_min) / t_range * w)
                y_peak = y1 - int((self.data[timestamps.index(t_peak)][1] - v_min) / v_range * h)
                cv2.circle(img, (x_peak, y_peak), 5, (0, 255, 0), -1)

        # Draw X-axis labels
        for i in range(0, len(timestamps), max(1, len(timestamps) // 10)):
            x_pos = x0 + int((timestamps[i] - t_min) / t_range * w)
            cv2.putText(img, f"{timestamps[i]:.1f}", (x_pos, y1 + 20), self.font, 0.4, (0, 0, 0), 1)
        cv2.putText(img, self.x_label, (x0 + w // 2 - 50, y1 + 40), self.font, 0.5, (0, 0, 0), 1)

        # Draw Y-axis labels
        for i in range(5):
            y_pos = y1 - int(i * h / 4)
            amplitude = v_min + i * (v_max - v_min) / 4
            cv2.putText(img, f"{amplitude:.1f}", (x0 - 40, y_pos + 5), self.font, 0.4, (0, 0, 0), 1)
        text_size = cv2.getTextSize(self.y_label, self.font, 0.5, 1)[0]
        text_x = x0 - 60 - text_size[0]
        text_y = y0 + h // 2 + text_size[1] // 2
        cv2.putText(img, self.y_label, (text_x, text_y), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img


class STFTPlot:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.stft_data = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.x_label = "Frequency (Hz)"  # X-axis now represents frequency
        self.y_label = "Magnitude"  # Y-axis represents magnitude

    def set_stft_data(self, stft_data):
        self.stft_data = stft_data

    def draw(self, img, x0, y0, x1, y1):
        if self.stft_data is None:
            return img

        f, _, magnitudes = self.stft_data  # Ignore time segments for FFT-like behavior
        w, h = x1 - x0, y1 - y0

        # Normalize magnitudes
        magnitudes = magnitudes.mean(axis=1)  # Average across time axis to get 1D array
        max_magnitude = magnitudes.max()
        if max_magnitude > 0:
            magnitudes /= max_magnitude  # Normalize to [0, 1]

        # Adjust Y-axis scaling to fit the frequency range
        f_min, f_max = f[0], f[-1]
        for i in range(1, len(f)):
            y_prev = y1 - int((f[i - 1] - f_min) / (f_max - f_min) * h)
            x_prev = x0 + int(magnitudes[i - 1] * w)
            y_curr = y1 - int((f[i] - f_min) / (f_max - f_min) * h)
            x_curr = x0 + int(magnitudes[i] * w)
            cv2.line(img, (x_prev, y_prev), (x_curr, y_curr), (0, 0, 255), 1)

        return img


class CombinedPlot:
    def __init__(self, width=1200, height=600, grid=False, x_window_size=10):
        self.width = width
        self.height = height
        self.data_plot = DataPlot(width, height, x_window_size, grid)
        self.stft_plot = STFTPlot(width, height)
        self.show_stft = False
        self.show_peaks = False
        self.grid_on = grid

    def toggle_grid(self):
        self.grid_on = not self.grid_on
        self.data_plot.grid = self.grid_on

    def toggle_stft(self):
        self.show_stft = not self.show_stft

    def toggle_peaks(self):
        self.show_peaks = not self.show_peaks

    def calculate_layout(self):
        if self.show_stft:
            plot_width = (self.width - 100) // 2
            data_area = (50, 50, 50 + plot_width, self.height - 100)
            stft_area = (50 + plot_width + 50, 50, self.width - 50, self.height - 100)
        else:
            data_area = (50, 50, self.width - 50, self.height - 100)
            stft_area = None
        return data_area, stft_area

    def draw(self, window_title="Plot Window"):
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        data_area, stft_area = self.calculate_layout()

        if data_area:
            x0, y0, x1, y1 = data_area
            img = self.data_plot.draw(img, x0, y0, x1, y1)

        if stft_area and self.show_stft:
            x0, y0, x1, y1 = stft_area
            img = self.stft_plot.draw(img, x0, y0, x1, y1)

        cv2.putText(img, window_title, (self.width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Draw buttons
        button_bar_y = self.height - 40
        cv2.rectangle(img, (0, button_bar_y), (self.width, self.height), (220, 220, 220), -1)

        button_width = 100
        button_height = 20
        spacing = 10
        buttons = [
            ("Grid", self.grid_on),
            ("STFT", self.show_stft),
            ("Peaks", self.show_peaks),
        ]
        for i, (label, active) in enumerate(buttons):
            x = self.width - (len(buttons) - i) * (button_width + spacing)
            y = button_bar_y + 10
            color = (0, 0, 0) if active else (150, 150, 150)
            cv2.rectangle(img, (x, y), (x + button_width, y + button_height), color, -1)
            cv2.putText(img, label, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_bar_y = self.height - 40
            button_width = 100
            button_height = 20
            spacing = 10
            buttons = [
                ("Grid", self.toggle_grid),
                ("STFT", self.toggle_stft),
                ("Peaks", self.toggle_peaks),
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


if __name__ == "__main__":
    plot = CombinedPlot(grid=True)
    current_time = 0.0
    sampling_rate = 1000
    signal_processor = SignalProcessing(signal=deque(maxlen=3000))

    # Example filter coefficients for delay compensation
    b, a = butter(4, 0.1, btype='low', fs=sampling_rate)
    group_delay = signal_processor.calculate_group_delay((b, a))

    plot.show(window_name="Plot Window")

    while True:
        current_time += 1 / sampling_rate
        value = 30 + 30 * np.sin(10 * np.pi * 2 * current_time) + np.random.normal(0, 5)
        signal_processor.add_sample(value)
        plot.data_plot.add_data(current_time, value)

        if plot.show_peaks:
            peaks = signal_processor.detect_peaks(threshold=0.5, smooth_window=11, normalize=True)
            plot.data_plot.set_peaks(peaks, delay_compensation=group_delay)

        if plot.show_stft and len(signal_processor.signal) >= 256:
            stft_data = signal_processor.compute_stft(window_size=256, overlap=128)
            plot.stft_plot.set_stft_data(stft_data)

        plot.update(window_name="Plot Window")
