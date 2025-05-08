# PIC-BitalinoComfy

This repository contains custom nodes and utilities for signal processing and visualization, designed to work with the ComfyUI framework. It provides tools for real-time signal analysis, filtering, and plotting.

---

## Project Structure

The project is organized into two main directories:

### 1. `src/`
This directory contains core utility scripts and processing logic that are independent of the ComfyUI framework. These scripts handle the core signal processing and visualization tasks.

- **`plot.py`**:  
  This script provides classes for real-time plotting and visualization of signals. It includes:
  - **`DataPlot`**: A class for plotting time-domain signals with features like grid display, peak detection, and dynamic scaling.
  - **`STFTPlot`**: A class for visualizing the Short-Time Fourier Transform (STFT) of signals, showing frequency-domain information.
  - **`CombinedPlot`**: A class that combines the time-domain and STFT plots into a single interface, with interactive toggles for grid, peaks, and STFT display.

  The `plot.py` script is designed for real-time signal visualization and supports dynamic updates, making it ideal for live signal monitoring.

- **`signalprocessing.py`**:  
  This script contains the `SignalProcessing` class, which provides a comprehensive set of signal processing utilities, including:
  - **STFT Computation**: Compute the Short-Time Fourier Transform for frequency-domain analysis.
  - **Peak Detection**: Detect peaks in signals with optional smoothing and normalization.
  - **Filtering**: Apply low-pass, high-pass, and band-pass filters using Butterworth filters.
  - **Moving Average**: Smooth signals using a moving average filter.
 
  - **Phase-Locked Loop (PLL)**: Synchronize a signal with a reference signal using a PLL.

  The `signalprocessing.py` script is the backbone of the project, providing the core signal processing functionality.

---

### 2. `comfy/`
This directory contains custom nodes designed specifically for the ComfyUI framework. These nodes integrate the core signal processing utilities into the ComfyUI environment, allowing users to perform advanced signal analysis and filtering directly within the UI.

#### Nodes:
- **`MovingAverageFilter`**: Applies a moving average filter to smooth signals.
- **`SignalPLL`**: Applies a Phase-Locked Loop (PLL) to synchronize a signal with a reference signal.
- **`SignalFilter`**: Provides low-pass, high-pass, and band-pass filtering options.
  - Supports configurable cutoff frequencies and sampling rates.

Each node is designed to be modular and reusable, making it easy to integrate into various signal processing workflows.

---

## Features

- **Real-Time Signal Visualization**:  
  The `CombinedPlot` class in `plot.py` enables real-time visualization of time-domain and frequency-domain signals.

- **Comprehensive Signal Processing**:  
  The `SignalProcessing` class provides a wide range of utilities for signal analysis, filtering, and transformation.

- **ComfyUI Integration**:  
  Custom nodes in the `comfy/` directory allow seamless integration of signal processing capabilities into the ComfyUI framework.

---

## How to Use

1. **Install Dependencies**:
   Ensure you have the required Python libraries installed:
   ```bash
   pip install numpy scipy matplotlib opencv-python-headless torch
   ```

2. **Run the Plotting Script**:
   To visualize signals in real-time, run the `plot.py` script:
   ```bash
   python src/plot.py
   ```

3. **Use Custom Nodes in ComfyUI**:
   Place the `comfy/` directory in your ComfyUI custom nodes folder. The nodes will appear under the "PIC/Filters" category in the ComfyUI interface.

---

## Future Improvements

- Add more advanced signal processing nodes (e.g., wavelet transforms, adaptive filtering).
- Optimize performance for large-scale real-time signal processing.
- Enhance the visualization capabilities with more interactive features.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
