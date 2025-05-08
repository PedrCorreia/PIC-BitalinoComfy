# PIC-BitalinoComfy

This repository provides tools and custom nodes for integrating PLUX/BITalino biosignal acquisition and processing into [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The focus is on real-time signal analysis, arousal measurement, and visualization, enabling further use in generative or interactive workflows.

---

## Overview

- **PLUX/BITalino Signal Integration:**  
  Scripts and nodes for acquiring signals (ECG, EDA, respiration, etc.) from BITalino devices using the PLUX API, and preparing them for use in ComfyUI.

- **Arousal Measurement:**  
  Tools and methods for analyzing physiological signals such as EDA, ECG, and respiration to derive arousal metrics. These metrics can be used in real-time or batch processing workflows, enabling integration into interactive or generative applications.

- **ComfyUI Node Integration:**  
  Modular nodes for signal filtering, feature extraction, and visualization, designed for seamless use in ComfyUI pipelines.

---

## Main Components

- **`report/bitalino_aq.py`**  
  Example script for acquiring signals from BITalino devices and saving them as JSON for further processing.

- **`src/signal_processing.py`**  
  Core utilities for signal filtering, normalization, peak detection, STFT, moving average, and more.

- **`src/rr_signal_processing.py`**  
  (Previously `respiration_rate_numpy.py`)  
  Respiration rate extraction and processing, refactored for future ComfyUI integration.

- **`src/ecg_signal_processing.py`**  
  ECG signal processing: artifact removal, heart rate and HRV extraction, and visualization.

- **`src/eda_signal_processing.py`**  
  EDA (Electrodermal Activity) processing: ADC conversion, tonic/phasic decomposition, event detection, and arousal metrics.

- **`plot.py`**  
  Real-time and static plotting utilities using Matplotlib and PyQtGraph.

- **`comfy/`**  
  Custom ComfyUI nodes for  processing tasks.

- **`Archive/`**  
  **Legacy nodes and scripts**. Older nodes have been moved here and may not work with the current codebase or ComfyUI. They are retained for reference only.

---

## Class Overview

- **NumpySignalProcessor**  
  Core static methods for filtering, normalization, peak detectio, and more.

- **ECG**  
  - Artifact removal, QRS/R-peak detection, heart rate and HRV extraction,.

- **EDA**  
  - ADC to μS conversion, tonic/phasic decomposition, event/arousal detection, metrics.

- **RR**  
  - Respiration rate extraction, deep breath detection, and visualization.

---
## Demos

Each signal processing class includes a demo script for working with recorded signals:

- **ECG Demo:**  
  Demonstrates artifact removal, QRS/R-peak detection, and heart rate/HRV extraction using a sample ECG recording.

- **EDA Demo:**  
  Showcases ADC to μS conversion, tonic/phasic decomposition, and arousal event detection with a recorded EDA signal.

- **RR Demo:**  
  Provides an example of respiration rate extraction and deep breath detection using a sample respiration signal.