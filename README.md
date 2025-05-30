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

-**Pygame Signal Vizualization UI**


## Usage

1. Add a "Plot Unit" node to your workflow
2. Connect your signal tensor to the "signal" input
3. Optionally connect a filtered version to "filtered_signal"
4. Run your workflow to visualize the data.

**Note:** Metrics will be displayed if available.
The window will persist between workflow runs, allowing you to monitor signals continuously.

## Interface

- Sidebar buttons:
  - R: Raw signal view
  - F: Filtered signal view
  - S: Settings view

## Example Workflows

[Add examples here]

## Extending PlotUnit

The PlotUnit architecture is designed to be extensible. Future updates might include:
- Additional visualization types
- Custom plotting options
- Data export capabilities