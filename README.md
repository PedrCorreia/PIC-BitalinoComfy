# PIC-BitalinoComfy

## Introduction

Welcome to the PIC-BitalinoComfy repository. This repository is dedicated to sharing custom nodes, scripts, and other tools that I'll develop along this project.

## Repository Structure

The repository is organized as follows:

### TestNodes

This directory contains the custom nodes for signal processing. The nodes are implemented in the following files:

- **[`periodic_signal.py`](TestNodes/periodic_signal.py)**: Defines nodes for generating and manipulating periodic signals.
  - `NoiseNode`: Generates a modulated noise signal across the frequency spectrum.
  - `PeriodicNode`: Allows selection of sine or cosine wave to generate a modulated periodic signal.


- **[`processing_tools.py`](TestNodes/processing_tools.py)**: Provides nodes for signal processing using FFT, low-pass filter, high-pass filter, and band-pass filter functionalities.
  - `FFTNode`: Applies Fast Fourier Transform (FFT).
  - `FilterNode`: Allows selection of low-pass, high-pass, or band-pass filter to apply to a given tensor.

- **[`tools.py`](TestNodes/tools.py)**: Defines a custom node for plotting signals using PyTorch tensors.
  - `PlotNode`: Plots a given tensor as a signal plot and returns the plot as an image tensor.
  - `PerSumNode2`: Sums two signal tensors.
  - `PerSumNode3`: Sums three signal tensors.
  - `PerSumNode4`: Sums four signal tensors.
  - `SavePlot`: Plots the input tensor and saves the resulting image to your ComfyUI output directory.
  - `PreviewPlot`: Plots the input tensor and previews the resulting plot.

- **[`__init__.py`](TestNodes/__init__.py)**: Initializes the nodes and maps them to their respective categories and display names.

### Node Categories

- **Periodic Signals**
  - `Noise`: NoiseNode
  - `Periodic`: PeriodicNode

- **Processing Tools**
  - `FFT`: FFTNode
  - `Filter`: FilterNode
- **Tools**
  - `SavePlot`: SavePlot
  - `PreviewPlot`: PreviewPlot
  - `PeriodicSum2`: PerSumNode2
  - `PeriodicSum3`: PerSumNode3
  - `PeriodicSum4`: PerSumNode4

### Obsolete Nodes

- **Periodic Signals**
  - `Sin`: SineNode
  - `Cos`: CosineNode

- **Tools**
  - `Plotter`: PlotNode

- **Basic Signal Processing**
  - `LowPassFilter`: LowPassFilterNode
  - `HighPassFilter`: HighPassFilterNode
  - `BandPassFilter`: BandPassFilterNode
