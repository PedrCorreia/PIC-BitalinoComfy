# PIC-BitalinoComfy

## Introduction

Welcome to the PIC-BitalinoComfy repository. This repository is dedicated to sharing custom nodes, scripts, and other tools that I'll develop along this project.

## Repository Structure

The repository is organized as follows:

### TestNodes

This directory contains the custom nodes for signal processing. The nodes are implemented in the following files:

- **[`periodic_signal.py`](TestNodes/periodic_signal.py)**: Defines nodes for generating and manipulating periodic signals.
  - `SineNode`: Generates a sine signal tensor.
  - `CosineNode`: Generates a cosine signal tensor.
  - `PerSumNode2`: Sums two periodic signal tensors.
  - `PerSumNode3`: Sums three periodic signal tensors.
  - `PerSumNode4`: Sums four periodic signal tensors.

- **[`processing_tools.py`](TestNodes/processing_tools.py)**: Provides nodes for signal processing using FFT, low-pass filter, high-pass filter, and band-pass filter functionalities.
  - `FFTNode`: Applies Fast Fourier Transform (FFT).
  - `LowPassFilterNode`: Applies a low-pass filter.
  - `HighPassFilterNode`: Applies a high-pass filter.
  - `BandPassFilterNode`: Applies a band-pass filter.

- **[`tools.py`](TestNodes/tools.py)**: Defines a custom node for plotting signals using PyTorch tensors.
  - `PlotNode`: Plots a given tensor as a signal plot and returns the plot as an image tensor.

- **[`__init__.py`](TestNodes/__init__.py)**: Initializes the nodes and maps them to their respective categories and display names.

### Node Categories

- **Periodic Signals**
  - `Sin`: SineNode
  - `Cos`: CosineNode
  - `PeriodicSum2`: PerSumNode2
  - `PeriodicSum3`: PerSumNode3
  - `PeriodicSum4`: PerSumNode4

- **Tools**
  - `Plotter`: PlotNode
  - `FFT`: FFTNode
  - `LowPassFilter`: LowPassFilterNode
  - `HighPassFilter`: HighPassFilterNode
  - `BandPassFilter`: BandPassFilterNode
