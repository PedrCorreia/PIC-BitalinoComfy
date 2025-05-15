#!/usr/bin/env python
"""
Simple test to check constant imports
"""

import os
import sys

print("=== Constants Import Check ===")

# Add the paths to import from
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)  # For src.plot.constants style
sys.path.insert(0, os.path.join(base_dir, "src"))  # For plot.constants style

print(f"Current dir: {base_dir}")
print(f"Python version: {sys.version}")
print(f"Path: {sys.path[0:3]}")

# Try importing
print("\nTrying direct import...")
try:
    from src.plot.constants import WINDOW_WIDTH, STATUS_BAR_TOP, VIEW_MODE_RAW
    print("SUCCESS: Direct import worked!")
    print(f"WINDOW_WIDTH = {WINDOW_WIDTH}")
    print(f"STATUS_BAR_TOP = {STATUS_BAR_TOP}")
    print(f"VIEW_MODE_RAW = {VIEW_MODE_RAW}")
except ImportError as e:
    print(f"FAILED: Direct import: {e}")

print("\nTrying alternative import...")
try:
    from plot.constants import WINDOW_WIDTH, STATUS_BAR_TOP, VIEW_MODE_RAW
    print("SUCCESS: Alternative import worked!")
    print(f"WINDOW_WIDTH = {WINDOW_WIDTH}")
    print(f"STATUS_BAR_TOP = {STATUS_BAR_TOP}")
    print(f"VIEW_MODE_RAW = {VIEW_MODE_RAW}")
except ImportError as e:
    print(f"FAILED: Alternative import: {e}")
    
print("\nChecking directory structure...")
print(f"src directory exists: {os.path.isdir(os.path.join(base_dir, 'src'))}")
print(f"src/plot directory exists: {os.path.isdir(os.path.join(base_dir, 'src', 'plot'))}")
print(f"src/plot/constants.py exists: {os.path.isfile(os.path.join(base_dir, 'src', 'plot', 'constants.py'))}")

print("\nTest complete")
