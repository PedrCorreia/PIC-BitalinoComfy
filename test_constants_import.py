#!/usr/bin/env python
"""
Test script to verify constants importing works correctly
"""

import os
import sys
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try different import approaches
try:
    print("\nAttempting to import constants...")
    from src.plot.constants import *
    print("Successfully imported constants from src.plot.constants")
    print(f"WINDOW_WIDTH = {WINDOW_WIDTH}")
    print(f"STATUS_BAR_TOP = {STATUS_BAR_TOP}")
    print(f"BUTTON_COLOR_SETTINGS = {BUTTON_COLOR_SETTINGS}")
    print(f"DEFAULT_SETTINGS = {DEFAULT_SETTINGS}")
except ImportError as e:
    print(f"First import attempt failed: {e}")
    
    try:
        # Try alternative path
        print("\nTrying alternative import path...")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
        from plot.constants import *
        print("Successfully imported constants from plot.constants")
        print(f"WINDOW_WIDTH = {WINDOW_WIDTH}")
        print(f"STATUS_BAR_TOP = {STATUS_BAR_TOP}")
        print(f"BUTTON_COLOR_SETTINGS = {BUTTON_COLOR_SETTINGS}")
        print(f"DEFAULT_SETTINGS = {DEFAULT_SETTINGS}")
    except ImportError as e:
        print(f"Second import attempt failed: {e}")

print("\nScript execution complete")
