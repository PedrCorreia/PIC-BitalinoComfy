#!/usr/bin/env python
"""
Test script focusing on ViewMode constants and enum import
"""

import os
import sys
import importlib
from enum import Enum

print("=== ViewMode Import Test ===")

# Configure import paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)  # For src.plot.constants style
sys.path.insert(0, os.path.join(base_dir, "src"))  # For plot.constants style

print(f"Current directory: {base_dir}")
print(f"Python path entries: {sys.path[0:2]}")

# Try importing view_mode directly
print("\n1. Testing ViewMode enum import:")
try:
    from src.plot.view_mode import ViewMode
    print("SUCCESS: Imported ViewMode enum from src.plot.view_mode")
    print(f"ViewMode.RAW = {ViewMode.RAW}")
    print(f"ViewMode.RAW.value = {ViewMode.RAW.value}")
except ImportError as e:
    print(f"FAILED: Could not import from src.plot.view_mode: {e}")
    try:
        from plot.view_mode import ViewMode
        print("SUCCESS: Imported ViewMode enum from plot.view_mode")
        print(f"ViewMode.RAW = {ViewMode.RAW}")
        print(f"ViewMode.RAW.value = {ViewMode.RAW.value}")
    except ImportError as e:
        print(f"FAILED: Could not import from plot.view_mode: {e}")

# Try importing constants
print("\n2. Testing VIEW_MODE constants import:")
try:
    from src.plot.constants import VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS
    print("SUCCESS: Imported VIEW_MODE constants from src.plot.constants")
    print(f"VIEW_MODE_RAW = {VIEW_MODE_RAW}")
    print(f"VIEW_MODE_PROCESSED = {VIEW_MODE_PROCESSED}")
    print(f"VIEW_MODE_TWIN = {VIEW_MODE_TWIN}")
    print(f"VIEW_MODE_SETTINGS = {VIEW_MODE_SETTINGS}")
except ImportError as e:
    print(f"FAILED: Could not import from src.plot.constants: {e}")
    try:
        from plot.constants import VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS
        print("SUCCESS: Imported VIEW_MODE constants from plot.constants")
        print(f"VIEW_MODE_RAW = {VIEW_MODE_RAW}")
        print(f"VIEW_MODE_PROCESSED = {VIEW_MODE_PROCESSED}")
        print(f"VIEW_MODE_TWIN = {VIEW_MODE_TWIN}")
        print(f"VIEW_MODE_SETTINGS = {VIEW_MODE_SETTINGS}")
    except ImportError as e:
        print(f"FAILED: Could not import VIEW_MODE constants: {e}")

# Check file paths
print("\n3. Verifying file paths:")
print(f"view_mode.py exists: {os.path.isfile(os.path.join(base_dir, 'src', 'plot', 'view_mode.py'))}")
print(f"constants.py exists: {os.path.isfile(os.path.join(base_dir, 'src', 'plot', 'constants.py'))}")

print("\nTest complete")
