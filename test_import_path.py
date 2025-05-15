#!/usr/bin/env python
"""
Test script to validate constant imports and path structure
"""

import os
import sys
import importlib

# Print environment information
print("=== Import Path Test ===")
print(f"Current working directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"Absolute path: {os.path.abspath(__file__)}")
print(f"Directory: {os.path.dirname(os.path.abspath(__file__))}")

# Add project directories to path
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\nAdding {base_dir} to sys.path")
sys.path.insert(0, base_dir)

# Try to import constants directly
print("\n=== Attempting imports ===")

def try_import(module_path):
    """Attempt to import a module and report result"""
    try:
        module = importlib.import_module(module_path)
        print(f"SUCCESS: Imported {module_path}")
        return module
    except ImportError as e:
        print(f"FAILED: Could not import {module_path}")
        print(f"  Error: {e}")
        return None

# Try various import paths
constants_module = try_import("src.plot.constants")
if constants_module:
    print("\n=== Constants from src.plot.constants ===")
    print(f"WINDOW_WIDTH = {constants_module.WINDOW_WIDTH}")
    print(f"STATUS_BAR_TOP = {constants_module.STATUS_BAR_TOP}")
    print(f"VIEW_MODE_RAW = {constants_module.VIEW_MODE_RAW}")
    print(f"DEFAULT_SETTINGS = {constants_module.DEFAULT_SETTINGS}")

# Try alternative import
if not constants_module:
    print("\nTrying alternative path...")
    alt_path = os.path.join(base_dir, 'src')
    print(f"Adding {alt_path} to sys.path")
    sys.path.insert(0, alt_path)
    constants_module = try_import("plot.constants")
    if constants_module:
        print("\n=== Constants from plot.constants ===")
        print(f"WINDOW_WIDTH = {constants_module.WINDOW_WIDTH}")
        print(f"STATUS_BAR_TOP = {constants_module.STATUS_BAR_TOP}")
        print(f"VIEW_MODE_RAW = {constants_module.VIEW_MODE_RAW}")
        print(f"DEFAULT_SETTINGS = {constants_module.DEFAULT_SETTINGS}")

print("\n=== Test Complete ===")
