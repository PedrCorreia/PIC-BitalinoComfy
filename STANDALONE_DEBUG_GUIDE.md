# PlotUnit Standalone Debug System

This document provides a guide to using the PlotUnit standalone debug application and its various enhanced versions that properly integrate with the module structure.

## Overview

The standalone debug application allows you to test the PlotUnit visualization system independently of ComfyUI. It provides a way to:

1. Test the UI components (sidebar, status bar, settings panel)
2. Test the visualization views (raw, processed, twin)
3. Test user interactions (clicking tabs, toggling settings)

## Available Debug Scripts

Several debug scripts are available to test different aspects of the system:

### 1. Basic Debug Application

- **File**: `standalone_debug.py`
- **Usage**: Run directly or via `run_standalone_debug.bat`
- **Features**: Basic visualization with hardcoded fallbacks

### 2. Constants Import Debug Application

- **File**: `standalone_debug_imports.py`
- **Usage**: Run via `run_debug_with_imports.bat`
- **Features**: Proper constants importing with fallbacks

### 3. Enhanced Debug Application

- **File**: `standalone_debug_enhanced.py`
- **Usage**: Run via `run_enhanced_debug.bat`
- **Features**: Comprehensive import system with detailed logging

### 4. Module-Based Debug Application

- **File**: `standalone_debug_imports.py` with updated module importing
- **Usage**: Run via `run_standalone_with_modules.bat`
- **Features**: Full module integration, specialized view implementations

## Testing Import Paths

To diagnose issues with the import system, several test utilities are available:

1. `test_constants_import.py` - Tests importing constants
2. `test_viewmode_import.py` - Tests importing the ViewMode enum
3. `test_import_path.py` - Tests various import paths
4. `simple_constants_check.py` - Simple check for constants availability

## Import Path Structure

The PlotUnit components are organized in a module hierarchy:

```
ComfyUI/
  custom_nodes/
    PIC-2025/             <- Base directory
      src/                <- Source directory
        plot/             <- Plot module directory
          constants.py    <- Constants definitions
          view_mode.py    <- ViewMode enum definition
          view/           <- View implementations
            base_view.py  <- Base view class
            settings_view.py <- Settings view implementation
            ...
          ui/             <- UI components
            sidebar.py    <- Sidebar implementation
            status_bar.py <- Status bar implementation
            ...
```

## Troubleshooting

If you encounter issues with the debug applications:

1. **Import issues**: Run the import test scripts to diagnose path problems
2. **UI component issues**: Check if the UI component modules exist and can be imported
3. **ViewMode issues**: Ensure the view mode module is accessible
4. **Path issues**: Verify that the Python path includes both the base directory and src directory

## Best Practices

When modifying the debug applications:

1. Always maintain fallback implementations for robustness
2. Use detailed logging to track import success/failure
3. Test with different Python environments to ensure compatibility
4. Use conditional imports and checks rather than assuming modules exist
