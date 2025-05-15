# PlotUnit Debug and Testing Suite

This folder contains various debug scripts and tools for testing the PlotUnit visualization system in ComfyUI.

## Files Overview

### Main Debug Applications
- `standalone_debug.py` - Original standalone debug application
- `standalone_debug_enhanced.py` - Enhanced version with robust import handling

### Import Test Utilities
- `test_constants_import.py` - Tests importing constants from the project
- `test_viewmode_import.py` - Tests importing ViewMode enum and view mode constants
- `test_import_path.py` - Tests various import paths available in the project
- `simple_constants_check.py` - Simple check for constants availability

### Batch Files
- `run_debug_with_constants.bat` - Runs the original debug script
- `run_enhanced_debug.bat` - Runs the enhanced debug application
- `run_viewmode_test.bat` - Runs the ViewMode import test
- `run_import_path_test.bat` - Runs the import path test utility

## Import System

The PlotUnit components are structured in a module hierarchy:

```
ComfyUI/
  custom_nodes/
    PIC-2025/              <- Base directory
      src/                 <- Source directory
        plot/              <- Plot module directory
          constants.py     <- Constants definitions
          view_mode.py     <- ViewMode enum definition
```

The enhanced debug application attempts to import from this structure in two ways:
1. First attempt: `from src.plot.constants import *`
2. If that fails: `from plot.constants import *`
3. If both fail: Fall back to default constant definitions

## Usage

To test the import system and ensure proper functionality:

1. Run `run_import_path_test.bat` to check if imports are working correctly
2. Run `run_viewmode_test.bat` to verify ViewMode enum imports
3. Run `run_enhanced_debug.bat` to launch the enhanced standalone debug app

## Troubleshooting

If import issues persist:

1. Verify that the project structure matches the expected hierarchy
2. Check if constants.py and view_mode.py exist in the correct locations
3. Look for any syntax errors in the Python files
4. Check the system path setup in the debug scripts

## Development Notes

The enhanced debug application features:
- Robust multi-stage import system with fallbacks
- Detailed logging of import successes and failures
- Support for both direct and qualified imports
- Configuration to use TAB_ICON_FONT_SIZE when available
- Graceful fallbacks for all required constants
