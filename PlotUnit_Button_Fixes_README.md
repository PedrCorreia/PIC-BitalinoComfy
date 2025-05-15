# PlotUnit Button Fixes

This update addresses the unresponsive settings buttons in the PlotUnit visualization system and improves the overall UI appearance.

## Key Improvements

1. **Fixed Settings Buttons**
   - Toggles for FPS Cap, Performance Mode, and Light Mode now work correctly
   - Action buttons (Reset Plots, Reset Registry) respond properly to clicks
   - Improved visual appearance of toggles and buttons

2. **UI Enhancements**
   - Moved status bar to the top of the window for better visibility
   - Improved sidebar tab styling with larger icons (R, P, T, S)
   - Enhanced button colors for better visual feedback
   - Better debug output for troubleshooting

3. **Code Quality**
   - Better state handling in event handler
   - More robust button click detection
   - Organized constants for easier customization

## Implementation

The following files have been modified:

- `constants.py`: Added status bar position and button styling constants
- `ui/status_bar.py`: Updated to support top position
- `ui/sidebar.py`: Modified to account for status bar at top
- `view/settings_view.py`: Improved button and toggle styling
- `event_handler.py`: Enhanced event handling for settings

## Installation

Run the `apply_button_fixes.bat` script to automatically apply the changes. This script will:

1. Back up your original files with a `.bak` extension
2. Apply all the necessary changes
3. Provide a summary of the changes

## Testing

1. Launch ComfyUI with the PlotUnit extension enabled
2. Click on the "S" tab in the sidebar to access the Settings view
3. Try toggling the switches for FPS Cap, Performance Mode, and Light Mode
4. Try clicking the action buttons (Reset Plots, Reset Registry)
5. Verify that the UI responds correctly to your clicks

## Reverting Changes

If you need to revert the changes, simply rename the `.bak` files back to their original names.

## Acknowledgments

These improvements build upon the lessons learned from the standalone debug application that was created to isolate and fix the button functionality issues.
