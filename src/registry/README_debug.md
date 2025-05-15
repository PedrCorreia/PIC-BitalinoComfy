# PIC-2025 Registry Debug Tools

This directory contains tools for debugging and visualizing signals in the PIC-2025 Registry system.

## Registry Visualization System

The registry debug tools allow you to:

1. Generate synthetic signals in real-time (ECG, EDA, sine waves)
2. Register them with the `SignalRegistry` and `PlotRegistry`
3. Visualize them in the standalone debug application
4. Monitor registry connections and signal flow
5. Enforce system constraints (3 raw signals + 3 processed signals maximum)

## Architecture

The tools follow the PIC-2025 two-registry architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│ Signal          │     │ Signal          │     │ Plot            │     │ Visualization   │
│ Generators      │ ──► │ Registry        │ ──► │ Registry        │ ──► │ System          │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Using the Debug Tools

### Running the Demo

#### Standard Version (No Signal Limits):
```bash
python -m src.registry.plot_generator_debug
```

Or use the included batch file:
```bash
run_registry_visualization.bat
```

#### Limited Version (Max 3 Raw + 3 Processed Signals):
```bash
python -m src.registry.plot_generator_debug_limited
```

Or use the included batch file:
```bash
run_limited_registry_visualization.bat
```

#### Fixed Version (3 Generators with Configurable Buffer):
```bash
python -m src.registry.plot_generator_debug_fixed [duration_seconds] [buffer_seconds]
```

Or use the included batch file:
```bash
run_fixed_registry_visualization.bat
```

#### Fixed Version v2 (View Mode Switching & Connection Management):
```bash
python -m src.registry.plot_generator_debug_fixed_v2 [duration_seconds] [buffer_seconds]
```

Or use the included batch file:
```bash
run_fixed_registry_visualization_v2.bat
```

### Understanding the UI

The visualization includes several registry-specific enhancements:

1. **Status Bar** - Shows the number of signals in the registry and last update time
2. **Settings View** - Displays registry connection information and signal counts
3. **Sidebar Indicators** - Shows a blinking indicator when registry signals are active

### Customizing Signal Generation

You can customize the generated signals by modifying the `RegistrySignalGenerator` class in `plot_generator_debug.py`:

- Add different signal types
- Change signal parameters
- Modify the update rate

## Components

- **plot_generator_debug.py** - Main registry visualization adapter
- **plot_generator_debug_limited.py** - Signal-limited version (3 raw, 3 processed max)
- **plot_generator_debug_fixed.py** - Fixed version with 3 generators and configurable buffer
- **plot_generator_debug_fixed_v2.py** - Fixed version v2 with improved view mode handling and connection management
- **sidebar_registry_enhancement.py** - Adds blinking registry indicator
- **settings_view_registry_enhancement.py** - Adds registry info to settings view
- **status_bar_registry_enhancement.py** - Enhances status bar with signal info

## Signal Limits and Buffer Options

### Limited Version
The `plot_generator_debug_limited.py` module enforces the system constraint of:
- Maximum 3 raw signals 
- Maximum 3 processed signals

When attempting to create signals beyond these limits:
1. Creation will fail with a warning message
2. The system will continue functioning with existing signals
3. The UI will display the current signal counts

### Fixed Version
The `plot_generator_debug_fixed.py` module provides:
- 3 different signal generators for raw signals (ECG, EDA, Sine)
- 3 different signal generators for processed signals (Wave1, Wave2, Wave3)
- Configurable buffer size (in seconds) for controlling data history
- Each signal is created only once (no duplicates)
- Improved sidebar indicators positioned lower to avoid UI crashes

### Fixed Version v2
The `plot_generator_debug_fixed_v2.py` module includes all improvements from the fixed version plus:
- Fixed GUI freezing issues when switching between raw and processed views
- Improved connection management that properly handles view mode changes
- Smart signal filtering based on current view mode
- Adaptive latency monitoring that integrates with the status bar
- Proper tracking of connected signals and smart connection/disconnection logic
- Buffer implementation that properly preserves the last generated data points

## Troubleshooting

If you encounter issues:

1. **No blinking indicator** - Make sure the registry_connected setting is being updated
2. **No signals displaying** - Check that signals are being registered in both registries
3. **UI not updating** - Ensure the signals are being passed to the PlotUnit data dictionary
4. **Signal limit reached** - If using limited version, check console for limit warnings
5. **GUI freezes when switching views** - Try using the fixed version v2 which has improved view mode handling
6. **View switching crashes** - Fixed version v2 includes proper connection management for view mode changes

## Integration with ComfyUI

In the future, this debug adapter will be adapted to integrate with ComfyUI nodes,
allowing workflow-based registry manipulation and visualization.

---

*PIC-2025 Registry System - May 2025*
