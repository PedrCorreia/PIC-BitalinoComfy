# PIC-2025 Registry System

This folder contains the modular registry system for the PIC-2025 visualization and signal processing architecture. It is designed for robust, real-time, and extensible signal management, supporting both synthetic and real signals for UI visualization and processing.

## Key Modules

- **signal_registry.py**: Core singleton registry for all signals. Used by generators and the plot registry.
- **plot_registry.py**: Main registry for signals and their metadata, providing robust, thread-safe access for visualization and adapters.
- **signal_generator.py**: Contains only signal generation and registration logic. Handles background thread generation, buffer management, and robust updates to the registry.
- **plot_registry_adapter.py**: Adapter for connecting the registry to the UI/PlotUnit. Handles view mode, node registration, and robust signal access for visualization. No signal generation logic.

## Design Principles

- **Separation of Concerns**: Signal generation, registry management, and UI/adapter logic are strictly separated for maintainability and extensibility.
- **Thread Safety**: All registry operations are thread-safe for real-time updates.
- **Robustness**: Signal fetching, registration, and plotting are resilient to malformed or missing data.
- **Extensibility**: New signal types, adapters, or UI integrations can be added without modifying core logic.

## Usage

- Use `RegistrySignalGenerator` to create and update synthetic signals in the background.
- Use `PlotRegistry` to access signals and metadata for visualization.
- Use `PlotUnitRegistryAdapter` to connect the registry to a UI or visualization component, handling node registration and view mode filtering.

## Migration Notes

- Legacy files (e.g., `plot_generator_debug_fixed.py`, `plot_generator_debug_fixed_v2.py`) are deprecated. All new code should use the modular files above.
- If you need integration with legacy or custom UIs, use the adapter as the only interface to the registry.

## See Also
- [../docs/plot_unit_integration_guide.md](../docs/plot_unit_integration_guide.md)
- [../docs/signal_architecture.md](../docs/signal_architecture.md)
- [../docs/registry_troubleshooting.md](../docs/registry_troubleshooting.md)

---

_Last updated: May 2025_
