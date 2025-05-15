@echo off
echo.
echo ===================================================
echo    PIC-2025 Registry Visualization Debug Tool
echo ===================================================
echo.
echo This tool connects the registry system to the
echo standalone debug visualization application.
echo.
echo Features:
echo  - Generates synthetic signals (ECG, EDA, sine)
echo  - Registers signals with SignalRegistry
echo  - Connects to PlotRegistry for visualization
echo  - Shows real-time signal data in PlotUnit
echo.
echo Press any key to start...
pause > nul

echo.
echo Starting registry visualization...
echo.

python -m src.registry.plot_generator_debug

echo.
echo Registry visualization completed.
echo Press any key to exit...
pause > nul
