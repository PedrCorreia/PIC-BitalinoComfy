@echo off
REM Fix all view modes in PIC-2025 visualization with registry signals demo
title PIC-2025 View Demo with Registry Signals

echo ===================================================
echo    PIC-2025 Registry Signal Visualization Demo
echo ===================================================
echo.
echo This script demonstrates registry signals visualization
echo with proper handling for all view modes (RAW, PROCESSED, TWIN)
echo.
echo Features:
echo - Registry demo with physiological signals (ECG, EDA, RESP)
echo - Proper handling for all view modes with automatic cycling
echo - Fixed signal filtering for each view mode
echo - Optimized rendering for all container types
echo - Auto-cycles through views for demonstration purposes
echo.

REM Check if visualization is already running
set START_VIZ=n
echo Would you like to start the visualization now? (y/n)
echo If it's already running, enter 'n'.
set /p START_VIZ="Start visualization? (y/n): "

if /i "%START_VIZ%"=="y" (
    echo.
    echo Starting visualization first...
    start "" cmd /c "%~dp0run_fixed_registry_visualization_v2.bat"
    echo Waiting 5 seconds for visualization to initialize...
    timeout /t 5 > nul
)

echo.
echo Applying registry signal demo and view mode fixes...
echo.

python "%~dp0fix_raw_view_glitch.py"

echo.
echo Script completed. The demo will continue to run and cycle
echo through all view modes as long as this window stays open.
echo Close this window when done.
echo.
