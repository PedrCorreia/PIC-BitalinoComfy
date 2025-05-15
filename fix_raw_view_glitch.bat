@echo off
REM Fix RAW view glitching in PIC-2025 visualization
title PIC-2025 RAW View Glitch Fix

echo ===================================================
echo      PIC-2025 RAW View Glitch Fix v3
echo ===================================================
echo.
echo This script will fix the window glitching issue when
echo switching between view modes in the PIC-2025 visualization.
echo.
echo New in v3:
echo - Registry demo with realistic signals (ECG, EDA, RESP)
echo - Proper handling for all view modes (RAW, PROCESSED, TWIN)
echo - Fixed signal filtering for each view mode
echo - Better mode-specific optimizations
echo - Will automatically start visualization if needed
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
echo Applying RAW view glitch fix...
echo.

python "%~dp0fix_raw_view_glitch.py"

echo.
echo Script completed. The fix will remain active as long as
echo this window stays open. Close this window when done.
echo.
