@echo off
title PIC-2025 Robust View Switcher

echo ================================================================
echo       PIC-2025 Robust View Switcher - Fix for View Crashes
echo ================================================================
echo.
echo This tool fixes the crash that occurs when switching from TWIN view 
echo back to RAW view at 32 seconds runtime.
echo.
echo Key features:
echo 1. Properly handles transitioning between all view modes
echo 2. Specially fixes the critical TWIN-to-RAW transition
echo 3. Uses minimal signals for improved stability
echo 4. Demonstrates all views without crashing
echo.
echo ================================================================

REM Run the Python script
echo Starting robust view switcher...
echo.
python "%~dp0robust_view_switcher.py"

pause
