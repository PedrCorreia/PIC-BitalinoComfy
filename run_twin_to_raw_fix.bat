@echo off
title PIC-2025 TWIN-to-RAW Crash Fix

echo ================================================================
echo       PIC-2025 TWIN-to-RAW View Transition Crash Fix
echo ================================================================
echo.
echo This tool specifically fixes the crash that occurs when switching
echo from TWIN view back to RAW view at 32 seconds runtime.
echo.
echo Key improvements:
echo 1. Ultra-defensive TWIN-to-RAW transition handling
echo 2. Completely bypasses registry for RAW view signals
echo 3. Uses minimal signal data to prevent memory issues
echo 4. Implements direct mode setting to avoid complex logic
echo 5. Multiple buffer renders with pause stages
echo.
echo ================================================================

REM Run the Python script
echo Starting robust view switcher with enhanced TWIN-to-RAW handling...
echo.
python "%~dp0robust_view_switcher.py"

pause
