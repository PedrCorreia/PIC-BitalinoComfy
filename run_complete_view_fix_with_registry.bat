@echo off
title PIC-2025 Complete View Mode Fix with Registry Integration

echo ================================================================
echo    PIC-2025 View Mode Fix with Registry Signal Integration
echo ================================================================
echo.
echo This tool provides a complete solution for PIC-2025 visualization:
echo.
echo 1. Fixes the crash when switching from TWIN view to RAW view
echo 2. Properly integrates with registry to show real signals
echo 3. Shows correct signals in each view mode:
echo    - RAW view: Shows all raw signals
echo    - PROCESSED view: Shows all processed signals 
echo    - TWIN view: Shows matching raw/processed signal pairs
echo 4. Activates registry indicator light in the sidebar
echo 5. Shows all signals in the status bar and settings
echo.
echo ================================================================

REM Run the Python script
echo Starting robust view switcher with full registry integration...
echo.
python "%~dp0robust_view_switcher.py"

pause
