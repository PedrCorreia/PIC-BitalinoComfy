@echo off
echo Testing PlotUnit Settings Buttons
echo ==================================

cd /d "%~dp0"
python -c "import os; import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pygame; from src.plot.plot_unit import PlotUnit; from src.plot.view_mode import ViewMode; plot = PlotUnit.get_instance(); plot.start(); import time; time.sleep(2); plot._set_mode(ViewMode.SETTINGS); print('Switched to settings view. Click on buttons to test.'); print('Press Ctrl+C to exit.')"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error running test. Check Python environment and dependencies.
    echo.
    pause
) else (
    echo.
    echo Test completed successfully!
    echo.
)
