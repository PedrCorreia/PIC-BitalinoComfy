@echo off
REM Run the fixed v2 version of the PIC-2025 registry visualization demo
REM This version has improved connection management and fixes GUI freezing issues

echo Starting PIC-2025 Registry Visualization (Fixed Version v2)...
echo.
echo This version fixes:
echo 1. GUI freezing when switching between raw and processed views
echo 2. Connection management issues with view changes
echo 3. Buffer implementation to properly handle last generated data
echo 4. Integration with latency parameter for better synchronization
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

python "%~dp0src\registry\plot_generator_debug_fixed_v2.py" 600 10
