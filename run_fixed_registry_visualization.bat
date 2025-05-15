@echo off
echo ===================================================
echo    PIC-2025 Registry Visualization (Fixed Version)
echo ===================================================
echo.
echo Usage:
echo   %~nx0 [duration_seconds] [buffer_seconds]
echo.
echo   duration_seconds: How long to run (default: 600)
echo   buffer_seconds: Buffer size in seconds (default: 10)
echo.

if "%1"=="" (
    set duration=600
) else (
    set duration=%1
)

if "%2"=="" (
    set buffer=10
) else (
    set buffer=%2
)

echo Starting with %duration%s runtime and %buffer%s buffer...
echo.

python -m src.registry.plot_generator_debug_fixed %duration% %buffer%
