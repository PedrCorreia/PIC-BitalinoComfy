@echo off
echo ===================================================
echo    PIC-2025 Registry Visualization Debug Tool
echo    (Limited to 3 raw + 3 processed signals)
echo ===================================================

REM Check if a duration argument was provided
IF "%~1"=="" (
  echo Running with default duration (600 seconds)
  python -m src.registry.plot_generator_debug_limited
) ELSE (
  echo Running with custom duration: %1 seconds
  python -m src.registry.plot_generator_debug_limited %1
)
