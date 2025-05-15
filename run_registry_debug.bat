@echo off
echo Running PIC-2025 Registry Visualization Debug
echo Connecting your registry system to the standalone debug visualization

python -m src.registry.plot_generator_debug

echo Debug process finished
pause
