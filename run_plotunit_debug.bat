@echo off
echo Running PlotUnit Debug...
cd /d %~dp0
python -m src.plot.debug.plot_debug
pause
