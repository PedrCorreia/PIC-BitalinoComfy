@echo off
echo ====================================================
echo Running Enhanced PlotUnit Debug with Module Imports
echo ====================================================
echo.

cd "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025"

echo Setting up Python path...
set PYTHONPATH=c:\Users\corre\ComfyUI\custom_nodes\PIC-2025;c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src;%PYTHONPATH%
echo PYTHONPATH: %PYTHONPATH%
echo.

echo Checking module availability...
echo.

echo Checking for constants module...
python -c "import sys; sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025'); sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025\\src'); print('Checking if plot.constants can be imported...'); try: import plot.constants; print('SUCCESS: plot.constants module found'); except ImportError as e: print(f'ERROR: {e}')"
echo.

echo Checking for view_mode module...
python -c "import sys; sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025'); sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025\\src'); print('Checking if plot.view_mode can be imported...'); try: import plot.view_mode; print('SUCCESS: plot.view_mode module found'); except ImportError as e: print(f'ERROR: {e}')"
echo.

echo Checking for UI component modules...
python -c "import sys; sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025'); sys.path.insert(0, 'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025\\src'); print('Checking if UI component modules can be imported...'); modules = ['plot.view.base_view', 'plot.view.settings_view', 'plot.ui.sidebar', 'plot.ui.status_bar']; for m in modules: try: __import__(m); print(f'SUCCESS: {m} module found'); except ImportError as e: print(f'ERROR importing {m}: {e}')"
echo.

echo Running the enhanced debug application...
python standalone_debug_imports.py

echo.
echo Debug session completed.
pause
