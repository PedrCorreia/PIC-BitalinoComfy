@echo off
echo Implementing fixes for settings buttons in the PlotUnit visualization system...

REM Backup original files
echo Creating backups...
copy "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\constants.py" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\constants.py.bak"
copy "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\ui\status_bar.py" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\ui\status_bar.py.bak"
copy "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\ui\sidebar.py" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\ui\sidebar.py.bak"
copy "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\view\settings_view.py" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\view\settings_view.py.bak"
copy "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\event_handler.py" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\event_handler.py.bak"

REM Copy new implementations
echo Implementing new versions...
move "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\view\settings_view.py.new" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\view\settings_view.py"
move "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\event_handler.py.new" "c:\Users\corre\ComfyUI\custom_nodes\PIC-2025\src\plot\event_handler.py"

echo.
echo Changes implemented successfully!
echo The following files have been updated:
echo - constants.py (STATUS_BAR_TOP, BUTTON_COLOR_SETTINGS)
echo - status_bar.py (top position)
echo - sidebar.py (handling for top status bar)
echo - settings_view.py (improved button and toggle styling)
echo - event_handler.py (improved event handling)
echo.
echo Backups have been created with .bak extension.
echo.
echo Press any key to exit...
pause > nul
