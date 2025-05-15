@echo off
echo ==========================================================
echo PlotUnit Debug Options
echo ==========================================================
echo.
echo This script allows you to choose which debug application to run.
echo Each option uses a different approach to import modules.
echo.
echo Options:
echo 1. Basic standalone debug (hardcoded fallbacks)
echo 2. Constants import debug (constants import with fallbacks)
echo 3. Enhanced debug (comprehensive import system)
echo 4. Module-based debug (full module integration)
echo 5. View import test (test ViewMode imports)
echo 6. Constants import test (test constants imports)
echo 7. Path import test (test import paths)
echo.

:MENU
set /p CHOICE="Enter option (1-7) or 'q' to quit: "

if "%CHOICE%"=="1" goto BASIC
if "%CHOICE%"=="2" goto CONSTANTS
if "%CHOICE%"=="3" goto ENHANCED
if "%CHOICE%"=="4" goto MODULES
if "%CHOICE%"=="5" goto VIEWTEST
if "%CHOICE%"=="6" goto CONSTTEST
if "%CHOICE%"=="7" goto PATHTEST
if /i "%CHOICE%"=="q" goto END

echo Invalid choice. Please try again.
goto MENU

:BASIC
echo Running basic standalone debug...
call run_standalone_debug.bat
goto END

:CONSTANTS
echo Running constants import debug...
call run_debug_with_imports.bat
goto END

:ENHANCED
echo Running enhanced debug...
call run_enhanced_debug.bat
goto END

:MODULES
echo Running module-based debug...
call run_standalone_with_modules.bat
goto END

:VIEWTEST
echo Running ViewMode import test...
call run_viewmode_test.bat
goto END

:CONSTTEST
echo Running constants import test...
call run_debug_with_constants.bat
goto END

:PATHTEST
echo Running import path test...
call run_import_path_test.bat
goto END

:END
echo.
echo Debug session completed.
pause
