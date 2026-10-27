@echo off
setlocal
cd /d "%~dp0"
if exist ".runtime\school-tools\git\cmd\git.exe" set "PATH=%~dp0.runtime\school-tools\git\cmd;%PATH%"
if not exist ".venv\Scripts\python.exe" (
    echo Run install_school.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" sammie_main.py %*
if errorlevel 1 pause
