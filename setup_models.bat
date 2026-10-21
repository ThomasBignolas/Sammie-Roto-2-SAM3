@echo off
cd /d "%~dp0"
if not exist ".runtime\sam3-env\Scripts\python.exe" (
    echo Run install_sam3.bat first.
    pause
    exit /b 1
)
".runtime\sam3-env\Scripts\python.exe" scripts\setup_models.py %*
pause
