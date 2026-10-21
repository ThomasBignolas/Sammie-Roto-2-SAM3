@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_sam3.ps1" %*
if errorlevel 1 (
    echo Installation failed. See the error above.
    pause
    exit /b 1
)
pause
