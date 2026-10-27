@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_corridorkey.ps1"
if errorlevel 1 (
    echo CorridorKey installation failed. Review the error above.
    pause
    exit /b 1
)
pause
