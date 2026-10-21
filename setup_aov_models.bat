@echo off
cd /d "%~dp0"
if not exist .runtime\aov-env\Scripts\python.exe (
    echo Run install_aov.bat first.
    pause
    exit /b 1
)
.runtime\aov-env\Scripts\python.exe -m scripts.setup_aov
pause
