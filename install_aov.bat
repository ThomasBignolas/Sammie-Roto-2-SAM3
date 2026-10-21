@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_aov.ps1
if errorlevel 1 (echo AOV installation failed.) else (echo AOV installation complete.)
pause
