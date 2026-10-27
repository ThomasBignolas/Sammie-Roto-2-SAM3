@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update_school.ps1" -Directory "%~dp0." %*
exit /b %ERRORLEVEL%
