@echo off
setlocal
set "SAMMIE_UPDATER=%TEMP%\sammie-update-%RANDOM%-%RANDOM%.ps1"
curl.exe --fail --location --retry 3 "https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download/update_school.ps1" --output "%SAMMIE_UPDATER%"
if errorlevel 1 exit /b 1
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SAMMIE_UPDATER%" %*
exit /b %ERRORLEVEL%
