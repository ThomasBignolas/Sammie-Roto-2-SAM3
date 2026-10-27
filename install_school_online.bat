@echo off
setlocal
set "SAMMIE_BOOTSTRAP=%TEMP%\sammie-school-%RANDOM%-%RANDOM%.ps1"
curl.exe --fail --location --retry 3 "https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/download/school-2026.10.27/bootstrap_school.ps1" --output "%SAMMIE_BOOTSTRAP%"
if errorlevel 1 exit /b 1
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SAMMIE_BOOTSTRAP%" %*
exit /b %ERRORLEVEL%
