@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_school.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo Installation stopped. Read the error above, then run install_school.bat again.
    echo Completed downloads will be reused. Your projects are not changed.
    if /I not "%SAMMIE_UNATTENDED%"=="1" pause
)
exit /b %RESULT%
