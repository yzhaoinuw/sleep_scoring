@echo off
setlocal

set "APP_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%APP_DIR%unblock_and_start.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Sleep Scoring App did not start successfully.
    echo Leave this window open and send Yue the message above.
    pause
)

exit /b %EXIT_CODE%
