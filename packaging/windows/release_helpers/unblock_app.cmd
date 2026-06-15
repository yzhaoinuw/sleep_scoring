@echo off
setlocal

set "APP_DIR=%~dp0"
set "SLEEP_SCORING_APP_DIR=%APP_DIR%"

if not exist "%APP_DIR%run_desktop_app.exe" (
    echo Cannot find run_desktop_app.exe next to this launcher.
    echo Move this launcher back into the unzipped app folder.
    echo.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $AppRoot = $env:SLEEP_SCORING_APP_DIR; function Invoke-UnblockTarget { param([string]$Path) if (-not (Test-Path -LiteralPath $Path)) { return }; $Item = Get-Item -LiteralPath $Path -Force; if ($Item.PSIsContainer) { Get-ChildItem -LiteralPath $Path -Recurse -Force -File | ForEach-Object { Unblock-File -LiteralPath $_.FullName -ErrorAction Stop } } else { Unblock-File -LiteralPath $Item.FullName -ErrorAction Stop } }; if (Get-Command Unblock-File -ErrorAction SilentlyContinue) { Write-Host 'Preparing Sleep Scoring App files...'; try { 'run_desktop_app.exe','unblock_app.cmd','_internal','app_src','models' | ForEach-Object { Invoke-UnblockTarget -Path (Join-Path $AppRoot $_) } } catch { Write-Warning ('Some files could not be unblocked automatically: ' + $_.Exception.Message); Write-Warning 'The app will still try to start.' } }"

echo Starting Sleep Scoring App...
pushd "%APP_DIR%"
"%APP_DIR%run_desktop_app.exe" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Sleep Scoring App did not start successfully.
    echo Leave this window open and send Yue the message above.
    pause
)

exit /b %EXIT_CODE%
