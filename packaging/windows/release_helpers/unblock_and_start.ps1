param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExePath = Join-Path $AppRoot "run_desktop_app.exe"

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Cannot find run_desktop_app.exe next to this launcher. Move this launcher back into the unzipped app folder."
}

function Invoke-UnblockTarget {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $Item = Get-Item -LiteralPath $Path -Force
    if ($Item.PSIsContainer) {
        Get-ChildItem -LiteralPath $Path -Recurse -Force -File |
            ForEach-Object { Unblock-File -LiteralPath $_.FullName -ErrorAction Stop }
    } else {
        Unblock-File -LiteralPath $Item.FullName -ErrorAction Stop
    }
}

$UnblockFile = Get-Command Unblock-File -ErrorAction SilentlyContinue
if ($UnblockFile) {
    Write-Host "Preparing Sleep Scoring App files..."

    $Targets = @(
        "run_desktop_app.exe",
        "Start Sleep Scoring.cmd",
        "unblock_and_start.ps1",
        "_internal",
        "app_src",
        "models"
    )

    try {
        foreach ($Target in $Targets) {
            Invoke-UnblockTarget -Path (Join-Path $AppRoot $Target)
        }
    } catch {
        Write-Warning "Some files could not be unblocked automatically: $($_.Exception.Message)"
        Write-Warning "The app will still try to start. If it fails, run PowerShell as described in the README installation steps."
    }
}

Write-Host "Starting Sleep Scoring App..."
Push-Location $AppRoot
try {
    & $ExePath @AppArgs
    $ExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
} finally {
    Pop-Location
}

exit $ExitCode
