param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [ValidateSet("Full", "AppUpdate")]
    [string]$Kind = "Full"
)

$ErrorActionPreference = "Stop"

$ReleasePath = Resolve-Path -LiteralPath $Path

function Assert-Exists {
    param([string]$RelativePath)

    $FullPath = Join-Path $ReleasePath $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath)) {
        throw "Missing expected release item: $RelativePath"
    }
}

function Assert-Any {
    param(
        [string]$RelativePath,
        [string]$Filter
    )

    $FullPath = Join-Path $ReleasePath $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath)) {
        throw "Missing expected release directory: $RelativePath"
    }

    $Matches = Get-ChildItem -LiteralPath $FullPath -Filter $Filter -File
    if (-not $Matches) {
        throw "No files matching $Filter found under $RelativePath"
    }
}

if ($Kind -eq "Full") {
    Assert-Exists "_internal"
    Assert-Exists "run_desktop_app.exe"
    Assert-Exists "unblock_app.cmd"
    Assert-Exists "models"
    Assert-Any "models\sdreamer\checkpoints" "*.pt"
}

Assert-Exists "app_src"
Assert-Exists "app_src\__init__.py"
Assert-Exists "app_src\app.py"
Assert-Any "app_src\assets" "*.js"

Write-Host "Smoke check passed for $Kind release: $ReleasePath"
