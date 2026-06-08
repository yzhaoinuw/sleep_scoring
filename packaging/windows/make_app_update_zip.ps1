param(
    [string]$TestEnv = "sleep_scoring_dash3.0",
    [string]$CondaExe = "",
    [switch]$SkipTests,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ScriptDir "..\..")
$ArtifactDir = Join-Path $Repo "release_artifacts"

Set-Location $Repo

if (-not $CondaExe) {
    $DefaultConda = Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"
    if (Test-Path -LiteralPath $DefaultConda) {
        $CondaExe = $DefaultConda
    } else {
        $CondaExe = "conda"
    }
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$CommandArgs
    )

    & $FilePath @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($CommandArgs -join ' ')"
    }
}

function Invoke-NativeCapture {
    param(
        [string]$FilePath,
        [string[]]$CommandArgs
    )

    $Output = & $FilePath @CommandArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($CommandArgs -join ' ')`n$($Output | Out-String)"
    }
    return ($Output | Out-String).Trim()
}

function Invoke-Conda {
    param(
        [string]$EnvName,
        [string[]]$CommandArgs
    )

    Invoke-Native -FilePath $CondaExe -CommandArgs (@("run", "-n", $EnvName) + $CommandArgs)
}

function Invoke-CondaCapture {
    param(
        [string]$EnvName,
        [string[]]$CommandArgs
    )

    return Invoke-NativeCapture -FilePath $CondaExe -CommandArgs (@("run", "-n", $EnvName) + $CommandArgs)
}

if (-not $AllowDirty) {
    $Status = Invoke-NativeCapture -FilePath "git" -CommandArgs @("status", "--short")
    if ($Status) {
        throw "Worktree is not clean. Commit, stash, or rerun with -AllowDirty for local test patches."
    }
}

$Version = Invoke-CondaCapture -EnvName $TestEnv -CommandArgs @(
    "python",
    "-c",
    "from app_src import VERSION; print(VERSION)"
)

$Version = $Version.Trim()
$UpdateName = "sleep_scoring_app_$Version-app_src_update"
$StagePath = Join-Path $Repo "build\$UpdateName"
$ZipPath = Join-Path $ArtifactDir "$UpdateName.zip"

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

Write-Host "Building $UpdateName from $Repo"
Write-Host "Test environment: $TestEnv"

if (-not $SkipTests) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "pytest",
        "--basetemp",
        ".pytest_tmp\app_src_update",
        "-p",
        "no:cacheprovider"
    )
}

if (Test-Path -LiteralPath $StagePath) {
    Remove-Item -LiteralPath $StagePath -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $StagePath | Out-Null
Copy-Item -LiteralPath (Join-Path $Repo "app_src") -Destination (Join-Path $StagePath "app_src") -Recurse -Force

Get-ChildItem -LiteralPath $StagePath -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force

$VideoDir = Join-Path $StagePath "app_src\assets\videos"
if (Test-Path -LiteralPath $VideoDir) {
    Remove-Item -LiteralPath $VideoDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $VideoDir | Out-Null

$Manifest = [ordered]@{
    version = $Version
    kind = "app-src-update"
    branch = Invoke-NativeCapture -FilePath "git" -CommandArgs @("branch", "--show-current")
    git_commit = Invoke-NativeCapture -FilePath "git" -CommandArgs @("rev-parse", "HEAD")
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    test_env = if ($SkipTests) { $null } else { $TestEnv }
}

$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $StagePath "manifest.json") -Encoding UTF8

@"
Sleep Scoring App $Version app_src update

Use this update only with an existing PyInstaller app folder.
Replace the existing app_src folder with the app_src folder in this zip.
Use a full app package instead if dependencies, _internal, models, or run_desktop_app.exe changed.
"@ | Set-Content -LiteralPath (Join-Path $StagePath "UPDATE_NOTES.txt") -Encoding UTF8

& (Join-Path $ScriptDir "smoke_check_release.ps1") -Path $StagePath -Kind AppUpdate

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Path $StagePath -DestinationPath $ZipPath

$Hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
"$($Hash.Hash)  $(Split-Path $ZipPath -Leaf)" |
    Set-Content -LiteralPath "$ZipPath.sha256.txt" -Encoding UTF8

$Manifest["artifact"] = Split-Path $ZipPath -Leaf
$Manifest["sha256"] = $Hash.Hash
$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath "$ZipPath.manifest.json" -Encoding UTF8

Write-Host "Built app update zip: $ZipPath"
Write-Host "SHA256: $($Hash.Hash)"
