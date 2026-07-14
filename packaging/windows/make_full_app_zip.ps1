param(
    [string]$BuildEnv = "sleep_scoring_dash3.0_dist",
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
$env:SLEEP_SCORING_REPO_ROOT = $Repo

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
        throw "Worktree is not clean. Commit, stash, or rerun with -AllowDirty for local test builds."
    }
}

$Version = Invoke-CondaCapture -EnvName $BuildEnv -CommandArgs @(
    "python",
    "-c",
    "from app_src import VERSION; print(VERSION)"
)

$Version = $Version.Trim()
$DistName = "sleep_scoring_app_$Version"
$DistPath = Join-Path $Repo "dist\$DistName"
$ZipPath = Join-Path $ArtifactDir "$DistName-windows.zip"
$TorchVersion = Invoke-CondaCapture -EnvName $BuildEnv -CommandArgs @(
    "python",
    "-c",
    "import torch; print(torch.__version__.replace('+', '-'))"
)
$TorchVersion = $TorchVersion.Trim()
$TorchRuntimeName = "torch"
$TorchRuntimeStage = Join-Path $Repo "build\$TorchRuntimeName"
$TorchRuntimeZipPath = Join-Path $ArtifactDir "$TorchRuntimeName.zip"

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

Write-Host "Building $DistName from $Repo"
Write-Host "Build environment: $BuildEnv"
Write-Host "Test environment:  $TestEnv"

Invoke-Conda -EnvName $BuildEnv -CommandArgs @("python", "-m", "pip", "check")

if (-not $SkipTests) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "pytest",
        "--basetemp",
        ".pytest_tmp\build",
        "-p",
        "no:cacheprovider"
    )
}

if (Test-Path -LiteralPath $DistPath) {
    Remove-Item -LiteralPath $DistPath -Recurse -Force
}

Invoke-Conda -EnvName $BuildEnv -CommandArgs @(
    "python",
    "-m",
    "PyInstaller",
    "--clean",
    "--noconfirm",
    "packaging\windows\app.spec"
)

$BundledTorchDir = Join-Path $DistPath "_internal\torch"
if (Test-Path -LiteralPath $BundledTorchDir) {
    if (Test-Path -LiteralPath $TorchRuntimeStage) {
        Remove-Item -LiteralPath $TorchRuntimeStage -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $TorchRuntimeStage | Out-Null
    Copy-Item -LiteralPath $BundledTorchDir -Destination (Join-Path $TorchRuntimeStage "torch") -Recurse -Force

    Get-ChildItem -LiteralPath $TorchRuntimeStage -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force

    if (Test-Path -LiteralPath $TorchRuntimeZipPath) {
        Remove-Item -LiteralPath $TorchRuntimeZipPath -Force
    }

    Compress-Archive -Path (Join-Path $TorchRuntimeStage "*") -DestinationPath $TorchRuntimeZipPath -Force

    $TorchRuntimeHash = Get-FileHash -LiteralPath $TorchRuntimeZipPath -Algorithm SHA256
    "$($TorchRuntimeHash.Hash)  $(Split-Path $TorchRuntimeZipPath -Leaf)" |
        Set-Content -LiteralPath "$TorchRuntimeZipPath.sha256.txt" -Encoding UTF8

    $TorchRuntimeManifest = [ordered]@{
        kind = "sdreamer-torch-runtime-windows"
        artifact = Split-Path $TorchRuntimeZipPath -Leaf
        source_full_app = $DistName
        torch_version = $TorchVersion
        generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        install_target = "Copy all zip contents directly into the app _internal folder."
        contains_internal_folder = $false
        required_paths = @("torch/")
        sha256 = $TorchRuntimeHash.Hash
    }

    $TorchRuntimeManifest | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath "$TorchRuntimeZipPath.manifest.json" -Encoding UTF8

    Remove-Item -LiteralPath $BundledTorchDir -Recurse -Force
    Remove-Item -LiteralPath $TorchRuntimeStage -Recurse -Force
} else {
    Write-Warning "PyInstaller did not produce _internal\torch; no optional Torch runtime zip was created."
}

$RuntimePath = Join-Path $DistPath "app_src"
if (Test-Path -LiteralPath $RuntimePath) {
    Remove-Item -LiteralPath $RuntimePath -Recurse -Force
}
Invoke-Conda -EnvName $BuildEnv -CommandArgs @(
    "python",
    "packaging\windows\export_runtime_from_git.py",
    "--repo",
    $Repo,
    "--ref",
    "HEAD",
    "--runtime-path",
    "app_src",
    "--destination",
    $DistPath
)
Copy-Item -LiteralPath (Join-Path $Repo "models") -Destination (Join-Path $DistPath "models") -Recurse -Force

$ReleaseHelperDir = Join-Path $ScriptDir "release_helpers"
foreach ($HelperFile in @("unblock_app.cmd")) {
    Copy-Item -LiteralPath (Join-Path $ReleaseHelperDir $HelperFile) -Destination $DistPath -Force
}

Get-ChildItem -LiteralPath $DistPath -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force

$VideoDir = Join-Path $DistPath "app_src\assets\videos"
if (Test-Path -LiteralPath $VideoDir) {
    Remove-Item -LiteralPath $VideoDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $VideoDir | Out-Null

& (Join-Path $ScriptDir "smoke_check_release.ps1") -Path $DistPath -Kind Full
Invoke-Native -FilePath (Join-Path $DistPath "run_desktop_app.exe") -CommandArgs @("--smoke")
Invoke-Native -FilePath (Join-Path $DistPath "run_desktop_app.exe") -CommandArgs @("--check-update")

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Path $DistPath -DestinationPath $ZipPath

$Hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
"$($Hash.Hash)  $(Split-Path $ZipPath -Leaf)" |
    Set-Content -LiteralPath "$ZipPath.sha256.txt" -Encoding UTF8

$Freeze = Invoke-CondaCapture -EnvName $BuildEnv -CommandArgs @("python", "-m", "pip", "freeze")
$Freeze | Set-Content -LiteralPath "$ZipPath.build_env_requirements.txt" -Encoding UTF8

$Manifest = [ordered]@{
    version = $Version
    kind = "full-windows"
    branch = Invoke-NativeCapture -FilePath "git" -CommandArgs @("branch", "--show-current")
    git_commit = Invoke-NativeCapture -FilePath "git" -CommandArgs @("rev-parse", "HEAD")
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    build_env = $BuildEnv
    test_env = if ($SkipTests) { $null } else { $TestEnv }
    python = Invoke-CondaCapture -EnvName $BuildEnv -CommandArgs @("python", "--version")
    pyinstaller = Invoke-CondaCapture -EnvName $BuildEnv -CommandArgs @("python", "-m", "PyInstaller", "--version")
    artifact = Split-Path $ZipPath -Leaf
    launcher = "unblock_app.cmd"
    build_env_requirements = Split-Path "$ZipPath.build_env_requirements.txt" -Leaf
    optional_torch_runtime = if (Test-Path -LiteralPath $TorchRuntimeZipPath) { Split-Path $TorchRuntimeZipPath -Leaf } else { $null }
    sha256 = $Hash.Hash
}

$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath "$ZipPath.manifest.json" -Encoding UTF8

Write-Host "Built full app zip: $ZipPath"
Write-Host "SHA256: $($Hash.Hash)"
if (Test-Path -LiteralPath $TorchRuntimeZipPath) {
    Write-Host "Built optional Torch runtime zip: $TorchRuntimeZipPath"
    Write-Host "Torch runtime SHA256: $($TorchRuntimeHash.Hash)"
}
