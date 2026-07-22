param(
    [Parameter(Mandatory = $true)]
    [string[]]$FromRef,
    [string]$ToRef = "HEAD",
    [string]$TestEnv = "sleep_scoring_dash3.0",
    [string]$CondaExe = "",
    [string]$AssetPrefix = "sleep_scoring_app_update_",
    [string]$Output = "",
    [string[]]$FromPackageZip = @(),
    [string[]]$PreserveRuntimePath = @("app_src/config.py"),
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

    $OutputText = & $FilePath @CommandArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($CommandArgs -join ' ')`n$($OutputText | Out-String)"
    }
    return ($OutputText | Out-String).Trim()
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

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

if ($Output) {
    $ZipPath = [System.IO.Path]::GetFullPath($Output)
} else {
    $ZipPath = Join-Path $ArtifactDir "$AssetPrefix$Version.zip"
}

Write-Host "Building source update asset: $ZipPath"
Write-Host "Compatible previous refs: $($FromRef -join ', ')"
Write-Host "Target ref: $ToRef"
Write-Host "Test environment: $TestEnv"

if (-not $SkipTests) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "pytest",
        "--basetemp",
        ".pytest_tmp\source_update_asset",
        "-p",
        "no:cacheprovider"
    )
}

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

$BuilderArgs = @(
    "python",
    "-m",
    "desktop_app_source_updater.build_update_asset",
    "--repo",
    $Repo,
    "--app-name",
    "sleep_scoring",
    "--runtime-path",
    "app_src",
    "--to-ref",
    $ToRef,
    "--version-file",
    "app_src/__init__.py",
    "--asset-prefix",
    $AssetPrefix,
    "--output",
    $ZipPath
)

foreach ($Ref in $FromRef) {
    $BuilderArgs += @("--from-ref", $Ref)
}

Invoke-Conda -EnvName $TestEnv -CommandArgs $BuilderArgs

if ($FromPackageZip.Count -gt 0 -or $PreserveRuntimePath.Count -gt 0) {
    $AlignArgs = @(
        "python",
        "packaging\windows\align_update_asset_with_package.py",
        "--update-zip",
        $ZipPath
    )
    foreach ($PackageSpec in $FromPackageZip) {
        $AlignArgs += @("--from-package-zip", $PackageSpec)
    }
    foreach ($RuntimePath in $PreserveRuntimePath) {
        $AlignArgs += @("--preserve-path", $RuntimePath)
    }
    Invoke-Conda -EnvName $TestEnv -CommandArgs $AlignArgs
}

$Hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
"$($Hash.Hash)  $(Split-Path $ZipPath -Leaf)" |
    Set-Content -LiteralPath "$ZipPath.sha256.txt" -Encoding UTF8

Write-Host "Built source update asset: $ZipPath"
Write-Host "SHA256: $($Hash.Hash)"
