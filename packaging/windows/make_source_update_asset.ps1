param(
    [Parameter(Mandatory = $true)]
    [string[]]$FromRef,
    [string]$ToRef = "HEAD",
    [string]$TestEnv = "sleep_scoring_dash3.0",
    [string]$CondaExe = "",
    [string]$AssetPrefix = "sleep_scoring_app_update_",
    [string]$Output = "",
    [string]$UpdaterRepo = "",
    [string[]]$InstalledBaselineManifest = @(),
    [string[]]$FromPackageZip = @(),
    [string[]]$PreserveRuntimePath = @("app_src/config.py"),
    [switch]$SkipTests,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ScriptDir "..\..")
$ArtifactDir = Join-Path $Repo "release_artifacts"
$ReleaseHelper = Join-Path $ScriptDir "lightweight_release.py"
$RequiredBuilderCommit = "eea84b5c138f7dfd72d18494c1e17f4e8de51049"

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

$CandidateArgs = @(
    "python",
    $ReleaseHelper,
    "validate-candidate",
    "--repo",
    $Repo,
    "--to-ref",
    $ToRef
)
foreach ($Ref in $FromRef) {
    $CandidateArgs += @("--from-ref", $Ref)
}
Invoke-Conda -EnvName $TestEnv -CommandArgs $CandidateArgs

$Version = Invoke-CondaCapture -EnvName $TestEnv -CommandArgs @(
    "python",
    $ReleaseHelper,
    "version",
    "--repo",
    $Repo,
    "--ref",
    $ToRef
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

if (-not $UpdaterRepo) {
    $SiblingUpdaterRepo = Join-Path (Split-Path $Repo -Parent) "desktop_app_source_updater"
    if (Test-Path -LiteralPath $SiblingUpdaterRepo) {
        $UpdaterRepo = $SiblingUpdaterRepo
    }
}

if ($InstalledBaselineManifest.Count -gt 0 -and -not $UpdaterRepo) {
    throw (
        "Compact installed baselines require the current desktop_app_source_updater " +
        "builder. Pass -UpdaterRepo or place that repo beside sleep_scoring."
    )
}

if ($UpdaterRepo) {
    $UpdaterRepo = (Resolve-Path -LiteralPath $UpdaterRepo).Path
    $BuilderPath = Join-Path $UpdaterRepo "desktop_app_source_updater\build_update_asset.py"
    if (-not (Test-Path -LiteralPath $BuilderPath)) {
        throw "Updater builder not found: $BuilderPath"
    }

    $SafeUpdaterRepo = $UpdaterRepo.Replace("\", "/")
    & git -c "safe.directory=$SafeUpdaterRepo" -C $UpdaterRepo merge-base --is-ancestor `
        $RequiredBuilderCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The maintainer-side updater builder is older than $RequiredBuilderCommit. " +
            "Update $UpdaterRepo before building this asset."
        )
    }
    $UpdaterSourceChanges = Invoke-NativeCapture -FilePath "git" -CommandArgs @(
        "-c",
        "safe.directory=$SafeUpdaterRepo",
        "-C",
        $UpdaterRepo,
        "diff",
        "--name-only",
        "HEAD",
        "--",
        "desktop_app_source_updater"
    )
    if ($UpdaterSourceChanges) {
        throw (
            "The maintainer-side updater package has uncommitted source changes:`n" +
            $UpdaterSourceChanges
        )
    }
}

if (-not $SkipTests) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    $PytestBaseTemp = Join-Path (
        Join-Path $Repo ".pytest_tmp"
    ) ("source_update_asset_" + [guid]::NewGuid().ToString("N"))
    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "pytest",
        "--basetemp",
        $PytestBaseTemp,
        "-p",
        "no:cacheprovider"
    )
}

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

$BaselineTempDir = ""
$OriginalPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
try {
    $PreparedBaselines = @()
    if ($InstalledBaselineManifest.Count -gt 0) {
        $BaselineTempDir = Join-Path (
            [System.IO.Path]::GetTempPath()
        ) "sleep-scoring-baselines-$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Path $BaselineTempDir | Out-Null

        $BaselineIndex = 0
        foreach ($Baseline in $InstalledBaselineManifest) {
            $BaselineSource = (Resolve-Path -LiteralPath $Baseline).Path
            $PreparedBaseline = Join-Path $BaselineTempDir "baseline-$BaselineIndex.json"
            $MaterializeArgs = @(
                "python",
                $ReleaseHelper,
                "materialize-baseline",
                "--source",
                $BaselineSource,
                "--output",
                $PreparedBaseline,
                "--repo",
                $Repo,
                "--to-ref",
                $ToRef
            )
            foreach ($Ref in $FromRef) {
                $MaterializeArgs += @("--from-ref", $Ref)
            }
            Invoke-Conda -EnvName $TestEnv -CommandArgs $MaterializeArgs
            $PreparedBaselines += $PreparedBaseline
            $BaselineIndex += 1
        }
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
    foreach ($Baseline in $PreparedBaselines) {
        $BuilderArgs += @("--installed-baseline-manifest", $Baseline)
    }
    foreach ($BlockedName in @(
        "app.spec",
        "environment.yml",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "run_desktop_app.py",
        "setup.cfg",
        "unblock_app.cmd"
    )) {
        $BuilderArgs += @("--blocked-path-name", $BlockedName)
    }
    foreach ($BlockedPrefix in @(
        ".worktrees/",
        "archive/",
        "build/",
        "cache/",
        "data/",
        "dist/",
        "models/",
        "packaging/windows/release_helpers/"
    )) {
        $BuilderArgs += @("--blocked-path-prefix", $BlockedPrefix)
    }

    if ($UpdaterRepo) {
        $PythonPathParts = @($UpdaterRepo)
        if ($OriginalPythonPath) {
            $PythonPathParts += $OriginalPythonPath
        }
        [Environment]::SetEnvironmentVariable(
            "PYTHONPATH",
            ($PythonPathParts -join [System.IO.Path]::PathSeparator),
            "Process"
        )
    }
    Invoke-Conda -EnvName $TestEnv -CommandArgs $BuilderArgs
} finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $OriginalPythonPath, "Process")
    if ($BaselineTempDir -and (Test-Path -LiteralPath $BaselineTempDir)) {
        Remove-Item -LiteralPath $BaselineTempDir -Recurse -Force
    }
}

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

$ValidateAssetArgs = @(
    "python",
    $ReleaseHelper,
    "validate-asset",
    "--update-zip",
    $ZipPath,
    "--checksum-file",
    "$ZipPath.sha256.txt",
    "--repo",
    $Repo,
    "--to-ref",
    $ToRef
)
foreach ($Ref in $FromRef) {
    $ValidateAssetArgs += @("--from-ref", $Ref)
}
Invoke-Conda -EnvName $TestEnv -CommandArgs $ValidateAssetArgs

Write-Host "Built source update asset: $ZipPath"
Write-Host "SHA256: $($Hash.Hash)"
