param(
    [string]$BuildEnv = "sleep_scoring_dash3.0_dist",
    [string]$TestEnv = "sleep_scoring_dash3.0",
    [string]$CondaExe = "",
    [string]$NpmExe = "",
    [switch]$ValidateOnly,
    [switch]$SkipQualityChecks,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$FullReleaseHelper = Join-Path $ScriptDir "full_release.py"
$PackageNamingHelper = Join-Path $ScriptDir "package_naming.py"
$BuildScript = Join-Path $ScriptDir "make_full_app_zip.ps1"

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

    return Invoke-NativeCapture -FilePath $CondaExe -CommandArgs (
        @("run", "-n", $EnvName) + $CommandArgs
    )
}

function Resolve-NpmExecutable {
    param([string]$ConfiguredPath)

    if ($ConfiguredPath) {
        return $ConfiguredPath
    }

    $NpmCommand = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
    if ($NpmCommand) {
        return $NpmCommand.Source
    }

    $CodexNodeRoot = Join-Path $env:LOCALAPPDATA "OpenAI\Codex\runtimes\cua_node"
    if (Test-Path -LiteralPath $CodexNodeRoot -PathType Container) {
        $BundledNpm = Get-ChildItem -LiteralPath $CodexNodeRoot -Recurse -Filter "npm.cmd" |
            Where-Object { $_.FullName -notmatch "\\node_modules\\" } |
            Select-Object -First 1
        if ($BundledNpm) {
            $env:PATH = "$($BundledNpm.Directory.FullName);$env:PATH"
            return $BundledNpm.FullName
        }
    }

    throw "npm.cmd was not found. Install Node.js or pass -NpmExe with its full path."
}

if (-not $AllowDirty) {
    $Status = Invoke-NativeCapture -FilePath "git" -CommandArgs @("status", "--short")
    if ($Status) {
        throw "Worktree is not clean. Commit the full-release candidate before running this command."
    }
}

$Candidate = Invoke-CondaCapture -EnvName $TestEnv -CommandArgs @(
    "python",
    $FullReleaseHelper,
    "--repo",
    $Repo
)
$CandidateParts = $Candidate -split "`t"
if ($CandidateParts.Count -ne 2) {
    throw "Unexpected full-release validation output: $Candidate"
}
$Version = $CandidateParts[0]
$UpdaterCommit = $CandidateParts[1]
$ReleaseLine = Invoke-CondaCapture -EnvName $TestEnv -CommandArgs @(
    "python",
    $PackageNamingHelper,
    $Version
)
$ExpectedZip = Join-Path $Repo "release_artifacts\sleep_scoring_app_$ReleaseLine-windows.zip"

Write-Host "Full-release candidate: $Version"
Write-Host "Full-package release line: $ReleaseLine"
Write-Host "Pinned updater commit: $UpdaterCommit"

if (-not $SkipQualityChecks) {
    $ResolvedNpmExe = Resolve-NpmExecutable -ConfiguredPath $NpmExe
    Invoke-Native -FilePath $ResolvedNpmExe -CommandArgs @("--version")

    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    $PytestBaseTemp = Join-Path (
        Join-Path $Repo ".pytest_tmp"
    ) ("full_release_" + [guid]::NewGuid().ToString("N"))
    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "python",
        "-m",
        "pytest",
        "--basetemp",
        $PytestBaseTemp,
        "-p",
        "no:cacheprovider",
        "-q"
    )

    $OriginalPythonUtf8 = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
    try {
        [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
        Invoke-Conda -EnvName $TestEnv -CommandArgs @(
            "pre-commit",
            "run",
            "black",
            "--all-files"
        )
    } finally {
        [Environment]::SetEnvironmentVariable(
            "PYTHONUTF8",
            $OriginalPythonUtf8,
            "Process"
        )
    }

    Invoke-Conda -EnvName $TestEnv -CommandArgs @(
        "python",
        "-m",
        "compileall",
        "-q",
        "app_src",
        "run_desktop_app.py",
        "packaging\windows\full_release.py",
        "packaging\windows\lightweight_release.py",
        "packaging\windows\package_naming.py"
    )

    Invoke-Native -FilePath $ResolvedNpmExe -CommandArgs @(
        "--prefix",
        "tests\js",
        "test"
    )

    $OriginalSkipUpdate = [Environment]::GetEnvironmentVariable(
        "SLEEP_SCORING_SKIP_UPDATE",
        "Process"
    )
    try {
        [Environment]::SetEnvironmentVariable("SLEEP_SCORING_SKIP_UPDATE", "1", "Process")
        Invoke-Conda -EnvName $TestEnv -CommandArgs @(
            "python",
            "run_desktop_app.py",
            "--smoke"
        )
    } finally {
        [Environment]::SetEnvironmentVariable(
            "SLEEP_SCORING_SKIP_UPDATE",
            $OriginalSkipUpdate,
            "Process"
        )
    }
}

if ($ValidateOnly) {
    Write-Host "Full-release validation passed. Packaging was not run."
    exit 0
}

$BuildParameters = @{
    BuildEnv = $BuildEnv
    TestEnv = $TestEnv
    CondaExe = $CondaExe
    # This wrapper already enforced the clean-worktree policy above.
    AllowDirty = $true
    # This wrapper already ran the complete source test suite above.
    SkipTests = $true
}
& $BuildScript @BuildParameters

foreach ($RequiredArtifact in @(
    $ExpectedZip,
    "$ExpectedZip.sha256.txt",
    "$ExpectedZip.manifest.json",
    "$ExpectedZip.build_env_requirements.txt"
)) {
    if (-not (Test-Path -LiteralPath $RequiredArtifact -PathType Leaf)) {
        throw "Missing expected full-release artifact: $RequiredArtifact"
    }
}

$ExpectedHash = (
    Get-Content -LiteralPath "$ExpectedZip.sha256.txt" -Raw
).Trim().Split()[0]
$ActualHash = (Get-FileHash -LiteralPath $ExpectedZip -Algorithm SHA256).Hash
if ($ActualHash -ne $ExpectedHash) {
    throw "Full-package checksum mismatch for $ExpectedZip"
}

$Manifest = Get-Content -LiteralPath "$ExpectedZip.manifest.json" -Raw |
    ConvertFrom-Json
if ($Manifest.version -ne $Version -or $Manifest.release_line -ne $ReleaseLine) {
    throw "Full-package manifest version/release-line mismatch"
}

Write-Host "Full-release candidate is ready: $ExpectedZip"
Write-Host "All candidate checks ran once through the standard full-release gate."
Write-Host "Tagging and publication remain explicit maintainer actions."
