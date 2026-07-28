param(
    [string[]]$FromRef = @(),
    [string]$MinimumCompatibleVersion = "v0.16.5",
    [string]$ToRef = "HEAD",
    [string]$TestEnv = "sleep_scoring_dash3.0",
    [string]$CondaExe = "",
    [string]$NpmExe = "",
    [string]$UpdaterRepo = "",
    [string]$ArtifactDir = "",
    [string]$FixtureArtifactDir = "",
    [switch]$SkipQualityChecks,
    [switch]$SkipInstalledAppTests,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$ReleaseHelper = Join-Path $ScriptDir "lightweight_release.py"
$BuildScript = Join-Path $ScriptDir "make_source_update_asset.ps1"
$BaselineDir = Join-Path $ScriptDir "installed_baselines"

if (-not $ArtifactDir) {
    $ArtifactDir = Join-Path $Repo "release_artifacts"
}
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)
if (-not $FixtureArtifactDir) {
    $FixtureArtifactDir = Join-Path $Repo "release_artifacts"
}
$FixtureArtifactDir = [System.IO.Path]::GetFullPath($FixtureArtifactDir)

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
        [string[]]$CommandArgs
    )

    Invoke-Native -FilePath $CondaExe -CommandArgs (@("run", "-n", $TestEnv) + $CommandArgs)
}

function Invoke-CondaCapture {
    param(
        [string[]]$CommandArgs
    )

    return Invoke-NativeCapture -FilePath $CondaExe -CommandArgs (
        @("run", "-n", $TestEnv) + $CommandArgs
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
        throw "Worktree is not clean. Commit the release candidate before running this command."
    }
}

if ($FromRef.Count -eq 0) {
    $CompatibleOutput = Invoke-CondaCapture -CommandArgs @(
        "python",
        $ReleaseHelper,
        "compatible-refs",
        "--repo",
        $Repo,
        "--minimum-version",
        $MinimumCompatibleVersion,
        "--to-ref",
        $ToRef
    )
    $FromRef = @(
        $CompatibleOutput -split "\r?\n" |
            Where-Object { $_ }
    )
}
if ($FromRef.Count -eq 0) {
    throw "No compatible release refs were found before $ToRef."
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
Invoke-Conda -CommandArgs $CandidateArgs

$Version = Invoke-CondaCapture -CommandArgs @(
    "python",
    $ReleaseHelper,
    "version",
    "--repo",
    $Repo,
    "--ref",
    $ToRef
)
$Version = $Version.Trim()
$OutputZip = Join-Path $ArtifactDir "sleep_scoring_app_update_$Version.zip"

Write-Host "Preparing lightweight release $Version"
Write-Host "Compatible refs: $($FromRef -join ', ')"

if (-not $SkipQualityChecks) {
    $ResolvedNpmExe = Resolve-NpmExecutable -ConfiguredPath $NpmExe
    Invoke-Native -FilePath $ResolvedNpmExe -CommandArgs @("--version")

    New-Item -ItemType Directory -Force -Path (Join-Path $Repo ".pytest_tmp") | Out-Null
    $PytestBaseTemp = Join-Path (
        Join-Path $Repo ".pytest_tmp"
    ) ("lightweight_release_" + [guid]::NewGuid().ToString("N"))
    Invoke-Conda -CommandArgs @(
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
        Invoke-Conda -CommandArgs @("pre-commit", "run", "black", "--all-files")
    } finally {
        [Environment]::SetEnvironmentVariable(
            "PYTHONUTF8",
            $OriginalPythonUtf8,
            "Process"
        )
    }
    Invoke-Conda -CommandArgs @(
        "python",
        "-m",
        "compileall",
        "-q",
        "app_src",
        "run_desktop_app.py",
        "packaging\windows\lightweight_release.py"
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
        Invoke-Conda -CommandArgs @("python", "run_desktop_app.py", "--smoke")
    } finally {
        [Environment]::SetEnvironmentVariable(
            "SLEEP_SCORING_SKIP_UPDATE",
            $OriginalSkipUpdate,
            "Process"
        )
    }
}

$InstalledBaselines = @(
    (Join-Path $BaselineDir "v0.16.5-windows.json"),
    (Join-Path $BaselineDir "v0.16.6-windows.json"),
    (Join-Path $BaselineDir "v0.16.6-from-v0.16.5-source-update.json")
)
foreach ($Baseline in $InstalledBaselines) {
    if (-not (Test-Path -LiteralPath $Baseline)) {
        throw "Missing tracked installed baseline: $Baseline"
    }
}

$BuildParameters = @{
    FromRef = $FromRef
    ToRef = $ToRef
    TestEnv = $TestEnv
    CondaExe = $CondaExe
    Output = $OutputZip
    InstalledBaselineManifest = $InstalledBaselines
    SkipTests = $true
    # This wrapper already enforced the clean-worktree policy above.
    AllowDirty = $true
}
if ($UpdaterRepo) {
    $BuildParameters["UpdaterRepo"] = $UpdaterRepo
}
& $BuildScript @BuildParameters

if (-not $SkipInstalledAppTests) {
    $V0165Package = Join-Path $FixtureArtifactDir "sleep_scoring_app_v0.16.5-windows.zip"
    $V0166Package = Join-Path $FixtureArtifactDir "sleep_scoring_app_v0.16.6-windows.zip"
    $V0166Update = Join-Path $FixtureArtifactDir "sleep_scoring_app_update_v0.16.6.zip"
    foreach ($FixtureArtifact in @($V0165Package, $V0166Package, $V0166Update)) {
        if (-not (Test-Path -LiteralPath $FixtureArtifact)) {
            throw (
                "Missing installed-app fixture artifact: $FixtureArtifact. " +
                "Restore the retained release artifact or use -SkipInstalledAppTests " +
                "for packaging-script development only."
            )
        }
    }

    Write-Host "Testing fresh v0.16.5 full-package fixture"
    Invoke-Conda -CommandArgs @(
        "python",
        $ReleaseHelper,
        "test-installed-update",
        "--package-zip",
        $V0165Package,
        "--expected-initial-version",
        "v0.16.5",
        "--update-zip",
        $OutputZip
    )

    Write-Host "Testing fresh v0.16.6 full-package fixture"
    Invoke-Conda -CommandArgs @(
        "python",
        $ReleaseHelper,
        "test-installed-update",
        "--package-zip",
        $V0166Package,
        "--expected-initial-version",
        "v0.16.6",
        "--update-zip",
        $OutputZip
    )

    Write-Host "Testing v0.16.5 patched through the v0.16.6 source update"
    Invoke-Conda -CommandArgs @(
        "python",
        $ReleaseHelper,
        "test-installed-update",
        "--package-zip",
        $V0165Package,
        "--expected-initial-version",
        "v0.16.5",
        "--prerequisite-update",
        $V0166Update,
        "--update-zip",
        $OutputZip
    )
}

$Hash = Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256
Write-Host "Lightweight release candidate is ready: $OutputZip"
Write-Host "SHA256: $($Hash.Hash)"
Write-Host "Publication remains an explicit maintainer step."
