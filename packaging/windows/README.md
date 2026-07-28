# Windows Packaging

These scripts keep the current zip-based user workflow, while making the build
steps repeatable and easier to audit.

## Full App Zip

Use this when dependencies, `_internal/`, `run_desktop_app.exe`, `models/`, the
PyInstaller runtime layout, or the bundled auto-updater changed.

The default full app zip intentionally removes Torch, which is the largest
runtime dependency. It keeps the sDREAMER code and checkpoint files, so users
who need sDREAMER can enable it by copying the optional sDREAMER Torch runtime
contents directly into `_internal/`.

This full zip is still the file to share with new Windows users. The generated
`build_env_requirements` sidecar is for release/debugging records, not a user
install step.

The generated app folder includes `unblock_app.cmd`, a double-click starter
that contains the unblock step and then launches `run_desktop_app.exe`. It is
included only in the full app zip, not in the small `app_src` update zip.

Use the standard one-command gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_full.ps1
```

During development, run the same preflight and quality checks without invoking
PyInstaller:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_full.ps1 -ValidateOnly -AllowDirty
```

The default build environment is `sleep_scoring_dash3.0_dist`; the default test
environment is `sleep_scoring_dash3.0`.

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_vX.Y-windows.zip
sleep_scoring_app_vX.Y-windows.zip.manifest.json
sleep_scoring_app_vX.Y-windows.zip.sha256.txt
sleep_scoring_app_vX.Y-windows.zip.build_env_requirements.txt
torch.zip
torch.zip.manifest.json
torch.zip.sha256.txt
```

The full ZIP and its extracted top-level folder use only the major/minor
release line, for example `sleep_scoring_app_v0.17`. The exact patch version,
such as `v0.17.0`, remains authoritative in the manifest, startup terminal
message, and app window title. This prevents an installation folder name from
becoming a stale version report after source-only updates.

Before creating the zip, the script checks that the release folder contains the
expected files, including the double-click starter. It runs
`run_desktop_app.exe --smoke` to verify that the built launcher can import the
side-by-side `app_src/` folder, verifies that the packaged config exposes
`STAGE_COLORS`, then runs `run_desktop_app.exe --check-update` in forced-check
mode. A discovery or updater failure stops the package build instead of
shipping a broken automatic update check. The build also refuses to run when
the updater installed in the PyInstaller environment does not match the exact
commit pinned in `requirements.txt`.
The packaged `app_src/` files are written directly from the release commit's
Git blobs, without checkout or archive transformations, so their bytes match
the automatic-update manifests on Windows as well as source runs.

`release_full.ps1` is the complete candidate gate. Do not rerun its individual
pytest, Black, JavaScript, compile, source-smoke, build, packaged-smoke, updater,
or checksum checks by hand after it passes unless the candidate commit changes.
The pushed candidate must still pass CI, but that confirmation does not require
repeating the same local checks. Tagging and GitHub publication remain one
explicit maintainer step after reviewing the artifacts and release notes.

## Optional sDREAMER Torch Runtime Zip

`make_full_app_zip.ps1` builds the app with Torch available so PyInstaller can
discover Torch, TorchVision, and related hidden imports. The script then creates
the optional runtime zip from the built `_internal\torch` folder and removes
that folder before zipping the main app. This keeps the full app zip smaller
without losing imports such as `cProfile` that Torch loads later.

The generated runtime zip does not contain an `_internal/` folder itself. Users
copy its contents directly into the app's existing `_internal/` folder. After
copying, `_internal\torch` should exist.

## Automatic Source Update Asset

Use this for future code-only releases when changes are only in `app_src/` and
the installed full app already includes the auto-updater. Attach the generated
zip to the matching GitHub Release; users do not unzip it manually.

For a release candidate, use the one-command gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_lightweight.ps1
```

It discovers every compatible tag from v0.16.5 through the release immediately
before the target version. It then validates the lightweight-release boundary,
runs pytest, Black, JavaScript, compile, and source smoke checks once, builds
and validates the schema-1 update asset, and tests it against:

- a fresh v0.16.5 Windows package;
- a fresh v0.16.6 Windows package; and
- v0.16.5 patched through the v0.16.6 source update.

The retained full-package ZIPs are fixture inputs in the ignored
`release_artifacts/` folder. Exact installed hashes are tracked compactly under
`installed_baselines/`, so normal asset construction no longer needs to open
the large package ZIPs just to align line-ending variants.

When a future full Windows base is introduced, create its compact baseline with
`lightweight_release.py export-baseline`, review the generated JSON, and update
the fixture policy in `release_lightweight.ps1` in the same change. Use
`-FixtureArtifactDir` when retained fixture ZIPs live somewhere other than
`release_artifacts/`.

The command allows `setup.py` only when its version assignment is the sole
change and it matches `app_src/__init__.py`. It refuses dependency, model,
launcher/updater, PyInstaller, release-helper, runtime deletion, and runtime
rename changes. Those require a new full Windows package.

The source asset still uses manifest schema 1. The newer shared updater checkout
is used only as a maintainer-side builder; the updater frozen into existing user
distributions does not change. A newly changed `app_src/config.py` is refused
because schema 1 must preserve the installed copy; schema-2 configuration
merging remains reserved for a future full redistribution.

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_update_vX.Y.Z.zip
sleep_scoring_app_update_vX.Y.Z.zip.sha256.txt
```

`make_source_update_asset.ps1` remains the lower-level builder for focused
packaging work. Pass `-FromRef` more than once for jump-ahead support and
`-InstalledBaselineManifest` more than once for compact package byte lineages.
`-FromPackageZip` remains available as a compatibility fallback.

Neither command publishes a GitHub Release automatically. Tagging and
publication remain an explicit maintainer action after reviewing the candidate
artifact and release notes.

## Manual app_src Update Zip

`make_app_update_zip.ps1` remains a fallback for manually replacing `app_src/`
when needed, but the automatic source update asset is the preferred code-only
release path for auto-update-enabled builds.

## Local Test Builds

Both scripts normally require a clean worktree so release artifacts can be tied
to a commit. For local testing before committing, pass `-AllowDirty`. To skip
tests during lower-level packaging-script development, pass `-SkipTests`.
`release_lightweight.ps1` also has `-SkipQualityChecks` and
`-SkipInstalledAppTests`; do not use either switch for a published release.
