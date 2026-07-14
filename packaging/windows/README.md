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

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1
```

The default build environment is `sleep_scoring_dash3.0_dist`; the default test
environment is `sleep_scoring_dash3.0`.

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_vX.Y.Z-windows.zip
sleep_scoring_app_vX.Y.Z-windows.zip.manifest.json
sleep_scoring_app_vX.Y.Z-windows.zip.sha256.txt
sleep_scoring_app_vX.Y.Z-windows.zip.build_env_requirements.txt
torch.zip
torch.zip.manifest.json
torch.zip.sha256.txt
```

Before creating the zip, the script checks that the release folder contains the
expected files, including the double-click starter. It runs
`run_desktop_app.exe --smoke` to verify that the built launcher can import the
side-by-side `app_src/` folder, then runs `run_desktop_app.exe --check-update`
against the configured GitHub Release endpoint. A metadata or updater failure
stops the package build instead of shipping a broken automatic update check.
The packaged `app_src/` files are written directly from the release commit's
Git blobs, without checkout or archive transformations, so their bytes match
the automatic-update manifests on Windows as well as source runs.

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

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_source_update_asset.ps1 -FromRef vX.Y.Z
```

When an older Windows ZIP contains different line endings from its Git tag,
pass the released package as an exact baseline, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_source_update_asset.ps1 `
  -FromRef v0.16.5 `
  -FromPackageZip "v0.16.5=release_artifacts\sleep_scoring_app_v0.16.5-windows.zip"
```

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_update_vX.Y.Z.zip
sleep_scoring_app_update_vX.Y.Z.zip.sha256.txt
```

Pass `-FromRef` more than once when one latest source update should support
users jumping from multiple older compatible versions.

## Manual app_src Update Zip

`make_app_update_zip.ps1` remains a fallback for manually replacing `app_src/`
when needed, but the automatic source update asset is the preferred code-only
release path for auto-update-enabled builds.

## Local Test Builds

Both scripts normally require a clean worktree so release artifacts can be tied
to a commit. For local testing before committing, pass `-AllowDirty`. To skip
tests during packaging-script development, pass `-SkipTests`.
