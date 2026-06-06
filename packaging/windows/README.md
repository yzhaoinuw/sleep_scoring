# Windows Packaging

These scripts keep the current zip-based user workflow, while making the build
steps repeatable and easier to audit.

## Full App Zip

Use this when dependencies, `_internal/`, `run_desktop_app.exe`, `models/`, or
the PyInstaller runtime layout changed.

The default full app zip intentionally removes Torch, which is the largest
runtime dependency. It keeps the sDREAMER code and checkpoint files, so users
who need sDREAMER can enable it by placing the optional `torch` folder inside
`_internal/`.

This full zip is still the file to share with new Windows users. The generated
`build_env_requirements` sidecar is for release/debugging records, not a user
install step.

The generated app folder includes `Start Sleep Scoring.cmd`, a double-click
starter that unblocks the packaged app files and then launches
`run_desktop_app.exe`. It is included only in the full app zip, not in the
small `app_src` update zip.

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
```

Before creating the zip, the script checks that the release folder contains the
expected files, including the double-click starter, and runs
`run_desktop_app.exe --smoke` to verify the built launcher can import the
side-by-side `app_src/` folder.

## app_src Update Zip

Use this when changes are only in `app_src/`. Users unzip it and replace the
old `app_src/` folder in their existing app folder.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_app_update_zip.ps1
```

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_vX.Y.Z-app_src_update.zip
sleep_scoring_app_vX.Y.Z-app_src_update.zip.manifest.json
sleep_scoring_app_vX.Y.Z-app_src_update.zip.sha256.txt
```

## Local Test Builds

Both scripts normally require a clean worktree so release artifacts can be tied
to a commit. For local testing before committing, pass `-AllowDirty`. To skip
tests during packaging-script development, pass `-SkipTests`.
