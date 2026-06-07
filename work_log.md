# Work Log

Prepend new session notes to the top of this file. If you log multiple
sessions on the same calendar date, add a new `###` subsection under the
existing `## YYYY-MM-DD` header instead of starting a second header for the
same date.

Historical verification commands may include absolute paths from the original
development machine. When replaying or adapting them, keep the project folder
name `sleep_scoring` and conda environment name `sleep_scoring_dash3.0`, but
replace the user/home prefix and clone location with the collaborator's local
setup.

Reading note for agents: this file holds at most the 5 most recent unique
calendar dates. Older entries are rotated in chunks of 5 dates into
`work_log_archive/work_log_<earliest>_to_<latest>.md`. Default to reading the
two most recent dated entries; search older entries with targeted terms using
the `^## [0-9]{4}-[0-9]{2}-[0-9]{2}` anchor, or open the relevant archive file
by its date range. See `AGENTS.md` for the full rotation policy.

## 2026-06-07

### Next Steps Cleanup

- Condensed the `next_steps.md` Installation Packaging section so it lists only remaining open items instead of repeating completed packaging status.
- Left completed packaging/rebuild evidence in `work_log.md` and settled workflow detail in `packaging/windows/README.md`.

### Release Artifact Rebuild

- Rebuilt the full Windows app artifact and app-src update artifact in `release_artifacts/` after the launcher/module cleanup.
- The first post-reboot full zip sidecar check found a corrupted SHA sidecar from the interrupted run, so the full Windows zip was rebuilt from scratch again.
- Verification:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_app_update_zip.ps1 -AllowDirty` -> `66 passed, 1 warning`, AppUpdate smoke check passed.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1 -AllowDirty` -> `66 passed, 1 warning`, Full release smoke check passed, `run_desktop_app.exe --smoke` passed.
  - Verified both generated SHA sidecars match their manifests and computed zip hashes.

### Launcher And Active Script Cleanup

- Renamed the full Windows package starter from `Start Sleep Scoring.cmd` to `unblock_app.cmd`.
- Folded the previous `unblock_and_start.ps1` behavior into the single `.cmd` launcher so the full app package ships one double-click starter.
- Updated `README.md` so Automatic Sleep Scoring explains the statistical model, `SLEEP_SCORING_MODEL` switching, and the stats-model tuning parameters in `app_src/config.py`.
- Removed inactive/scratch `app_src` files (`app.py`, `components.py`, `make_figure.py`, `debug_tool.py`) after moving the active `*_dev` modules into the unsuffixed names.
- Updated runtime imports, tests, packaging checks, `next_steps.md`, and `project_overview.md` to use the unsuffixed active module names.
- Verification:
  - `C:\Users\yzhao\miniconda3\condabin\conda.bat run -n sleep_scoring_dash3.0 python -m py_compile run_desktop_app.py app_src\app.py app_src\components.py app_src\make_figure.py app_src\run_inference_stats_model.py app_src\inference.py`
  - `C:\Users\yzhao\miniconda3\condabin\conda.bat run -n sleep_scoring_dash3.0 pytest --basetemp .pytest_tmp\codex -p no:cacheprovider` -> `66 passed, 1 warning`

## 2026-06-06

### Installation Packaging Docs

- User manually confirmed the generated no-Torch full app zip can be unzipped and launched on Windows.
- Updated `README.md` so Windows installation and Automatic Sleep Scoring explain the current distribution model: the app zip includes sDREAMER code/checkpoints but not the optional Torch runtime, and users who need automatic scoring place the unzipped `torch` folder directly inside `_internal/`.
- Updated `packaging/windows/README.md` to clarify that the full app zip is still the file shared with new Windows users, while generated build-env requirement snapshots are release/debugging records.
- Updated `next_steps.md` to mark the full app zip manual launch as validated and to use `app_src` update wording consistently.
- Added `.pytest_tmp` parent-directory creation inside both packaging scripts so clean builds can run the repo-local pytest basetemp path on Windows.
- Added `Start Sleep Scoring.cmd` and `unblock_and_start.ps1` to the full Windows app package so users can double-click a starter that unblocks packaged files and launches `run_desktop_app.exe`.
- Removed obsolete root-level PyInstaller specs now that the active Windows spec lives under `packaging/windows/`, and updated `project_overview.md` to describe the new packaging layout.
- Rotated the previous five live work-log dates into `work_log_archive/work_log_2026-05-25_to_2026-06-05.md` per the live-log size policy.
