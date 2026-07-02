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

## 2026-07-01

### Startup Update Console Messages

- Added brief console messages for packaged startup update checks:
  checking, no update available, updated, not applied, or failed while
  continuing normal startup.
- Added focused launcher tests for the new startup-update console output and
  non-blocking failure/skip messages.
- Verification:
  - `conda run -n sleep_scoring_dash3.0 python -m pytest tests\test_run_desktop_app.py -q`
    -> `4 passed`.
  - `conda run -n sleep_scoring_dash3.0 python -m py_compile run_desktop_app.py tests\test_run_desktop_app.py`
    passed.
  - `conda run -n sleep_scoring_dash3.0 python -m pytest --basetemp .pytest_tmp\codex -p no:cacheprovider -q`
    -> `82 passed, 1 warning` (pre-existing `flask_caching` deprecation
    warning).
  - `conda run -n sleep_scoring_dash3.0 python -m black run_desktop_app.py tests\test_run_desktop_app.py`
    failed before formatting with Black internal `AssertionError: LAZY` in this
    conda environment.

### Experimental Auto-Update Baseline

- Created the `auto-update` experiment branch from current `dev`.
- Wired `run_desktop_app.py` to run `desktop_app_source_updater` before
  importing `app_src` in packaged builds, with environment-variable overrides
  for local source-update testing.
- Added `desktop-app-source-updater` to dependency metadata and PyInstaller
  hidden imports so the next full Windows package can bundle the updater.
- Added `packaging/windows/make_source_update_asset.ps1` for future GitHub
  Release source-update assets and documented the automatic update path in
  `README.md`, `packaging/windows/README.md`, and `AGENTS.md`.
- Bumped the experimental app/source-install version metadata to
  `v0.16.4.post1`.
- Verification:
  - `conda run -n sleep_scoring_dash3.0 python -m pytest --basetemp .pytest_tmp\codex -p no:cacheprovider -q`
    -> `78 passed, 1 warning` (pre-existing `flask_caching` deprecation
    warning).
  - `conda run -n sleep_scoring_dash3.0 python run_desktop_app.py --smoke`
    -> `Sleep Scoring App v0.16.4.post1 smoke check OK`.
  - `conda run -n sleep_scoring_dash3.0 python -m py_compile run_desktop_app.py`
    passed.
  - PowerShell parser check for `packaging/windows/make_source_update_asset.ps1`
    passed.
  - `git diff --check` passed.

## 2026-06-30

### v0.16.4 Unscored Save Reminder

- Added a Save Annotations completeness reminder that reports the first
  unscored sleep-score range as `[start, end] (duration s)` when any part of
  the recording is still unscored.
- The reminder is computed before the native `.mat` save dialog returns, so it
  still appears in the annotation message if the user cancels the `.mat` save.
- Reused the same unscored check to gate sleep-bout spreadsheet export, keeping
  the existing behavior that Excel export is offered only after complete
  scoring.
- Documented the Save Sleep Scores reminder in `README.md` and bumped the
  app/source-install version metadata to `v0.16.4`.
- Added focused tests for `-1`, `NaN`, and `None` unscored labels plus the
  canceled-save reminder path.
- Verification:
  - `conda run -n sleep_scoring_dash3.0 python -m pytest --basetemp .pytest_tmp\codex -p no:cacheprovider -q`
    -> `78 passed, 1 warning` (pre-existing `flask_caching` deprecation
    warning).
- Published commit `f900346` to `dev` and `main`, pushed annotated tag
  `v0.16.4`, and verified `dev`, `main`, `origin/dev`, `origin/main`, and
  `v0.16.4^{}` all resolved to
  `f900346bc9f3cc44737734c47365740c3aa8008a`.

### Contributor Workflow Docs

- Added GitHub issue forms for app bugs, data file problems, feature requests,
  and questions, plus a pull request template for future collaborator changes.
- Published the templates through `dev` and fast-forwarded `main` so GitHub's
  default-branch issue flow shows the new issue chooser.
- Refactored `CONTRIBUTING.md` into a human-contributor section and an
  agent-collaborator section, including PR targets, environment setup, test
  expectations, and guidance for using AI agents.
- Added `AGENTS.md` guidance for recurring Windows Git credential-helper and
  lock-file friction, including the known-good PowerShell push shape and
  post-operation ref verification.
- Corrected the Git-friction guidance to say agents should request
  approval/escalation up front for known-friction switch, merge, fetch, push, and
  tag/ref operations after checking state, instead of burning a failed first
  attempt.
- Trimmed `CONTRIBUTING.md` pull-request guidance to link out to GitHub's
  general contributor guide while keeping only project-specific expectations in
  this repo, and linked issue reporters directly to the repository Issues page.

## 2026-06-24

### v0.16.3 Publish

- Bumped the app/source-install version metadata to `v0.16.3` for the
  `fp_frequency` alias compatibility release.
- Planned publish path: push `dev`, fast-forward `main`, push `main`, then
  create and push the `v0.16.3` tag.

## 2026-06-23

### NE Sampling-Rate Field Alias (`fp_frequency`)

- Added `fp_frequency` as an accepted alias for `ne_frequency`. Recordings
  that carry an `ne` signal but name the fiber-photometry sampling rate
  `fp_frequency` (depending on the upstream preprocessing tool) are now
  handled the same as those using `ne_frequency`, including in the
  visualization.
- Introduced `app_src/mat_utils.py` with a read-only `get_ne_frequency(mat)`
  helper: prefers `ne_frequency`, falls back to `fp_frequency`, returns `None`
  when neither is present. Read-only by design, so the alias is never written
  back into the user's `.mat` on save (both save paths copy every non-`_`
  key into the output).
- Routed all five `ne_frequency` read sites through the helper: the
  visualization (`make_figure.py`), NE reshaping in `reshape_sleep_data_ne`
  (`preprocessing.py`, previously a hard-indexed `mat["ne_frequency"]`
  `KeyError`), REM-transition validation (`postprocessing.py`), and the
  stats_model NE feature/timing paths (`run_inference_stats_model.py`, two
  sites).
- Behavior change to note: files with `ne` + `fp_frequency` but no
  `ne_frequency` previously crashed in the visualization and were scored
  NE-blind on the stats_model/postprocessing paths (the frequency resolved to
  `None` and NE handling was silently skipped); they are now scored NE-aware.
- Documented the alias in `README.md` (input-field table and sampling-rate
  note).
- Verification:
  - `python -m pytest -q` -> `73 passed, 1 warning` (pre-existing
    flask_caching deprecation warning, unrelated).
  - Added `tests/test_mat_utils.py` (precedence, fp fallback, present-but-`None`
    fallback, neither-present -> `None`, no-mutation), a
    `mock_mat_data_with_fp_alias` fixture in `tests/conftest.py`, and a
    `reshape_sleep_data_ne` consumer test in `tests/test_preprocessing.py`
    exercising the alias end-to-end.

### Live-Log Rotation

- Adding today's date pushed the live log past five unique dates, so rotated
  2026-06-06 through 2026-06-15 as a chunk into
  `work_log_archive/work_log_2026-06-06_to_2026-06-15.md` (content identical to
  the same-named archive already present on the `publication` branch).
