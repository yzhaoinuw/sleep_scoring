# Work Log

Prepend new session notes to this file. Record project decisions, reusable
evidence, and shared-state changes—not routine content already explained by the
diff. Each session ends with a `- Verification:` subsection. See
[`treaty_conventions.md`](treaty_conventions.md#work-log-discipline).

If today's date is already at the top, add a new `###` subsection beneath it.
The live log holds at most five unique dates; when a new date would exceed that
limit, move the oldest five-date chunk to
`work_log_archive/work_log_<earliest>_to_<latest>.md`.

Historical commands can contain machine-specific paths. When replaying them,
keep the `sleep_scoring` folder and `sleep_scoring_dash3.0` environment names
but adapt the user prefix and clone location. Default to the two newest dates;
search older entries by date anchor rather than reading every archive.

## 2026-09-15

### NREM-anchored Active/Quiet Wake threshold (Codex GPT-5; effort/tokens not reported)

- Replaced the circular Wake-distribution reference with a valid-NREM EMG reference:
  the automatic initial cutoff is the NREM RMS 75th percentile plus two MAD-derived
  robust standard deviations. A missing automatic baseline now fails transparently;
  a configured numeric threshold remains the explicit fallback.
- Adopted a one-second automatic Active Wake minimum at the user's request. Manual
  Active/Quiet labels remain authoritative and can refine the NREM-derived initial
  cutoff without becoming future-run annotations.
- Wake-activity metadata now records the initial cutoff, NREM baseline statistics,
  and versioned algorithm provenance. The NE-analysis methods document the upstream
  scoring contract and its pilot-validation boundary.
- Soft-locked the current 20 Hz intermediate RMS envelope, 0.5-second RMS/gap
  settings, and one-second labels. The detector docstrings, rather than the README,
  hold the detailed signal-processing explanation; compare future reviewed labels
  with a direct one-RMS-value-per-second alternative before making envelope rate a
  user-facing experimental parameter.
- Verification:
  - Focused detector/persistence/callback pytest run: 58 passed, one existing
    Flask-Caching deprecation warning.
  - Full pytest run: 254 passed, one existing Flask-Caching deprecation warning.
  - Pinned Black pre-commit hook passed for the changed Python files using
    repository-local caches; `python run_desktop_app.py --smoke` passed.
  - `git diff --check` passed.
  - After the documentation relocation, focused wake-activity/pipeline pytest:
    25 passed, one existing Flask-Caching deprecation warning.

### Experimental branch delivery (Codex GPT-6; effort/tokens not reported)

- User authorized committing and pushing `active-quiet-wake` only; keep the
  experiment separate from `dev`/`main`, with no release, tag, or PR.
- Preserved the working-copy setting `STATS_MODEL_DETECT_WAKE_ACTIVITY = True`.
  Fine annotations alone do not enable subdivision; the config flag controls it
  and changing the flag requires restarting the app.
- Pre-delivery checks exposed four coarse-only test cases that depended on the
  local flag being off. Those tests now explicitly disable subdivision; existing
  enabled-pipeline tests still exercise complete Wake coverage and repeatability.
- Verification:
  - Conda pytest: `python -m pytest --basetemp
    .pytest_tmp\codex-wake-push-0915-fixed -p no:cacheprovider -q`: 250 passed,
    one existing Flask-Caching deprecation warning.
  - JavaScript: `npm.cmd test -- --runInBand` in `tests/js`: 57 passed.
  - `python run_desktop_app.py --smoke`: passed.
  - Pinned Black hook passed for tracked files and the three new Python files;
    `git diff --check` passed.

## 2026-09-13

### Config-only wake activity and repeatability (Codex GPT-6; effort/tokens not reported)

- Removed the experimental settings panel and EMG preview at the user's request.
  The existing prediction action reads wake settings solely from `config.py`;
  permanent manual keys 5/6 and stage colors remain available.
- Existing coarse-scored MAT files are the pilot entry point: absent or empty
  annotation fields fall back to saved scores. Explicit all-unscored annotation
  arrays remain empty, preserving Clear and preventing automatic-only outputs
  from becoming training examples after reload.
- All Wake, including coarse manual annotations, is subdivided on a successful
  enabled run. Coarse labels never become Quiet training examples. Repeated runs
  use the same configured baselines rather than previously fitted thresholds.
- Real signal-processing tests exercise independent coarse and EMG repeatability,
  reapplying subdivision to its own output with unchanged annotations, and full
  legacy-MAT load/predict/save/reload flows with and without fine labels. Both
  stages return identical results for unchanged inputs. Collect reviewed fine-label
  MAT files for the future NE pipeline; NE analysis itself remains out of scope.
- Work remains uncommitted on local `active-quiet-wake`, pending user testing.
- Verification:
  - Focused Conda pytest run covering pipeline, annotation layers, and app helpers:
    48 passed, one existing Flask-Caching deprecation warning.
  - Full `python -m pytest --basetemp .pytest_tmp\codex-wake-config-final
    -p no:cacheprovider -q`: 250 passed, the same deprecation warning.
  - `npm.cmd test -- --runInBand` in `tests/js`: 57 passed.
  - `python run_desktop_app.py --smoke`: passed.
  - Pinned repository Black hook and `git diff --check`: passed.

## 2026-09-10

### Wake activity pilot validation (Codex GPT-6; effort/tokens not reported)

- The two calibrations remain sequential and separate. Explicit subtype labels
  force Wake eligibility, calibrate the EMG threshold before manual overrides,
  and remain authoritative even when shorter than the configured minimum.
- Keep the five-second duration fixed during adaptation. The pilot bridges
  at most 0.5-second gaps and measures bouts on a 20 Hz RMS envelope before
  assigning one-second scores. These settings require real-recording/video
  feedback; synthetic checks establish software behavior, not behavioral accuracy.
- Save sparse annotations independently from predictions to prevent accidental
  self-calibration after reload. Legacy files retain existing annotation semantics.
  Last-run detector metadata records provenance and is not a claim that the current
  manually edited or undone labels are identical to that automatic run.
- Coarse summaries merge the Wake family before existing short-Wake-to-MA rules;
  subtype bout exports preserve the experimental labels. Invalid Wake EMG aborts
  subdivision without replacing the current scores, rather than declaring it quiet.
- Local `active-quiet-wake` is ready for user pilot testing, with changes uncommitted
  and unpushed. Official `dev` and `main` remain at the trace-identification commit.
- Verification:
  - Conda `sleep_scoring_dash3.0`: full `python -m pytest --basetemp
    .pytest_tmp\codex-wake-final2 -p no:cacheprovider -q`: 241 passed, one existing
    Flask-Caching deprecation warning. Includes fine-label undo, feature-off
    preservation, MAT/Excel round trips, and two-stage prediction/error paths.
  - `npm.cmd test -- --runInBand` in `tests/js`: 57 passed.
  - `python run_desktop_app.py --smoke`: passed.
  - Pinned repository Black hook and `git diff --check`: passed.
  - Browser QA with an isolated synthetic recording: both calibrations reported
    ten manual seconds; automatic subdivision and RMS overlay rendered; key 6
    applied a one-second correction with detection off, and Undo restored it.

### Active/quiet Wake calibration design (Codex GPT-6; effort/tokens not reported)

- Created and switched to local `active-quiet-wake` from `9ebcad2`, the verified
  trace-identification commit on both `dev` and `main`. Feature code is not yet
  implemented; this branch is not pushed.
- Keep two sequential calibration steps: manual Wake, Active Wake, and Quiet
  Wake all supply Wake targets to the stats model; only explicit Active/Quiet
  annotations supply subtype targets to the EMG detector. Generic Wake must
  not be treated as a Quiet Wake example.
- Apply coarse manual corrections before constructing the detector's Wake mask,
  so fine-grained manual examples remain inside Wake even if the raw model
  disagrees. Evaluate each detector candidate before applying fine-grained
  overrides, and never turn automatic subtype predictions into manual targets.
- For the pilot, propose adapting the EMG amplitude threshold while keeping
  the user-selected minimum duration (default five seconds) fixed. Explicit
  fine-grained manual labels remain authoritative after prediction.
- Verification:
  - Inspected current calibration masks and manual-overlay helpers.
  - `git status --short --branch` and `git rev-parse HEAD dev main` confirmed the
    new branch started clean at the same commit as both official branches.

### Stable score-trace identification (Codex GPT-6; effort/tokens not reported)

- Score selection, annotation, and repainting now identify heatmaps through
  `meta.role = "sleep_scores"`. Visible names and draw order remain unchanged;
  additional traces can be inserted without redirecting edits to another signal.
- Missing score roles cause no update rather than a positional fallback. Real
  figure serialization with and without NE retains the roles on all three overlays.
- Kept this prerequisite separate from active/quiet Wake work. The user confirmed
  the interactive check passed and authorized commit/push, with delivery to both
  `dev` and `main` before starting the experiment.
- Verification:
  - `conda run -n sleep_scoring_dash3.0 npm.cmd test -- --runInBand` in
    `tests/js`: 51 passed, including reordered/appended traces and renamed overlays.
  - `conda run -n sleep_scoring_dash3.0 python -m pytest tests/test_smoke.py
    tests/test_app_helpers.py --basetemp .pytest_tmp\codex-trace-roles
    -p no:cacheprovider -q`: 35 passed, one Flask-Caching deprecation warning.
  - `conda run -n sleep_scoring_dash3.0 python run_desktop_app.py --smoke`: passed.
  - Repository Black hook passed for the two touched Python files using ignored
    repository-local pre-commit, virtualenv, and Black caches. The direct Conda
    Black invocation hit an environment grammar assertion; the pinned hook worked.
  - `git diff --check`: passed. The user also confirmed the app works interactively.

## 2026-09-09

### Adaptive settings and prediction-message lifecycle (Codex GPT-5; effort/tokens not reported)

- Exposed the two REM low-NE defaults in `config.py` alongside the existing
  statistical-model settings, and report the per-recording tuned configuration
  after adaptive scoring.
- Separated prediction feedback from annotation/save feedback. Repainting the
  score heatmap writes during prediction completion, so sharing the annotation
  message caused a last-writer-wins race that could erase the tuned settings.
- Kept the existing annotation/save timer unchanged. The dedicated prediction
  timer is reset at prediction start, then clears ordinary completion text
  after five seconds or calibrated settings after sixty seconds.
- Rotated the prior five-date live work log intact to
  `work_log_archive/work_log_2026-08-13_to_2026-09-01.md`.
- Verification:
  - Focused Python checks passed: 37 tests (one Flask-Caching deprecation
    warning).
  - Client-side Jest checks passed: 40 tests.
  - The repository-pinned Black hook and `python run_desktop_app.py --smoke`
    passed.
  - `git diff --check` passed.

### v0.17.3 lightweight source-update candidate (Codex GPT-5; effort/tokens not reported)

- Prepared the v0.17.3 lightweight update for the exposed REM low-NE
  configuration settings and the briefly displayed adaptive tuning results.
  Those new settings are included in the schema-2 editable allowlist, so later
  source updates retain user-selected values.
- Corrected the lightweight gate's stale v0.17.0 fixture policy. v0.17.0
  crosses the frozen launcher/runtime boundary and cannot receive a source-only
  update; the gate now validates the supported v0.17.1 full base and v0.17.2
  patched state instead.
- Verification:
  - The full Python suite passed: 214 tests (one Flask-Caching deprecation
    warning). A one-time port-allocation retry passed after a neighboring
    ephemeral port was occupied.
  - Black, compilation, 40 client-side Jest tests, and the v0.17.3 source
    smoke check passed.
  - The schema-2 asset validated against v0.17.1 and v0.17.2. The fresh
    v0.17.1 frozen-app update and smoke check passed.
  - Candidate asset SHA-256:
    `0F8C337484375E855704F82C79CA6BE88566DB649ABA6039B13429F64655DBC0`.
