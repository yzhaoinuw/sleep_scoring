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

## 2026-10-06

### v0.17.4 partial-release candidate (Codex GPT-6; effort/tokens not reported)

- The maintainer tested multiple MAT/AVI pairs, varied snippet timing and
  duration, and switching before playback finished; all worked normally.
  Proceed with a partial release of the cleanup improvement. The original
  reporter's frozen-frame failure remains unconfirmed.
- Candidate v0.17.4 includes native video playback, nonfatal cleanup of locked
  clips, and the already-integrated score-overlay identification change since
  v0.17.3. Experimental Active/Quiet Wake work stays on its separate branch.
- Align app/setup/CFF versions and the verified 2026-10-06 release date. No
  dependency, launcher, or frozen runtime changes are needed. Correct the
  packaging README's stale v0.17.0 fixture description to match the existing
  v0.17.1 release gate; no packaging code changes.
- Supported update baselines are v0.17.1, v0.17.2, and v0.17.3. Commit the
  candidate, then run the full lightweight gate before any tag or publication.
- Verification:
  - `Get-Date -Format yyyy-MM-dd`: 2026-10-06.
  - Fetched `origin/dev`, `origin/main`, and tags; local and remote dev/main
    all start at `9ebcad2`. GitHub's latest release is v0.17.3.
  - Prior focused validation: 55 tests, Black, source smoke, and ten browser
    player replacement cycles passed; see 2026-10-02. The maintainer's normal
    desktop playback checks above add interactive coverage.
  - Release gate and publication are pending for this candidate.

## 2026-10-02

### Video cleanup improvement on dev (Codex GPT-6; effort/tokens not reported)

- The maintainer could not reproduce the reported freeze after more than ten
  varied snippets and authorized improving the confirmed cleanup defects.
  Treat this as cleanup work: the earlier timer-leak reproduction did not
  establish a causal link to the reporter's frozen frame or a failure threshold.
- Use Dash's native HTML video element for local MP4 playback. The app consumes
  none of DashPlayer's polled properties; replacing that wrapper avoids its
  unmount interval leak without patching compiled third-party JavaScript.
  Native controls and preloading remain enabled, with the existing clip
  extraction, timing, and per-window storage. Keep dependency/package metadata
  unchanged for this local runtime change.
- Old MP4 deletion is best effort: a Windows-open clip remains for a later
  cleanup instead of preventing generation of a different snippet. Regression
  coverage verifies generation while one file is locked, deletion of other old
  clips, retry after release, and preservation of non-video files.
- Switched the clean checkout from `active-quiet-wake` to `dev`; the experimental
  branch and earlier investigation-note stash remain parked. Changes are local
  and uncommitted; no push or release was requested.
- Verification:
  - `conda run --no-capture-output -n sleep_scoring_dash3.0 python -m pytest
    tests/test_app_helpers.py tests/test_multi_session.py tests/test_smoke.py
    --basetemp .pytest_tmp/video-cleanup -p no:cacheprovider -q`: 55 passed,
    one existing Flask-Caching deprecation warning.
  - Repository-pinned Black hook passed for the three touched Python files;
    `conda run --no-capture-output -n sleep_scoring_dash3.0 python
    run_desktop_app.py --smoke`: passed.
  - Isolated browser harness calling the actual `show_clip` component completed
    ten mount/remove cycles across three AVI-derived clips: zero player timers,
    zero JavaScript/media errors, and the same single diagnostic status timer
    before and after. A subsequent clip played through its full two seconds.
    Harness: ignored `.pytest_tmp/video-investigation/cleanup_probe.py`.
  - These checks cover source behavior in the in-app browser, not a rebuilt
    packaged Windows WebView2 app or the reporter's original failure.
  - `git diff --check` and `treaty validate .`: passed.

## 2026-09-10

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
