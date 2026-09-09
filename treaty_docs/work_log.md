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
