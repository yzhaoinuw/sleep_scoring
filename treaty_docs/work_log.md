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

## 2026-08-30

### Rebase and harden the local usage tracker (Codex GPT-5; effort/tokens not reported)

- Rebased `feature/usage-tracker` onto current `dev`, preserving the adaptive
  score-history flow, current update guidance, and the local-only impact
  tracker.
- Usage-stat updates now take an operating-system file lock, so saves from the
  app's supported multiple windows cannot overwrite one another's increment.
  The lock is released automatically when a process exits; accounting failures
  still cannot interrupt a user save.
- Verification:
  - Focused usage-tracker tests, including same-process and cross-process lock
    coverage, passed.
  - The repository-pinned Black hook passed for the changed Python files.
  - `python run_desktop_app.py --smoke` passed.
  - `git diff --check` passed.

## 2026-08-28

### Begin the adaptive statistical-model branch

- Created `feature/adaptive-stats-model` from the clean `dev` checkout.
- Separated sparse, explicitly user-labelled scores from the ordinary displayed
  score history. Existing MAT labels seed both layers; predictions update only
  the displayed layer, while user labels are reapplied on top of later model
  results.
- Added keyboard `0` to clear a selected score back to unscored. Clear and
  prediction operations remain ordinary score-history updates, so Undo restores
  the immediately preceding displayed score state and its matching user layer.
- Added live stats-model calibration from any finite user Wake/NREM/REM labels.
  It evaluates raw predictions before the user overlay, retains the closest
  default configuration on a tie, and reuses one recording feature extraction
  across the candidate search.
- Verification:
  - Focused Python tests passed: `27 passed`.
  - Full Python suite passed: `181 passed` (one Flask-Caching deprecation
    warning).
  - Client-side Jest tests passed: `39 passed`.
  - The repository-pinned Black hook passed for every changed Python file.
  - `python run_desktop_app.py --smoke` passed.
  - Calibration of `35_app13_groundtruth.mat` used 10,300 labels and completed
    in 2.11 seconds.
  - Rendered the local app shell and confirmed the initial file-selection UI
    loads cleanly.

## 2026-08-13

### Adopt the treaty docs-folder layout (Codex GPT-5; effort/tokens not reported)

- The maintainer chose the treaty's folder layout to reduce root-level
  documentation clutter. Keep `AGENTS.md` and `project_overview.md` at the root
  as the agent and human entry points; keep treaty mechanics, active plans, the
  live work log, and its archive under `treaty_docs/`.
- Updated the project from Agent Collab Treaty `v0.6.0` to `v0.9.0`. A normal
  update intentionally preserved the legacy flat layout, so the migration used
  the treaty's explicit `relocate` command and then repaired project-owned links
  outside the managed treaty files.
- Rotated the prior five-date live log intact to
  `work_log_archive/work_log_2026-07-29_to_2026-08-04.md`; this keeps the live
  log within the treaty's five-date limit while preserving all history.
- Verification:
  - `treaty validate .` passed against treaty `v0.9.0`.
  - All relative Markdown links across the moved or directly changed docs
    resolve, and a stale-path scan found no references to the former root-level
    treaty paths.
  - `git ls-files -u` and the conflict-marker scan were empty.
  - `git diff --check` passed.
