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

## 2026-08-31

### Add opt-in per-app usage reporting (Codex GPT-5; effort/tokens not reported)

- Corrected the reporting unit from a Windows account to an app copy: a shared
  app folder, including one on an external drive, keeps one opaque ID, local
  total, deduplication set, and upload queue.
- Added disabled-by-default, idempotent aggregate reporting. The app queues an
  enrollment total and later completed-recording deltas without transmitting
  file names, paths, signal data, annotations, animal identifiers, or local
  fingerprints. A missing endpoint keeps the feature inert.
- Deployed the Cloudflare Worker/D1 service at
  `https://sleep-scoring-usage-reporting.brainflowzzz.workers.dev`, with the
  reviewed schema and an administrator secret for authenticated daily or weekly
  aggregate queries. The default app endpoint now targets that service.
- Added Cloudflare Worker rate-limit bindings on the public ingest route: 600
  requests per minute route-wide and 20 requests per minute per opaque app-copy
  ID. This is a free Worker-level guard; account-level WAF rules are an
  Enterprise-only dashboard feature for this account.
- Restored the prior visible app interface: all usage-reporting controls are
  removed from the home screen. The sole opt-in is now
  `ENABLE_USAGE_REPORTING` in `app_src/config.py`; queued reports are sent
  only at the next app launch, never while annotations are saved.
- Verification:
  - Full test suite passed: 202 tests (one pre-existing Flask-Caching
    deprecation warning); source smoke check passed.
  - Rendered Dash check confirmed the export, sharing, and stop-sharing
    controls, and the sharing button opens its explicit confirmation prompt.
  - Cloudflare applied the D1 schema and deployed Worker version
    `d071f327-2e83-4c3d-8247-cb0645589c96`; its health route returned
    `{"status":"ok"}` through the app's Python request path, and an invalid
    POST was rejected with HTTP 400 without creating a usage event.
  - The rate-limit configuration passed Wrangler validation and was deployed in
    Worker version `8f18077c-47a0-46d0-89ec-b5a409829a7f`; an invalid app-like
    POST still returned HTTP 400, confirming the public route remains usable
    while the guard is active.
  - After restoring the original UI, the full test suite passed: 203 tests
    (one pre-existing Flask-Caching deprecation warning); the source smoke
    check and repository-pinned Black hook passed. The locally rendered home
    layout contains the original MAT-file selector and no usage controls.

### Address usage-reporting review findings (Codex GPT-5; effort/tokens not reported)

- Source-only update assets now preserve a user's `ENABLE_USAGE_REPORTING`
  choice, while the reporting endpoint remains release-managed.
- Usage state is no longer replaced after a transient local read failure.
  Pending reports are bounded, permanent client rejections are discarded, and
  overflow totals are coalesced into a later report so one bad event cannot
  block future accounting.
- The checked-in Worker now validates opaque IDs, timestamps, and app-version
  length before D1 writes. Administrative daily and weekly summaries use
  Worker receipt time rather than an app-provided clock. Deployment remains a
  separate post-merge step.
- Removed stale UI/export remnants, clarified Cloudflare request-IP handling,
  and documented that each copied app folder is a separate reporting site.
- Verification:
  - Full Python test suite passed: 208 tests (one Flask-Caching deprecation
    warning).
  - Worker unit tests passed: 3 tests. Wrangler `deploy --dry-run` validated
    the Worker and its D1/rate-limit bindings.

### Address usage-reporting follow-up review (Codex GPT-5; effort/tokens not reported)

- Startup report sync now treats a temporarily unreadable state file as
  unavailable rather than allowing its background thread to emit a traceback.
- The bounded fingerprint cache retains the most recent fingerprints, avoiding
  repeated saves of a recently scored recording inflating totals after the cap.
- Worker summaries use an index-friendly `received_at` range predicate. The
  schema now migrates the obsolete `occurred_at` index to `received_at` when
  applied to the deployed D1 database.
- A batch that drops a permanently invalid event now reports `rejected`, not
  `sent`, while later valid events still proceed.
- Verification:
  - Focused usage-stat tests passed: 24 tests.
  - Worker unit tests passed: 4 tests.
  - Full Python test suite passed: 210 tests (one Flask-Caching deprecation
    warning); the repository-pinned Black hook and source smoke check passed.
  - Wrangler `deploy --dry-run` validated the Worker, D1 binding, and
    rate-limit bindings without deploying the pending migration.

### Merge usage reporting into dev and prepare release notes (Codex GPT-5; effort/tokens not reported)

- Fast-forwarded the validated `feature/usage-tracker` branch into `dev`.
- Added unreleased changelog entries for the disabled-by-default,
  aggregate-only impact reporting and adaptive statistical-model calibration.
- Kept the pending Worker/D1 deployment explicit: the receipt-time index
  migration is checked in but has not yet been applied remotely.
- Verification:
  - `dev` fast-forwarded from `efb827a` through the validated feature head;
    the next-release changelog and delivery notes passed `git diff --check`.

### Add the research-impact badge workflow (Codex GPT-5; effort/tokens not reported)

- Added a weekly GitHub workflow, with a one-time `dev` bootstrap trigger,
  that reads only the authenticated aggregate Worker summary and publishes
  rounded hours and recordings to a public Shields-compatible JSON file on
  `usage-metrics`.
- Added the badge to the README and documented that it is an opted-in impact
  total rather than a people, site, or installation count.
- The workflow requires the repository secret `USAGE_REPORTING_ADMIN_TOKEN` to
  match the deployed Worker's `ADMIN_TOKEN`; no desktop-app update is involved.
- The first live run confirmed authenticated summary retrieval and badge-data
  construction. It stopped only while bootstrapping the orphan metrics branch;
  the follow-up removes an unnecessary `git rm` that finds no tracked files on
  that branch.

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
