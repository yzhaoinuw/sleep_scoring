# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `../project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- The public Zenodo example is published and verified: dataset v1.0.0,
  DOI `10.5281/zenodo.23489642` (concept DOI `10.5281/zenodo.23489641`).
  All six files match the prepared copies; both READMEs link to the dataset.
  Deliver the documentation update from `dev` through the normal branch handoff.
- Direct Windows downloads and generated release notes are reviewed and on
  `dev`, `publication`, and `main`. See
  `packaging/windows/README.md#release-documentation`. Still to do: tighten
  the existing v0.17.6 GitHub notes from the changelog, retaining its
  calibration limitation. No published release body has been edited yet.
- Annotation save protection is implemented: verified sibling staging before replacement,
  cleanup on success, and a complete local recovery MAT on destination
  failure. Test native saving and overwriting on the affected shared drive;
  v0.17.5 release validation covers synthetic failure paths and a fresh
  frozen installation. Separately handle corrupt MAT selection in the
  load UI, and inspect an intact file reported to have missing scores.
  Actual user-side truncation timing remains unconfirmed.
- Video cleanup shipped in v0.17.4 after automated and maintainer playback
  checks. The original frozen-frame report remains unconfirmed; if it recurs,
  collect the app version, selected time range, and playback/extraction errors
  before attributing it to the corrected cleanup defects.
- Next feature work belongs on `active-quiet-wake`: permanent Active/Quiet Wake
  annotations (keys 5/6) and optional EMG detection after stats-model scoring.
  Keep the experiment separate from the official branches.
- JOSS paper submitted on 2026-10-20 (software version v0.17.6; receipt
  email received). Watch `openjournals/joss-reviews` for the pre-review
  issue and respond to editor/reviewer comments on `publication`.
- Keep the full-path video-association fix for a later app-source-only update
  based on the published v0.17.1 package.
- Continue the REM-within-Wake statistical-model experiment.
- Correct the v0.17.1 release notes, which claim the NIH BRAIN acknowledgment
  reached `CITATION.cff` when CFF 1.2.0 has no funding field.

## Lightweight Source Releases

- Use v0.17.1 as the current full Windows base. For later compatible
  `app_src`-only changes, tag the tested commit and publish only the automatic
  source-update ZIP, its SHA-256 file, and release notes.
- Use `packaging/windows/release_lightweight.ps1` as the standard candidate
  gate. v0.17.6 targets v0.17.1–v0.17.5; the frozen source-update fixture is v0.17.1; retain the v0.17.0
  package only for historic full-package verification because its frozen
  launcher/runtime boundary cannot accept source-only updates.
- Let the builder use schema 1 when `app_src/config.py` is unchanged and schema
  2 when it changes. Keep the approved editable-assignment allowlist aligned
  with the documented user-facing settings.
- Keep v0.17.x automatic assets on the frozen
  `sleep_scoring_app_update_vX.Y.Z.zip` convention. Before the next full base,
  add configurable suffix-style discovery to the shared updater, then switch
  later automatic assets to `sleep_scoring_app_vX.Y.Z_update.zip`.
- Make the normalized full-path MAT-to-video association and collision-proof
  generated-clip identity the next lightweight patch, with regression tests
  for identical MAT and video basenames in different folders.
- While packaging is open, check whether the `_internal/assets` copy produced by
  `app.spec:26` is vestigial. Dash serves `<exe dir>/app_src/assets`, and
  nothing in the tree reads `sys._MEIPASS`, so the bundled copy appears to be
  dead weight that a lightweight update never patches.
- Cut a new full base only when the frozen/package boundary changes or when a
  deliberate periodic roll-up is useful.

## Research Impact Measurement

Status: app-copy totals plus opt-in reporting were released in v0.17.2 on
`main`. The Worker/D1 service is live at
`sleep-scoring-usage-reporting.brainflowzzz.workers.dev`, with receipt-time
summaries and its matching D1 index. The v0.17.2 source-update asset remains
separate from the v0.17.1 full Windows base.

Settled decisions:

- A shared app folder is the reporting unit, not a person or Windows account.
  Its state holds one opaque app-instance ID, local deduplication fingerprints,
  totals, and queued reports. The fingerprints never leave that folder.
- Reporting stays explicitly opt-in. A report contains only opaque app/event
  IDs, a completed-recording delta, scored-second delta, app version, and
  timestamp. Users opt in by setting `ENABLE_USAGE_REPORTING = True` in
  `app_src/config.py`; it is idempotent at the Worker by event ID, so retries
  do not inflate totals.
- "Scored" means a recording saved with every second annotated. It counts once,
  on the first such save, deduplicated by a truncated one-way digest of the EEG
  signal, so reopening, re-saving, or saving under a new name cannot inflate
  the totals.
- The store holds no recording names, paths, signal values, annotations, or
  animal identifiers. Fingerprints never leave the app folder and opaque IDs
  never appear in the exported summary.

Ongoing work:

- The GitHub workflow now has its administrator secret and publishes the
  rounded weekly aggregate to `usage-metrics`. It never publishes app IDs or
  event rows; the README's research-impact badge reads only that public JSON.
- Collect GitHub release-asset download counts when useful. These
  need no instrumentation (`assets[].download_count`), but record the caveats:
  they are not deduplicated, they include bots and mirrors, and they are
  per-asset, so the 2026-07-30 rename of the v0.17.0 full package reset that
  asset's counter.
- Add Zenodo record views and downloads to the metric set now that the
  repository is enabled. These can complement GitHub counts in weekly, daily,
  or other periodic research-impact reviews, and the DOI also accrues citations.
- Decide whether to add distinct-sites and manual-correction-volume metrics.
  Correction volume is the most direct "this tool saved time" number and the
  best answer to the JOSS "who uses this outside your group?" question, but
  every added metric is added surface; keep v1 to what the report prints.
  In particular, `reporting_app_copies` is not a distinct-sites metric: a new
  app folder receives a new opaque ID, even when it belongs to the same lab.
- Revisit shared reporting only if the adopter base outgrows asking directly.
  If that happens, the startup update check is the natural carrier, since it is
  already a once-per-24-hour request with tested throttle and backoff. Note
  that it would convert github.com-only traffic into traffic to a server we
  run, which is the line needing explicit opt-in and a conversation with UR
  research compliance first.

## Statistical Model

- Continue validating the new objective on `dev` and `publication` with
  difficult recordings and mixed-stage examples. The maintainer reports
  smoother running-app behavior; local tests pass and the cached 33-second
  collapse is avoided offline. The 0.25 change penalty and reliance on
  defaults still need broader validation across recordings after v0.17.6.
  Confirm manual labels remain intact after prediction and undo. See the
  2026-10-13 work-log evidence and private `wake33_diagnostics/` comparisons.
- Validate the adaptive statistical model side by side on recordings with
  difficult REM/NE valleys and shifted spectral distributions. Check one- and
  few-example calibrations as well as user-label preservation after another
  model run.
- Improve REM detection when a long Wake bout contains a smaller likely REM
  subsection instead of relabeling the entire Wake bout.
- Compare identifying a low-NE subsection before Wake-to-REM promotion with
  splitting the candidate after the initial REM relabeling.
- Validate side by side in the app on recordings where merged Wake is too broad,
  paying particular attention to REM subsections and post-REM Wake boundaries.
- Keep the current default behavior stable until the alternative is clearly
  better on the targeted recordings.
- Maybe later: keep expert examples confined to their own stage. Wake-only
  labels still change REM (35_app13: tuned Wake threshold 0.70→0.85 grew REM
  788→1,323 s with REM knobs unchanged) because REM is relabelled from Wake
  candidate bouts, so the Wake threshold gates REM. Ideas, preferred first:
  (1) tune only the knobs for stages that were labelled; (2) give REM
  candidates their own fixed threshold so the tuned one only splits
  Wake/NREM (changes unlabelled defaults, so validate against ground truth);
  (3) merge default and tuned runs per labelled stage; (4) apply tuning only
  near the examples (likely not, since examples should generalize). See the
  2026-10-09 work-log entry.

## Citation And Publication

### Zenodo Archive DOI

Latest verified archive: v0.17.6, DOI `10.5281/zenodo.23385811`, confirmed
through the Zenodo records API on 2026-10-15 (see the release work log).

Done as of v0.17.2: the release is archived, concept DOI
`10.5281/zenodo.21748494` is in the README badge and `CITATION.cff`, and the
v0.17.2 version DOI is `10.5281/zenodo.22225514`. Later releases inherit this
setup and need no repository DOI metadata changes, but each published release
still requires Zenodo record verification. Zenodo record 21748495 also carries
the NIH grant `U19NS128613`.

### JOSS Paper (submitted 2026-10-20)

`paper/` is final as of 2026-10-20: the maintainer reviewed the full rendered
PDF, Acknowledgments are final, and no inline TODOs remain. The body is about
1,730 words by the `paper/README.md` rule. Keep the checked-in review PDF
current after any further manuscript, bibliography, or figure edit. Drafting
history is in the work log and its archive.

- Submitted 2026-10-20 as v0.17.6, main subject "Neuroscience: sleep scoring
  of rodent electrophysiology recordings". The editor note disclosed sDREAMER
  as separately published and that the companion packages may be submitted
  separately in the future; no conflicts of interest. Pre-review issue not yet
  open when checked the same day.
- At acceptance, archive the final release on Zenodo, confirm it shows the
  paper's title, and update the `sleepscoring_zenodo` title in `paper.bib`.
- After acceptance, add a `preferred-citation:` block with the JOSS DOI to
  `CITATION.cff`.
- Likely reviewer questions: use outside the BrainFlowZZZ program (name
  external adopters once they exist) and adapting other labs' data (the
  README's Input Files section documents the `.mat` contract).

## Later Ideas

- Make figure height responsive so top-bottom tiled windows do not require
  vertical scrolling without making the four subplots unreadably cramped.
- Consider precomputed downsample tiers only if on-demand resampling becomes a
  bottleneck again.
- Consider an installer and code signing once the zip workflow is routine and
  repeatable.

## Further Down The Line / Just A Thought

- Multi-session support on one computer is low priority. If ever needed, launch
  each app instance on its own free port and isolate cache/temp/video outputs per
  process/session; current user guidance is one app session per computer.
