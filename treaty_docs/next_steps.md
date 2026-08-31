# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `../project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

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
  gate. Keep the tracked v0.17.0 and v0.17.1 installed baselines and retained
  full-package ZIPs available for their frozen-app fixtures.
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

Status: app-copy totals plus opt-in reporting are implemented on
`feature/usage-tracker`. The Worker/D1 service is deployed at
`sleep-scoring-usage-reporting.brainflowzzz.workers.dev`; keep the app source
update separate from the v0.17 updater/package work.

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

Remaining work:

- No backend deployment step remains before a broad public rollout. The public
  ingest route has route-wide and per-app-copy Cloudflare Worker rate limits;
  no app data is sent until a user opts in.
- Collect GitHub release-asset download counts when useful. These
  need no instrumentation (`assets[].download_count`), but record the caveats:
  they are not deduplicated, they include bots and mirrors, and they are
  per-asset, so the 2026-07-30 rename of the v0.17.0 full package reset that
  asset's counter.
- Add Zenodo record views and downloads to the metric set now that the
  repository is enabled. For an annual research-impact report these are a
  stronger number than GitHub's, and the DOI also accrues citations.
- Decide whether to add distinct-sites and manual-correction-volume metrics.
  Correction volume is the most direct "this tool saved time" number and the
  best answer to the JOSS "who uses this outside your group?" question, but
  every added metric is added surface; keep v1 to what the report prints.
- Revisit shared reporting only if the adopter base outgrows asking directly.
  If that happens, the startup update check is the natural carrier, since it is
  already a once-per-24-hour request with tested throttle and backoff. Note
  that it would convert github.com-only traffic into traffic to a server we
  run, which is the line needing explicit opt-in and a conversation with UR
  research compliance first.

## Statistical Model

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

## Citation And Publication

### Zenodo Archive DOI

Done as of v0.17.1: the release is archived, concept DOI
`10.5281/zenodo.21748494` is in the README badge and `CITATION.cff`, and the
v0.17.1 version DOI is `10.5281/zenodo.21748495`. Later releases inherit this
setup and need no repository DOI metadata changes, but each published release
still requires Zenodo record verification. Zenodo record 21748495 also carries
the NIH grant `U19NS128613`.

### JOSS Paper (under construction, deferred)

The draft in `paper/` is not submission-ready and is not being actively worked.
See `paper/README.md` for its status. Open items:

- Fill the remaining `paper.md` TODOs: co-authors, affiliations (with their
  ORCIDs), and the Acknowledgments (PI, data/model contributors, funding/grant
  numbers).
- Verify every claim in the paper against the current shipped app.
- Strengthen the "useful beyond the BrainFlowZZZ program" angle for JOSS
  reviewers: name one or two external adopters (in Acknowledgments or a
  short Statement-of-need sentence) once such use exists, even at sandbox
  scale. JOSS reviewers commonly ask "who outside the author's group uses
  this?" and a concrete answer is the easiest mitigation.
- Add an "Adapting input data" subsection to `README.md` documenting the
  `.mat` field contract precisely enough for a non-BrainFlowZZZ lab to write
  a thin (~50-line) converter. This neutralizes the most likely
  "internal infrastructure" objection without forcing the ingestion code
  itself to be generalized.
- Set up the JOSS submission (fork of the `joss-reviews` process): confirm the
  repo is public, has an OSI license (MIT, present), and a clear README/docs.
- After acceptance, add a `preferred-citation:` block with the JOSS DOI to
  `CITATION.cff`.

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
