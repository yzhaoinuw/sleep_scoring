# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- Keep the full-path video-association fix for a later app-source-only update
  based on the published v0.17.1 package.
- Continue the REM-within-Wake statistical-model experiment.
- Complete the remaining author and submission work for the JOSS paper.
- Optionally add the NIH BRAIN award (U19NS128613) to Zenodo record 21748495
  through Zenodo's edit UI, and correct the v0.17.1 release notes, which claim
  the acknowledgment reached `CITATION.cff` when CFF 1.2.0 has no funding field.

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
- Cut a new full base only when the frozen/package boundary changes or when a
  deliberate periodic roll-up is useful.

## Research Impact Measurement

Status: planning only; keep separate from the v0.17 updater/package work.

- Define a small annual-report metric set, initially GitHub release-asset
  downloads, number of recordings scored, and total recording hours scored.
- Decide whether app-use totals should remain local for a user-exported impact
  summary or be sent through an explicit opt-in reporting mechanism.
- Collect only aggregate counts. Do not collect recording names, paths, signal
  data, annotations, animal identifiers, or other research data.
- Define what counts as "scored" and prevent repeated saves or reopening the
  same recording from inflating the totals.
- Choose where any shared aggregates would be received, retained, and exported
  for the group's annual research impact report before adding instrumentation.

## Statistical Model

- Improve REM detection when a long Wake bout contains a smaller likely REM
  subsection instead of relabeling the entire Wake bout.
- Compare identifying a low-NE subsection before Wake-to-REM promotion with
  splitting the candidate after the initial REM relabeling.
- Validate side by side in the app on recordings where merged Wake is too broad,
  paying particular attention to REM subsections and post-REM Wake boundaries.
- Keep the current default behavior stable until the alternative is clearly
  better on the targeted recordings.

## Citation And Publication

### Zenodo Archive DOI (current priority)

`CITATION.cff` is ready for v0.17.1: it carries an abstract and cites the
upstream sDREAMER model. Remaining steps, in order:

- Confirm with the PI and UR Ventures that the adapted `models/sdreamer/` code
  may be redistributed under MIT. `NOTICE` now carries the confirmed NIH grant
  number; its redistribution statement is the remaining assertion to confirm.
- Publish the prepared v0.17.1 full release now that the repository is enabled
  in Zenodo, then verify that Zenodo creates the expected version record.
- Put the resulting concept DOI (not the version DOI) in the README badge and
  in `CITATION.cff`.

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
- Revisit explicit full-bout selection with a right-click/context-menu gesture.
- Consider precomputed downsample tiers only if on-demand resampling becomes a
  bottleneck again.
- Consider an installer and code signing once the zip workflow is routine and
  repeatable.

## Further Down The Line / Just A Thought

- Multi-session support on one computer is low priority. If ever needed, launch
  each app instance on its own free port and isolate cache/temp/video outputs per
  process/session; current user guidance is one app session per computer.
