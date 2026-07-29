# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- Keep the full-path video-association fix for a later app-source-only update
  based on the published v0.17.0 package.
- Continue the REM-within-Wake statistical-model experiment.
- Complete the remaining author and submission work for the JOSS paper.

## Lightweight Source Releases

- Use v0.17.0 as the current full Windows base. For later compatible
  `app_src`-only changes, tag the tested commit and publish only the automatic
  source-update ZIP, its SHA-256 file, and release notes.
- Use `packaging/windows/release_lightweight.ps1` as the standard candidate
  gate. Keep the tracked v0.17.0 installed baseline and retained full-package
  ZIP available for its frozen-app fixture.
- Let the builder use schema 1 when `app_src/config.py` is unchanged and schema
  2 when it changes. Keep the approved editable-assignment allowlist aligned
  with the documented user-facing settings.
- Make the normalized full-path MAT-to-video association and collision-proof
  generated-clip identity the next lightweight patch, with regression tests
  for identical MAT and video basenames in different folders.
- Cut a new full base only when the frozen/package boundary changes or when a
  deliberate periodic roll-up is useful.

## Research Impact Measurement

Status: planning only; keep separate from the v0.17 updater/package work.

- Define a small annual-report metric set, initially GitHub release-asset
  downloads, private full-package downloads where SharePoint exposes them,
  number of recordings scored, and total recording hours scored.
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

## Publication / JOSS Paper

The draft remains on the `publication` branch in `paper/paper.md` and
`paper/paper.bib`.

- Fill the remaining paper TODOs: co-authors, affiliations and ORCIDs,
  acknowledgments, contributors, and funding/grant numbers.
- Mirror finalized author details into `CITATION.cff`.
- Verify every paper claim against the shipped app and resolve every bibliography
  reference.
- Prepare the JOSS submission and confirm the repository, MIT license, README,
  and supporting documentation meet its requirements.
- After acceptance, add the JOSS DOI as `preferred-citation` in `CITATION.cff`.

## Later Ideas

- Make figure height responsive so top-bottom tiled windows do not require
  vertical scrolling without making the four subplots unreadably cramped.
- Revisit explicit full-bout selection with a right-click/context-menu gesture.
- Consider precomputed downsample tiers only if on-demand resampling becomes a
  bottleneck again.
- Consider an installer and code signing once the zip workflow is routine and
  repeatable.
