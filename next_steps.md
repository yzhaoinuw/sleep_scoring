# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- Build and inspect the v0.17 full Windows base with the rate-limit-safe
  updater and stable v0.17 package-folder naming; keep tagging and publication
  separate until the candidate artifact is reviewed.
- Hold additional v0.16 lightweight releases. Keep the full-path
  video-association fix for a later app-source-only update.
- Continue the REM-within-Wake statistical-model experiment.
- Complete the remaining author and submission work for the JOSS paper.

## v0.17 Full Windows Base

Status: candidate scope and metadata are complete; packaging and publication
remain.

- Run the single `release_full.ps1` candidate gate. Do not repeat its
  individual checks unless the candidate commit changes.
- Review the full ZIP, manifest, hashes, packaged smoke checks, and forced
  update check before tagging or publication.
- After the full package is published, export its compact installed baseline,
  switch future lightweight-release fixtures to the v0.17 base, and enable
  schema-2 config merging before the first post-v0.17 lightweight release.

## Lightweight Source Releases

- Keep v0.16.6 as the current full Windows base until v0.17 is published. For
  later compatible `app_src`-only changes, tag the tested commit and publish
  only the automatic source-update ZIP, its SHA-256 file, and release notes.
- Use `packaging/windows/release_lightweight.ps1` as the standard candidate
  gate. Keep its tracked v0.16.5/v0.16.6 installed baselines and three frozen
  fixture paths intact while those distributions remain supported.
- Keep lightweight assets on manifest schema 1 through the v0.17 full-base
  publication. Before the first post-v0.17 lightweight release, enable
  schema-2 configuration merging and refresh the compatible base/fixture
  policy together.
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
