# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- Establish lightweight `app_src` patch releases on top of the v0.16.6 full
  Windows base with the v0.16.7 sleep-stage color configuration release, then
  continue with the full-path video-association fix.
- Continue the REM-within-Wake statistical-model experiment.
- Complete the remaining author and submission work for the JOSS paper.

## Lightweight Source Releases

- Keep v0.16.6 as the current full Windows base. For compatible `app_src`-only
  changes, tag the tested commit and publish only the automatic source-update
  ZIP, its SHA-256 file, and release notes. Direct new users to the v0.16.6 full
  package, which should patch itself to the latest source release on first
  launch.
- Add one lightweight-release command that runs full pytest, the repository
  Black hook, compile/smoke checks, source-update construction, manifest
  validation, and representative installed-app update tests without rebuilding
  PyInstaller or Torch.
- Make that command refuse the lightweight path when dependencies, models,
  launcher/updater runtime, PyInstaller configuration, packaged layout, or
  unsupported runtime deletions/renames changed. Allow tests, documentation,
  and a validated version-only `setup.py` change alongside `app_src/` updates.
- Preserve jump-ahead support for all live installation states: the original
  v0.16.5 full package, the canonical v0.16.6 full package, and v0.16.5 patched
  to v0.16.6. Store compact installed-baseline hashes and emit every accepted
  previous hash for an existing file so line-ending differences do not reject
  an otherwise untouched installation.
- Use three release fixtures for the first lightweight release: fresh v0.16.5,
  fresh full v0.16.6, and v0.16.5 updated to v0.16.6. Require each to discover,
  apply, and smoke-test the new release successfully.
- After the v0.16.7 color-configuration trial, make the normalized full-path
  MAT-to-video association and collision-proof generated-clip identity the
  next lightweight patch, with regression tests for identical MAT and video
  basenames in different folders.
- Cut a new full base only when the frozen/package boundary changes or when a
  deliberate periodic roll-up is useful.

### Shared Updater Follow-Up

- No frozen runtime change is currently required in
  `desktop_app_source_updater`: the installed updater already accepts multiple
  previous SHA-256 values for one payload file.
- Improve the shared maintainer-side asset builder so it can accept compact
  installed-baseline manifests and multiple valid byte lineages for the same
  installed version directly, instead of leaving that work to an app-specific
  postprocessor. Add coverage for full-package versus source-patched installs
  that report the same version but have different untouched-file line endings.
- Keep any builder-only improvement separable from the frozen runtime pin so an
  application does not need a full-package rebuild merely to adopt safer
  release tooling.

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
