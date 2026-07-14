# Next Steps

Use this as the forward-looking checklist. Completed work, validation results,
and deferred technical decisions belong in `work_log.md`; architecture belongs
in `project_overview.md` and `dash_app_cookbook.md`.

## Currently Hot

- Continue the REM-within-Wake statistical-model experiment.
- Complete the remaining author and submission work for the JOSS paper.

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
