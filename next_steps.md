# Next Steps

Use this as the forward-looking checklist. Completed experiments and measured
outcomes live in `work_log.md`.

## Currently Hot

- `app.py` restructure on the `refactor` branch: Phases 1-2 landed and were
  manually validated in a user-run app session, all 2026-07-07 (see "app.py
  Restructure" below). Next action: merge `refactor` into `dev`. Phase 3
  (JS to assets) stays optional.
- Auto-update packaging: after the next `app_src`-only change, publish a source
  update asset and verify an installed `v0.16.4.post1` app updates itself.
- No active visualization performance experiment is planned before the next
  shipped build.

## app.py Restructure

Goal: shrink `app_src/app.py` into single-concern modules so a session can
read and diff only the part it touches. Pure rehoming, no behavior changes.
All landed 2026-07-07: callback ordering/naming tidy-up on `dev`; Phase 1
(`dialogs.py`, `resampling.py`) and Phase 2 (layout below) on `refactor`.

Layout after Phase 2:

- `app_src/server.py`: Dash instance, cache, components, `TEMP_PATH` /
  `VIDEO_DIR`, and the `run_inference` availability probe. Parameterize
  here (cache dir, paths, port) for the multi-session idea under "Further
  Down The Line".
- `app_src/routes.py`: the two Flask endpoints.
- `app_src/session.py`: per-recording setup helpers (cache init, temp-dir
  housekeeping, metadata, figure creation).
- `app_src/callbacks/`: one module per concern (`clientside`, `loading`,
  `navigation`, `prediction`, `saving`, `video`), registered on import.
- `app_src/app.py`: thin aggregator; `from app_src.app import app` still
  works for `run_desktop_app.py`.
- Tests patch the new namespaces (e.g. `app_src.callbacks.saving.loadmat`,
  `app_src.session.TEMP_PATH`).

Remaining:

- Merge `refactor` into `dev` (manual validation done in a user-run app
  session on 2026-07-07).
- Before the next release, confirm a source-update asset cleanly adds the
  new `app_src` modules (`server.py`, `routes.py`, `session.py`,
  `callbacks/`) on top of an installed build; ship a full app zip if it
  does not.

Phase 3 (optional, separate decision, not tidy-up):

- Move the clientside JS strings into real files under `app_src/assets/`
  wired via `ClientsideFunction`, for JS syntax highlighting and linting.
  Treat as its own verified change, not part of the rehoming.

Verification gate (run for Phases 1-2; rerun for Phase 3 if picked up):

- black `--check`, full pytest, and `run_desktop_app.py --smoke` pass.
- Manual app launch touching navigation, selection, annotation, save, and
  video flows before merging back to `dev`.

## Installation Packaging

Open items:

- After the next app_src-only change, attach a generated automatic source
  update asset to a GitHub Release and verify an installed `v0.16.4.post1`
  app updates itself on startup.
- Manually test `unblock_app.cmd` from a fresh unzip of a generated
  full app zip.
- Keep the manual `app_src` replacement zip only as a fallback if automatic
  update testing exposes a compatibility issue.

Possible later upgrades:

- Consider an installer and code signing only after the zip workflow is boring
  and repeatable.

## Visualization Performance

Current status:

- Navigation now feels smooth across keyboard, mouse drag, wheel zoom, and reset.
- Server-side resampler work is no longer the main bottleneck.
- Final refresh payloads are compacted, usually around `95-110 KB`.
- Direct browser-side `Plotly.restyle` is the active final refresh path.
- Remaining cost is mostly Plotly/WebGL redraw time.
- Perf logging is off by default for shipped users; env-var overrides remain
  available for opt-in profiling.
- Mac M4 baseline captured on 2026-05-25 (see
  `ui_response_time_optimization_progress.txt`).

Active experiments:

- None before the next shipped build. The latest low-hanging UI optimization
  probes have been resolved; see "Do not revisit for now" below for each
  outcome.

Measurement protocol (for any future probe):

- Use the 2026-05-25 Mac M4 baseline in
  `ui_response_time_optimization_progress.txt` as the anchor.
- Each item lands as its own commit so per-item before/after numbers stay
  attributable.

Possible later ideas:

- Consider precomputed downsample tiers only if on-demand resampling
  becomes a bottleneck again.

Do not revisit for now:

- Adaptive final refresh density by visible window width.
- Fast server trace updates during active movement.
- Visualization-only source downsampling to 128 Hz.
- Wrapping `x`/`y` as `Float32Array` before `Plotly.restyle`. Probed on
  2026-05-26 with no measurable change in `dash_apply` or auto-pan
  `apply` vs the baseline; Plotly already converts internally.
- Switching `hovermode` from `"x unified"` to `"x"`. Probed on
  2026-05-26. Apply time dropped by a real but modest ~5-10 ms across
  gesture types, but this drops the cross-subplot synchronized spike
  line and combined tooltip. Multiple users specifically requested
  the unified-crosshair behavior, so the UX trade-off is not worth
  the speedup. Do not revisit unless we find a way to keep the
  unified visual.
- Swapping `Plotly.restyle` for `Plotly.react` (with bumped
  `layout.datarevision`) in `graphDirectRestyle.js`. Probed on
  2026-05-26 and clearly regressed: `dash_apply` jumped by ~100-150 ms
  across native release, keyboard, and custom-drag gestures vs the
  baseline. Cause is that `react` re-diffs the full figure
  (spectrogram heatmap, sleep-score Heatmap, legend, layout) on every
  call, while `restyle` patches only the named props on the named
  trace indices. Auto-pan was unaffected because it lives in
  `annotationAutoPan.js` and has its own merge path.

## Annotation Selection

Current status:

- Edge-triggered x-axis panning during annotation drag selection is implemented in
  `app_src/assets/annotationAutoPan.js`.
- The final selected `[start, end]` range is preserved for the existing annotation flow.
- Auto-pan direct trace refreshes use `/_sleep_scoring/resample` and browser-side
  `Plotly.restyle` so live selection does not compete with normal Dash graph updates.
- Browser-side merge buffers are capped so long auto-pan drags stay around `7k-8k`
  active points instead of growing without bound.
- This work still needs manual validation on top of the optimized
  `codex/next-level-navigation` stack after the branch integration.

Pre-ship validation:

- Re-test normal zooming, keyboard panning, mouse-drag panning, and reset.
- Re-test annotation click selection, normal drag selection, and drag-select auto-pan.
- Re-test mode switches and sampling levels `x0.5`, `x1`, `x2`, and `x4`.
- Watch for stale-trace snapback, lost final detail refresh, or annotation selection drift.

Guardrails:

- Keep normal click thin-box selection and drag-box selection intact.
- Keep the nonzero-`start_time` click-selection fix intact.
- Do not start with double-click or modifier-click gesture inference.

Possible later experiment:

- Revisit explicit full-bout selection with a right-click/context-menu style gesture.

## Statistical Model

Current status:

- `app_src/run_inference_stats_model.py` is the active app-side stats model.
- Broader experiment scripts are local-only and ignored under `scripts/`, not
  part of the shipped app source.
- Immediate goal: improve REM detection inside long Wake bouts.

Next experiment:

- Allow the REM detector to carve out a likely REM subsection from within a Wake bout
  instead of relabeling the entire Wake bout.
- Compare:
  - identifying a low-NE subsection before Wake-to-REM promotion
  - splitting a Wake-derived REM candidate after initial REM relabeling
- Focus on files where merged Wake currently swallows a smaller likely REM region.

Validation:

- Use side-by-side visual inspection in the app first.
- Prioritize Wake bouts containing likely REM subsections, post-REM Wake boundary
  placement, and files where merged Wake is too broad.

Guardrails:

- Do not destabilize current defaults while improving REM-in-Wake detection.

## Publication / JOSS Paper

The JOSS paper draft lives on the `publication` branch in `paper/paper.md`
(summary, statement of need, key features, implementation) and
`paper/paper.bib`. A `CITATION.cff` at the repo root makes the software citable
now, via GitHub's "Cite this repository" button, before the paper is published.

Done:

- Draft `paper/paper.md` and `paper/paper.bib` written.
- `CITATION.cff` added so users can cite the software immediately.
- `README.md` has a `Citation` section pointing to the "Cite this repository"
  button.
- Lead author ORCID (`0000-0002-0819-5012`) filled in `paper/paper.md` and
  `CITATION.cff`.

Open items:

- Fill the remaining `paper.md` TODOs: co-authors, affiliations (with their
  ORCIDs), and the Acknowledgments (PI, data/model contributors, funding/grant
  numbers).
- Mirror any co-author details into `CITATION.cff`.
- Verify every claim in the paper against the current shipped app and check that
  each `paper.bib` reference resolves.
- Set up the JOSS submission (fork of the `joss-reviews` process): confirm the
  repo is public, has an OSI license (MIT, present), and a clear README/docs.
- After acceptance, add a `preferred-citation:` block with the JOSS DOI to
  `CITATION.cff`.

## Further Down The Line / Just A Thought

- Multi-session support on one computer is low priority. If ever needed, launch
  each app instance on its own free port and isolate cache/temp/video outputs per
  process/session; current user guidance is one app session per computer.
  When picked up, implement the isolation in `app_src/server.py` (created in
  app.py Restructure Phase 2): it owns the Dash instance, the `Cache` dir,
  `TEMP_PATH`, and `VIDEO_DIR`, so per-instance dirs and port selection are
  one-place changes. Module globals such as the fig-resampler store are
  per-process and need no change in the one-process-per-window design.
