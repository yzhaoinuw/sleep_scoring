# Next Steps

Use this as the forward-looking checklist. Completed experiments and measured
outcomes live in `work_log.md`.

## Currently Hot

- Multi-session support on one computer: implemented 2026-07-09 on
  `feature/multi-session` (branched from `dev` after PR #7 merged), with
  tests and doc updates. See "Multi-Session Support" below for design and
  the remaining manual validation before merge. The next release after this
  merges needs a full app zip (the launcher changed).
- Auto-update packaging: after the next `app_src`-only change, publish a source
  update asset and verify an installed `v0.16.4.post1` app updates itself.
- No active visualization performance experiment is planned before the next
  shipped build.

## app.py Restructure

Goal: shrink `app_src/app.py` into single-concern modules so a session can
read and diff only the part it touches. Pure rehoming, no behavior changes.
All landed 2026-07-07: callback ordering/naming tidy-up on `dev`; Phase 1
(`dialogs.py`, `resampling.py`), Phase 2 (layout below), and Phase 3 (the
clientside JS strings moved to `app_src/assets/clientsideCallbacks.js`,
registered via `ClientsideFunction`) on `refactor`.

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
- `app_src/assets/clientsideCallbacks.js`: the clientside callback JS, in
  the `dash_clientside.sleep_scoring` namespace; names and sections mirror
  `callbacks/clientside.py`.
- `app_src/app.py`: thin aggregator; `from app_src.app import app` still
  works for `run_desktop_app.py`.
- Tests patch the new namespaces (e.g. `app_src.callbacks.saving.loadmat`,
  `app_src.session.TEMP_PATH`).

Remaining:

- Before the next release, confirm a source-update asset cleanly adds the
  new `app_src` files (`server.py`, `routes.py`, `session.py`,
  `callbacks/`, `assets/clientsideCallbacks.js`) on top of an installed
  build; ship a full app zip if it does not.

PR #7 (`refactor` -> `dev`) merged 2026-07-09 after agent review.

## Multi-Session Support (One Computer)

Goal: let a user open up to three app windows side by side to compare
visualizations of different mat files. Requested by users; planned and
implemented 2026-07-09 on `feature/multi-session`.

Design: one process per window. Each launch of `run_desktop_app.py` is a
fully independent instance with its own Dash server, port, and pywebview
window. No shared server, no session-aware callbacks: module globals (the
fig-resampler store, `cache`, `components`) stay per-process, so callback
code does not change. Cross-window coordination uses the OS, not shared
Python state.

Key mechanism, the port slot:

- Fixed port range: `BASE_PORT` (8050) through 8052, i.e. at most three
  instances. At startup, scan 8050 -> 8052 and take the first free port; if
  all three are bound, show a message and exit. The kernel's port table is
  the instance counter and cannot go stale (a dead process releases its
  port).
- Slot number = port - base. One integer provides instance identity,
  directory namespace, updater guard, perf-log guard, and peer address.
- `BASE_PORT` / `MAX_SESSIONS` live in `run_desktop_app.py`, not
  `config.py` (the old `config.PORT` is gone): the slot claim must happen
  before the update check, which must happen before any `app_src` import.
  The launcher passes the slot and peer ports to the app via
  `SLEEP_SCORING_INSTANCE_SLOT` / `SLEEP_SCORING_PEER_PORTS`; absent env
  vars (tests, scripts, `--smoke`, or an old launcher running updated
  `app_src`) default to slot 0 with no peers, i.e. today's single-window
  behavior. So `app_src`-only source-update assets stay compatible with
  installed builds; multi-window activates once the new launcher ships in
  a full app zip.

Implemented 2026-07-09 (all landed with tests on `feature/multi-session`):

- `run_desktop_app.py`: `claim_session_slot()` scans/claims the slot in
  `main()` before anything else, holds the probe socket during startup and
  releases it just before `app.run` to shrink the claim/bind race. Slot > 0
  skips the startup auto-update check (prevents patching `app_src/` under a
  running instance and concurrent update applies). A fourth launch shows a
  "too many windows" webview notice and exits. Windows on slot > 0 get a
  numbered title, e.g. "(2)".
- `app_src/server.py`: per-slot dirs. `TEMP_PATH` becomes
  `.../sleep_scoring_app_data/slot_<n>`, `VIDEO_DIR` becomes
  `assets/videos/slot_<n>`; the flask-caching `CACHE_DIR` follows
  `TEMP_PATH`. `clear_temp_dir` and the mp4 purge then self-heal since they
  only iterate their own dir. Slot dirs are reused across runs, so there is
  no orphan-dir accumulation to clean up. On the first slot-0 launch after
  the upgrade, loose files in the old flat temp dir move into `slot_0`
  (`adopt_legacy_temp_files`) so pre-upgrade unsaved-annotation salvage
  survives, and stale pre-upgrade mp4s in the videos root are removed.
- `app_src/callbacks/video.py`: clip URL becomes
  `/assets/videos/slot_<n>/<clip>` (still served by Dash's assets route).
- `app_src/routes.py`: add `GET /_sleep_scoring/current-file` returning
  `cache.get("filepath")`.
- `app_src/callbacks/loading.py`: before `initialize_cache`, query the
  other slots' `current-file` endpoints (short timeout, at most two peers).
  If a live peer has the same mat file open, refuse the load and show a
  message telling the user that file is already open in another window and
  to pick a different one. No lock files: a crashed window stops answering
  its port, so its claim evaporates.
- Perf logging: slots > 0 force the perf flags off (env override read by
  `app_src/config.py`, matching the existing `SLEEP_SCORING_*` pattern).
  This silences both server-side prints and browser log mirroring, since
  `routes.py` already returns 204 when profiling is off.
- Tests: `TEMP_PATH` stayed a module global (existing tests patch
  `app_src.session.TEMP_PATH`). `tests/test_multi_session.py` covers the
  config env contract, legacy temp adoption, the current-file endpoint,
  the peer same-file check, the load refusal, and the slot clip URL;
  slot scanning/exhaustion tests live in `tests/test_run_desktop_app.py`.

Docs updated 2026-07-09: `README.md` gained a "Multiple Windows" section
and a per-window crash-recovery note (salvage follows window order; a
wrong order degrades to no recovery, not wrong recovery);
`project_overview.md` reflects the slot claim and per-slot dirs.

User-validated 2026-07-09: two windows side by side (different files), the
same-file refusal message, crash-recovery salvage in a two-window scenario,
save/export under multi-session, and a plain single-window session.
Follow-ups from that session landed the same day: the refusal message now
names the file, the window `min_size` dropped to (800, 500) so two windows
can tile side by side or top-bottom on a 1080p screen (safe: the figure is
fixed at 800 px height and only width-responsive, so small windows scroll
rather than distort), and the mat upload button sizes to its label
(`fit-content` + `nowrap`) instead of 15% of the window so its shape
survives resizes.

Remaining before merge/release:

- Manual validation still open: a fourth-launch "too many windows" notice
  and video clips in both windows.
- The next release must ship as a full app zip: `run_desktop_app.py`
  changed, and source-update assets only deliver `app_src/`. An
  `app_src`-only asset on top of an old launcher stays safe (defaults to
  single-window behavior) but does not enable multi-window.

Possible later idea:

- Make the figure height adapt to the window height so a top-bottom tiled
  window shows the whole figure without vertical scrolling. Today
  `make_figure.py` pins `height=800` and the graph container sets
  `minHeight: 800px`; only width is responsive. Risk: four stacked
  subplots get cramped in a ~500 px window.

Accepted compromises (documented, not engineered against):

- Memory/CPU scale per window, bounded at 3x by the slot cap.
- Salvage semantics depend on window launch order (see README note above).
- Recent-files/video records are per-window, not shared.
- Two windows loading the same file in the same instant can slip past the
  peer check; not worth engineering against for a human-paced desktop app.

## Installation Packaging

Open items:

- After the next app_src-only change, attach a generated automatic source
  update asset to a GitHub Release and verify an installed `v0.16.4.post1`
  app updates itself on startup.
- Upload/share the rebuilt `v0.16.4.post1` full app zip together with its
  companion `torch.zip`.
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

- (Multi-session support on one computer was promoted to an active section
  above on 2026-07-09 after user requests for side-by-side comparison.)
