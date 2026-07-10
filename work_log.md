# Work Log

Prepend new session notes to the top of this file. If you log multiple
sessions on the same calendar date, add a new `###` subsection under the
existing `## YYYY-MM-DD` header instead of starting a second header for the
same date.

Historical verification commands may include absolute paths from the original
development machine. When replaying or adapting them, keep the project folder
name `sleep_scoring` and conda environment name `sleep_scoring_dash3.0`, but
replace the user/home prefix and clone location with the collaborator's local
setup.

Reading note for agents: this file holds at most the 5 most recent unique
calendar dates. Older entries are rotated in chunks of 5 dates into
`work_log_archive/work_log_<earliest>_to_<latest>.md`. Default to reading the
two most recent dated entries; search older entries with targeted terms using
the `^## [0-9]{4}-[0-9]{2}-[0-9]{2}` anchor, or open the relevant archive file
by its date range. See `AGENTS.md` for the full rotation policy.

## 2026-07-10

### Multi-Session PR Opened After Rebase Onto dev (Claude Fable 5, default mode)

- Rebased `feature/multi-session` (5 commits) onto `dev` at `aec3f0f` before
  opening the PR, so the reviewer sees a conflict-free diff. The only
  conflicts were in `work_log.md` (both branches added 2026-07-09/10
  entries); resolved by merging the sessions under shared date headers,
  newest first. `README.md` and `next_steps.md` auto-merged; no code overlap.
- Verification: full pytest on the rebased branch -> `104 passed`; local and
  remote refs confirmed equal after the `--force-with-lease` push.
- Opened PR #8 (`feature/multi-session` -> `dev`):
  https://github.com/yzhaoinuw/sleep_scoring/pull/8.
- Still open before merge: the two remaining manual checks (fourth-launch
  notice, video clips in both windows), tracked in `next_steps.md`.

### Full App Build With Companion Torch Runtime (GPT-5, default mode)

- Ran the revised full Windows packaging flow with `-AllowDirty` because the
  worktree contains the packaging/doc edits made during this packaging fix.
- Suppressed the harmless CPU-only PyTorch warning from sDREAMER inference by
  setting DataLoader `pin_memory` only when the selected device is CUDA in both
  sDREAMER inference paths.
- Renamed the companion runtime artifact to the plain user-facing `torch.zip`
  and restored the default packaged model setting to `stats_model` without
  rerunning PyInstaller: updated source and generated `dist/` config, then
  repacked the existing app folder after removing `_internal\torch`.
- Build output:
  - `release_artifacts/sleep_scoring_app_v0.16.4.post1-windows.zip`
    (202,081,644 bytes), SHA256
    `ECA0B7F20C0EF2339DF560F7671801461E35FD279F4F039AD9E305143FDD2E09`.
  - `release_artifacts/torch.zip`
    (92,097,938 bytes), SHA256
    `07A65EB415C9225C20BE3EDE7196D27A8151779C478B0992B627590A29F0104D`.
  - Full app manifest records
    `optional_torch_runtime = "torch.zip"`.
- Verification:
  - Initial `make_full_app_zip.ps1 -AllowDirty` ran `pip check`, full pytest
    (`84 passed, 1 warning`), release structure smoke, and
    `run_desktop_app.exe --smoke`.
  - After the `pin_memory` warning fix, a rebuild with tests hit a Windows
    permission error while pytest tried to remove the stale
    `.pytest_tmp\build` basetemp; this was a temp-directory cleanup issue, not
    a test assertion failure. Reran full pytest with
    `--basetemp .pytest_tmp\pin_memory`, which passed (`84 passed, 1 warning`).
  - Final artifact rebuild used `make_full_app_zip.ps1 -AllowDirty -SkipTests`
    after the separate clean-basetemp pytest pass; it ran `pip check`, release
    structure smoke, and `run_desktop_app.exe --smoke`.
  - Direct post-build repack verified the final app zip has
    `SLEEP_SCORING_MODEL = "stats_model"` and does not contain
    `_internal\torch`; final `torch.zip` contains root-level `torch/`.
  - Expanded the companion Torch runtime into
    `dist/sleep_scoring_app_v0.16.4.post1/_internal`; confirmed
    `_internal\torch`, `_internal\torch\__init__.py`, and
    `_internal\torch\lib\torch_cpu.dll` exist.
  - Temporary generated-app import probe under the frozen exe imported
    `cProfile`, `torch`, `timm.layers`, `torchvision`, and
    `torch._dynamo.convert_frame` successfully.
  - Temporary generated-app inference probe loaded a 300-second NE slice from
    `user_test_files/408_yfp.mat` and ran packaged NE sDREAMER inference via
    `app_src.inference.run_inference`, producing `299 predictions`; after the
    `pin_memory` fix, the PyTorch CPU/no-accelerator warning no longer
    appeared.
  - Removed temporary probe code from the generated `dist/` app copy and reran
    plain `run_desktop_app.exe --smoke` successfully.
- Remaining follow-up: upload/share both artifacts together. Do not upload the
  earlier manually pip-built runtime zip from 2026-07-09; it is superseded.

## 2026-07-09

### Optional sDREAMER Torch Runtime Builder (GPT-5, default mode)

- Reworked the packaging plan after the generated runtime got past `torchgen`
  but the old app failed later on `ModuleNotFoundError: No module named
  'cProfile'`. That error is baked into the old executable because it was built
  with `torch` excluded from PyInstaller analysis.
- Changed `packaging/windows/app.spec` so Torch is no longer excluded during
  PyInstaller analysis. `packaging/windows/make_full_app_zip.ps1` now creates
  the optional Torch runtime zip from the built `_internal\torch` folder, then
  removes that folder before creating the main app zip.
- Updated `README.md` and `packaging/windows/README.md` to describe the
  runtime as an optional sDREAMER Torch runtime rather than a single `torch/`
  folder. User-facing install check: after copying, `_internal\torch` should
  exist.
- Previously generated
  `release_artifacts/torch_runtime_torch-2.9.1-cpu.zip`
  (120,427,844 bytes) plus `.sha256.txt` and `.manifest.json` sidecars.
  SHA256:
  `294FF4F9340C36BE1C296C9C6C536565A90BDC3FD49782A51EC94F8F1B55E664`.
  Treat this as superseded until a new full app package is built by the revised
  full-package script.
- Verification:
  - Builder installed `torch==2.9.1+cpu` into a clean staging directory.
  - Builder verified `import torch` and `import torchgen` with `python -S`;
    imported paths came from the staging directory.
  - Zip inspection confirmed root-level `torch/` and `torchgen/` are present
    and `_internal/` is not present in the archive.
- Note: a raw normal-Python `sys.path` probe against the existing Desktop
  app's `_internal/` is not a valid frozen-app substitute, because PyInstaller
  stores some NumPy pieces as side data there while the real app also uses its
  PYZ importer.
- Remaining follow-up: build a new full app package with the revised packaging
  flow, copy its companion Torch runtime zip into that new app's `_internal/`,
  and run a real sDREAMER prediction before uploading/sharing artifacts.

### sDREAMER Optional Torch Runtime Import Failure (GPT-5, default mode)

- Investigated a user-reported packaged-app failure after enabling
  `SLEEP_SCORING_MODEL = "sdreamer"` and placing the documented optional
  `torch` folder inside `_internal/`.
- Reproduced evidence from the local test distribution:
  `_internal\torch` exists and is PyTorch `2.9.1+cpu`, but
  `_internal\torchgen` is absent. The callback traceback fails during
  `import torch` at `torch\utils\_python_dispatch.py`, where PyTorch imports
  the top-level `torchgen` package.
- Direct repo testing passes because the conda env has a complete PyTorch
  install on `sys.path` (`torch`, `torchgen`, `functorch`, and required pure
  dependencies), so it is not representative of the slim packaged runtime.
- Current packaging/docs still describe the optional payload as only the
  `torch` folder. The safer fix is to rebuild the optional payload as a
  PyTorch runtime bundle extracted into `_internal/`, at minimum including
  `torch/` and the matching top-level `torchgen/` from the same PyTorch
  install, then smoke-test `import torch` and sDREAMER inference inside a
  fresh app unzip.

### Multi-Session Follow-Ups After User Testing (Claude Fable 5, default mode)

- User validated in real app sessions: two windows side by side (different
  files), the same-file refusal message, crash-recovery salvage in a
  two-window scenario, save/export under multi-session, and a plain
  single-window session. Still open: fourth-launch notice and video clips
  in both windows (tracked in `next_steps.md`).
- Two user-requested tweaks:
  - The refusal message now names the file, e.g. `"a.mat" is already open
    in another Sleep Scoring App window.` (`callbacks/loading.py`).
  - Window `min_size` lowered from (1200, 800) to (800, 500) in
    `config.py` so two windows can tile side by side or top-bottom on a
    1080p screen. Verified safe for the visualization: `make_figure.py`
    pins the figure at `height=800` with a `minHeight: 800px` container
    and only width is responsive (`autosize`), so a small window scrolls
    vertically and squeezes only the x-axis; nothing distorts. A
    height-responsive figure is noted in `next_steps.md` as a possible
    later idea.
- Third tweak from user testing: the mat upload button (`components.py`)
  was `width: 15%` of the window, so resizes rewrapped its label and
  changed its shape. Now `width: fit-content` + `whiteSpace: nowrap` with
  `padding: 6px 16px`, sizing the button to its label at every window
  size. User confirmed the fix in a real app session.
- Verification: full pytest -> `104 passed` (the refusal test now pins the
  filename in the message).
- Session wrap: doc sweep for the feature (`dash_app_cookbook.md` Recipe 1
  slot claim + source map, `project_overview.md` clip paths + test list,
  `AGENTS.md` slot-0 update-check note). Branch state:
  `feature/multi-session` pushed with the plan, implementation, and both
  follow-up commits; PR to `dev` deferred to next session pending the two
  remaining manual checks (fourth-launch notice, video clips in both
  windows).

### Multi-Session Support Implemented (Claude Fable 5, default mode)

- Implemented the full multi-session design on `feature/multi-session`,
  same session as the planning entry below.
- `run_desktop_app.py`: `claim_session_slot()` binds the first free port in
  `BASE_PORT` 8050 - 8052 (`MAX_SESSIONS` 3) and holds the probe socket
  until `run_dash` releases it just before `app.run`; a fourth launch gets
  a webview "too many windows" notice and exit code 1. Slot > 0 skips the
  startup update check and gets a numbered window title, e.g. "(2)".
- Deviation from the planning notes: the base port lives in the launcher
  (`config.PORT` removed), because the slot claim must precede the update
  check, which must precede any `app_src` import. The launcher exports
  `SLEEP_SCORING_INSTANCE_SLOT` / `SLEEP_SCORING_PEER_PORTS`; without them
  `app_src` defaults to slot 0 / no peers, so tests, `--smoke`, and
  old-launcher installs keep single-window behavior and `app_src`-only
  source-update assets stay compatible.
- `app_src` changes: per-slot `TEMP_PATH` / `CACHE_DIR` / `VIDEO_DIR`
  (`slot_<n>`) in `server.py` plus one-time adoption of legacy flat temp
  files into `slot_0` (preserves pre-upgrade salvage) and legacy loose mp4
  cleanup; `GET /_sleep_scoring/current-file` in `routes.py`;
  `find_peer_session_with_file` in `session.py` (0.5 s timeout, ignores
  non-app listeners); `choose_mat` in `callbacks/loading.py` refuses a mat
  file already open in a live peer window; clip URLs in
  `callbacks/video.py` include the slot subdir. Perf logging force-off for
  slot > 0 in `config.py`.
- Docs: README "Multiple Windows" section + per-window crash-recovery
  note; `project_overview.md` entrypoint/module descriptions;
  `next_steps.md` section updated to implemented-state with the remaining
  manual validation checklist.
- Verification:
  - Full pytest -> `104 passed` (84 before; new `tests/test_multi_session.py`
    plus slot-claim tests in `tests/test_run_desktop_app.py`).
  - `python run_desktop_app.py --smoke` -> OK.
  - Headless two-process drive (scratch script): instance 1 claimed slot 0
    / port 8050, instance 2 claimed slot 1 / 8051 with per-slot temp and
    video dirs; `current-file` reported the loaded file on 8050 and empty
    on 8051; a slot-1 process's peer check returned 8050 for the same file
    and None for a different file.
- Not yet done: manual validation in real windows (list in
  `next_steps.md`); next release needs a full app zip (launcher changed).

### Multi-Session Support Planned (Claude Fable 5, default mode)

- PR #7 (`refactor` -> `dev`) merged; user pulled `dev`. Created
  `feature/multi-session` off `dev` for the multi-session work.
- Promoted multi-session support on one computer from "Further Down The
  Line" to an active `next_steps.md` section with the full design, after
  users asked to compare mat files in side-by-side windows. Documentation
  only this session; no implementation yet.
- Design decisions (discussed with user, all agreed):
  - One process per window; no shared server or session-aware callbacks.
  - Port slot (scan base 8050 -> 8052) as the single coordination
    primitive: caps instances at three, provides instance identity,
    per-slot dir namespace (`TEMP_PATH`, `VIDEO_DIR`, cache), auto-update
    guard (slot > 0 skips), and perf-log guard (slot > 0 forces off).
  - Same-file protection via a new `GET /_sleep_scoring/current-file`
    peer endpoint: a second window refuses to load a mat file already
    open in a live peer and shows a message instead. No lock files.
- README updates (salvage-per-slot note, "Multiple windows" note) are
  listed in the plan and deferred until the feature lands, so shipped-app
  docs keep describing shipped behavior.
- Also cleared the completed PR #7 items from `next_steps.md`.

## 2026-07-07

### Restructure PR To dev (Claude Fable 5, default mode)

- User manually validated the Phase 3 clientside interactions in a real app
  session ("ran the app and everything seemed to work"), completing the
  last pre-merge gate.
- Opened PR #7 from `refactor` to `dev` for agent review:
  https://github.com/yzhaoinuw/sleep_scoring/pull/7. It bundles restructure
  Phases 1-3, the jest suite + `js-test` CI job (first CI run happens on
  this PR), and the doc refreshes.
- `next_steps.md` updated: remaining actions are PR review/merge and the
  release-time source-update check.

### Clientside JS Test Harness (Claude Fable 5, default mode)

- Added the first automated coverage for the clientside callback layer:
  `tests/js/` holds a jest suite (`clientsideCallbacks.test.js`, 38 tests)
  that loads `app_src/assets/clientsideCallbacks.js` in Node with stubbed
  `window` / `dash_clientside` globals (`setup.js` provides a `PatchStub`
  that records figure mutations for assertions).
- Coverage highlights: box/click/bout/auto-pan selection rounding, clamping,
  and tie-breaks (including nonzero `start_time` and the null-vs-NaN
  "unscored" bout equivalence from the cache round-trip), `make_annotation`
  half-open range semantics and non-mutation of the figure array,
  `update_sleep_scores` last-three-trace repaint, pan/mode/restyle/message
  callbacks, and the video-button 1-300 s visibility rule.
- Wired a `js-test` job into `.github/workflows/ci.yml` (setup-node 22, npm
  cache keyed on `tests/js/package-lock.json`, `npm ci && npm test`);
  documented the check in `CONTRIBUTING.md` and the suite in
  `project_overview.md`. Added `node_modules/` to `.gitignore`;
  `package-lock.json` is committed for reproducible CI.
- Note: pytest ignores `tests/js/` automatically (`node_modules` is in
  pytest's default norecursedirs and the dir has no Python test files).
- Added `tests/js/README.md` for collaborators new to jest/Node: what jest
  is, why the decision-layer tests need no browser, the `PatchStub` trick,
  the covered/not-covered boundary, and how to run and extend the suite.
  Linked from `CONTRIBUTING.md`; deliberately kept out of the agent-facing
  docs.
- Verification:
  - `npm test` in `tests/js` (Node v26.4.0, jest 30) -> `38 passed`.
  - Full pytest -> `84 passed` (unchanged, 0.7 s).

### app.py Restructure Phase 3 (Claude Fable 5, default mode)

- Moved the 10 inline clientside JS strings out of
  `app_src/callbacks/clientside.py` (720 -> 164 lines) into a new asset,
  `app_src/assets/clientsideCallbacks.js` (588 lines), under the
  `dash_clientside.sleep_scoring` namespace. The Python module now registers
  each callback via `ClientsideFunction(namespace="sleep_scoring",
  function_name=...)` with the Output/Input/State signatures unchanged;
  section headers and callback names mirror each other in both files.
- JS bodies moved verbatim (uniform re-indent only); the extraction script
  round-tripped every transformed block back to the original string to prove
  equality, and checked brace/paren/bracket balance outside strings and
  comments.
- Rationale: real-file JS gets editor syntax highlighting and diffs, and
  stops shipping ~700 lines of JavaScript inside Python strings.
- Docs updated in the same pass: cookbook (browser-layer note, architecture
  diagram, the clientside recipe Source lines, source-file map row),
  `project_overview.md` (callbacks bullet, structure map, assets listing),
  and `next_steps.md` (Phase 3 recorded as landed; remaining actions are
  clientside re-validation, then merge).
- Verification:
  - `app._callback_list` identical to the pre-change baseline in every field
    except the clientside function pointers (now
    `sleep_scoring.<name>` instead of Dash-generated inline namespaces);
    `app._inline_scripts` dropped from 10 to 0.
  - Flask test client: `/assets/clientsideCallbacks.js` -> 200 and the
    served JS defines all 10 registered function names.
  - black `--check app_src/ tests/` -> clean; full pytest -> `84 passed`;
    `python run_desktop_app.py --smoke` -> OK.
  - Not yet browser-validated: the clientside interactions need a manual
    app session (recorded in `next_steps.md`) before merging to `dev`.

### Restructure Docs Refresh (Claude Fable 5, default mode)

- User manually validated the restructured app in a real session ("ran the
  app, everything seemed to work fine"); recorded in `next_steps.md`, where
  the remaining pre-merge action is now just merging `refactor` into `dev`.
- Updated `project_overview.md` for the new `app_src` layout: section 2 now
  describes the app package (`app.py` aggregator, `server.py`, `routes.py`,
  `dialogs.py`, `session.py`, `resampling.py`, `callbacks/`), and the repo
  structure map, tests blurb, video section, active-files list, and reading
  order were repointed accordingly.
- Updated all 17 stale `app.py` source pointers in `dash_app_cookbook.md`
  (architecture diagram, recipe Source lines, and the source-file map table,
  which now lists the six new modules/package individually).
- Checked the other docs: `README.md`, `AGENTS.md`, and `CONTRIBUTING.md`
  only reference `app_src/` generally or `config.py`, so no changes needed.

### app.py Restructure Phase 2 (Claude Fable 5, default mode)

- Split the remaining `app_src/app.py` (1634 lines) into single-concern
  modules on `refactor`: `server.py` (Dash instance, cache, components,
  `TEMP_PATH`/`VIDEO_DIR`, `run_inference` probe), `routes.py` (the two
  Flask endpoints), `session.py` (per-recording setup helpers), and a
  `callbacks/` package with one module per concern (`clientside`,
  `loading`, `navigation`, `prediction`, `saving`, `video`) registered on
  import. `app.py` is now a thin aggregator (14 lines) so
  `from app_src.app import app` keeps working for `run_desktop_app.py`.
- All code blocks moved verbatim with full line accounting; each module
  declares its own imports. The serverside `# ----` section headers became
  the module boundaries; the clientside subsection headers and the
  commented-out `debug_box_select` block moved into
  `callbacks/clientside.py`.
- One deliberate addition: `run_inference = None` in `server.py`'s
  `except ImportError` branch so `callbacks/prediction.py` can import the
  name when ML dependencies are missing (previously the name was unbound;
  unreachable either way because the pred button is disabled then).
- Repointed `tests/test_app_helpers.py` imports and patch targets to the
  new namespaces (`app_src.session.*`, `app_src.callbacks.saving.*`,
  `app_src.callbacks.video.*`, `app_src.resampling.*`).
- Noted in `next_steps.md`: manual pre-merge validation still pending, and
  the next release must confirm a source-update asset cleanly adds the new
  `app_src` modules on an installed build (else ship a full app zip).
- Verification:
  - Sorted `app.callback_map` keys (26) plus the Flask `url_map` are
    byte-identical before and after the split.
  - black 25.12.0 `--check app_src/ tests/` -> clean; full pytest ->
    `84 passed`; `python run_desktop_app.py --smoke` -> OK.
  - Flask test client: `/_sleep_scoring/resample` with no figure -> 404;
    `/_sleep_scoring/profile-log` -> 204.
  - Both inference branches: normal import (`run_inference` available on
    this Mac) and a blocked-import simulation (`run_inference is None`,
    26 callbacks still registered).

### app.py Restructure Phase 1 (Claude Fable 5, default mode)

- Created the `refactor` branch from `dev`, added the phased app.py
  restructure plan to `next_steps.md` (committed as `d5c20bf`), and pushed
  the branch to `origin/refactor`.
- Phase 1 extraction: moved the pywebview file dialogs
  (`open_file_dialog`, `save_file_dialog`) to `app_src/dialogs.py`, and the
  resampler machinery (fig-resampler store, x-bounds/clamp helpers, patch
  compaction/summary, direct-restyle payload builder, relayout-event
  parsing, navigation-profile tracking) to `app_src/resampling.py`.
  `app.py` went from 1953 to 1634 lines.
- All moves are verbatim; the only new code is a
  `latest_navigation_profile_id()` accessor in `resampling.py`, called by
  the two perf-log lines in `update_fig_resampler` that previously read the
  now-moved `NAVIGATION_LATEST_PROFILE_ID` global directly. Verified via
  line accounting that nothing else was added, lost, or altered.
- The two Flask routes stayed in `app.py` because they decorate
  `app.server`; moving them waits for `server.py` in Phase 2 (decision
  recorded in `next_steps.md`).
- `app.py` imports the moved names, so `run_desktop_app.py`
  (`from app_src.app import app`) and the `app_src.app.*` patch targets in
  `tests/test_app_helpers.py` work unchanged.
- Verification:
  - black 25.12.0 `--check app_src/` -> clean (all 16 files).
  - In env `sleep_scoring_dash3.0`: `import app_src.dialogs`,
    `import app_src.resampling`, and
    `from app_src.app import app, build_direct_restyle_payload,
    save_file_dialog, open_file_dialog` -> OK; full pytest -> `84 passed`;
    `python run_desktop_app.py --smoke` ->
    `Sleep Scoring App v0.16.4.post1 smoke check OK`.

### app.py Callback Reorganization (Claude Fable 5, default mode)

- Merged `cookbook` into `dev` (fast-forward `597cda6..3e9e61c`) and pushed;
  local `dev`, `origin/dev`, and the remote head all verified at `3e9e61c`.
- Reorganized the callback sections of `app_src/app.py` on `dev`. Pure block
  moves plus new comment lines only: verified against HEAD that exactly 11
  comment lines were added (10 subsection headers, 1 callback name) and no
  code line was lost or altered.
- Clientside section now ordered by user workflow under subsection headers:
  mode switching and navigation (`switch_mode`, `pan_figure`,
  `apply_direct_restyle`); selection (`read_box_select`, `read_click_select`,
  `read_bout_context_select`, `read_annotation_auto_pan_select`); annotation
  (`make_annotation`, `update_sleep_scores`); message cleanup
  (`clear_display`).
- Serverside section likewise: file loading and visualization (`choose_mat`,
  `create_visualization`, `change_sampling_level`); graph navigation
  (relayout/profile helper functions and `RESAMPLER_CALLBACK_OUTPUT` kept
  directly above `update_fig_resampler`, then
  `log_browser_navigation_profile`); prediction (`show_confirm_pred_modal`,
  `read_mat_pred`, `generate_prediction`); annotation history and saving
  (`update_sleep_scores_history`, `undo_annotation`, `save_annotations`);
  video (`prepare_video`, `choose_video`, `reselect_video`, `make_clip`,
  `show_clip`); debug (the commented-out `debug_box_select` block, moved to
  the end of the file).
- Named the one previously unnamed clientside callback `apply_direct_restyle`
  (applies the serverside resampler patch payload via
  `window.sleepScoringDirectRestyle` and reports status), using the existing
  comment-line naming convention.
- Unified the clientside callback invocation style: the three bare
  `clientside_callback(` calls (`pan_figure`, `apply_direct_restyle`,
  `clear_display`) now use `app.clientside_callback(` like the other seven;
  removed the now-unused `clientside_callback` import from `dash`.
- Rotated the work log: archived the 2026-06-23..2026-07-05 chunk (5 dates)
  to `work_log_archive/work_log_2026-06-23_to_2026-07-05.md`.
- Changes left uncommitted on `dev` for review.
- Verification:
  - `date +%Y-%m-%d` -> `2026-07-07` (macOS/darwin session).
  - black 25.12.0 `--check app_src/app.py` -> clean;
    `python -m py_compile app_src/app.py` -> OK.
  - In env `sleep_scoring_dash3.0`: `import app_src.app` -> OK;
    `python -m pytest --basetemp .pytest_tmp/claude -p no:cacheprovider -q`
    -> `84 passed`; `python run_desktop_app.py --smoke` ->
    `Sleep Scoring App v0.16.4.post1 smoke check OK`.
  - After the invocation unification, re-ran black `--check` (clean), the
    pytest suite (`84 passed`), and the smoke check (OK).
