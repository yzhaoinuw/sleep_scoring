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

## 2026-07-14

### v0.16.6 Crash Recovery Release Preparation (GPT-5, default mode)

- Changed crash recovery to preserve unsaved annotations only when the cached
  and newly selected recordings normalize to the same absolute path. A missing
  legacy path, malformed cached path, or same-named file in another folder now
  resets recovery safely.
- Added regression coverage for normalized same-path preservation, same-basename
  different-folder reset, and older filename-only caches.
- Set runtime and package metadata to `v0.16.6` and updated the user-facing
  crash-recovery note.
- Built the automatic source-update asset from compatible baseline `v0.16.5`.
  Its payload contains only `app_src/__init__.py` and `app_src/session.py`.
- A pre-publication test from a fresh corrected-v0.16.5 extraction caught that
  Git-blob hashes used LF bytes while the released Windows ZIP contained CRLF
  copies. The updater correctly refused that first asset as a possible local
  edit instead of overwriting the files.
- Added a release helper that replaces previous-version hashes with the exact
  bytes from the released full ZIP. Full packages now export `app_src/` from
  the release commit's Git archive, keeping new installs and future update
  manifests byte-aligned across platforms.
- Rebuilt the local update asset against the official v0.16.5 ZIP. The same
  fresh extraction then printed `updated to v0.16.6 (2 changed files)`, and its
  frozen executable passed the v0.16.6 smoke check.
- Pre-release verification:
  - Focused recovery and multi-session tests: `39 passed, 1 warning`.
  - Full pytest and the initial source-update build gate: `115 passed, 1
    warning`; after adding packaging-contract regression tests, full pytest:
    `117 passed, 1 warning`.
  - Repository-pinned Black hook, `git diff --check`, and
    `python run_desktop_app.py --smoke`: passed.

### v0.16.5 Startup Updater Reissue (GPT-5, default mode)

- Revoked the initially published v0.16.5 GitHub Release after a freshly
  extracted package reported HTTP 415 during its startup update check. All
  release assets still showed zero GitHub downloads; the tag was retained
  while the replacement package was prepared.
- Traced the failure to the shared updater sending the binary asset media type
  to GitHub's JSON release-metadata endpoint. Fixed and regression-tested the
  shared package, then pushed `desktop_app_source_updater` commit `f2f79a8` to
  its `main` branch.
- Pinned `sleep_scoring` to the fixed updater commit and added
  `run_desktop_app.exe --check-update` as a full-package release gate. The
  command exits nonzero when metadata retrieval or updater execution fails, so
  the Windows build cannot silently ship the same failure again.
- Trial replacement-package verification:
  - Full pytest: `113 passed, 1 warning` (the existing Flask-Caching
    deprecation warning).
  - Packaged structure smoke and `run_desktop_app.exe --smoke`: passed.
  - Frozen executable online check contacted the real GitHub Release endpoint
    and printed `[startup-update] no update available`.
  - The replacement release remained unpublished pending a final clean build,
    corrected tag/ref delivery, and remote asset verification.
- Final clean reissue verification and delivery:
  - Rebuilt from clean commit `9e8903c`; the package manifest records that
    commit and pins updater commit `f2f79a8`.
  - Full pytest again passed `113 passed, 1 warning`; packaged structure smoke,
    packaged import smoke, and the frozen online update check all passed.
  - Extracted the final ZIP into a new directory and reran both executable
    checks. After publication, the extracted executable queried the new public
    v0.16.5 release and printed `[startup-update] no update available`.
  - Published seven assets only after every GitHub SHA-256 digest matched its
    local file. Main ZIP SHA-256:
    `730BF52A547C19B4FB69CCD361A1A76630913D7C4002A38E62C457DA0F25FAF8`.
  - Corrected tag `v0.16.5` points to `9e8903c`; `dev` and `main` received that
    release commit before publication. This log-only follow-up records the
    final remote state without moving the release tag.

### v0.16.5 Release Preparation (GPT-5, default mode)

- Set the next official version to `v0.16.5` in the runtime and package
  metadata.
- Kept the changelog focused on changes users will notice: multiple windows,
  per-window recovery/video behavior, tiled-window usability, automatic startup
  updates, and the optional sDREAMER Torch runtime.
- User-validated video creation, playback, and association in two simultaneous
  windows. The fourth-window refusal check was already validated.
- Deferred normalized full-path crash recovery to the next compatible
  `app_src` update, where it will also exercise the installed-user automatic
  update flow from `v0.16.5`.
- Verification:
  - Full pytest: `111 passed, 1 warning` (the existing Flask-Caching
    deprecation warning).
  - Repository-pinned Black hook: passed across all tracked Python files.
  - `python run_desktop_app.py --smoke`: `Sleep Scoring App v0.16.5 smoke
    check OK`.
  - `python -m compileall -q app_src run_desktop_app.py` and
    `git diff --check`: passed.
  - Manual multi-session checks: two-window video behavior and the fourth-window
    refusal notice passed.

## 2026-07-13

### Multi-Session Recovery Docs And Roadmap Cleanup (GPT-5, default mode)

- User-validated that a fourth launch is refused with the expected
  close-one-window notice.
- Updated the README crash-recovery note with the multi-window launch-order,
  title-number, and wrong-file reset behavior, including the current same-name
  caveat.
- Rewrote `next_steps.md` as a focused forward-looking checklist. Removed
  completed implementation history and performance postmortems, consolidated
  the remaining release work, and retained the active statistical-model,
  publication, and later-idea threads.

- Verification:
  - `git diff --check` passed; `next_steps.md` was reduced from 342 lines to
    73.
  - The repo-local `treaty.exe validate .` accepted this session's structure;
    it still reports older live work-log entries that predate the current
    verification-subsection contract.
  - No code tests were run for this documentation-only update.

## 2026-07-10

### Cookbook Multi-Session Recipe (GPT-5, default mode)

- Added `dash_app_cookbook.md` Recipe 19 as the authoritative explanation of
  the one-process-per-window design: slot claiming, the pre-import environment
  contract, atomic startup-update guards, per-slot cache/temp/video paths,
  process-local current-file state, peer duplicate-file refusal, legacy-cache
  adoption, recovery semantics, and the accepted concurrency trade-offs.
- Tightened Recipes 1, 3, 4, and 17 into cross-references and refreshed the
  recipe index, adaptation checklist, gotcha catalog, and source-file map so
  the multi-session contract is documented once without losing local context.
- Verification: checked recipe headings/anchors and stale multi-session wording;
  `git diff --check` passed.

### PR #8 Reviewed And Merged To dev (GPT-5, default mode)

- Completed a multi-round diff-based review of PR #8
  (`feature/multi-session` -> `dev`). The review found and verified fixes for
  the updater-under-live-peer race, stale cached `current-file` reports,
  bound-but-not-listening peer detection, port-test isolation, and the final
  updater time-of-check/time-of-use gap.
- Fast-forwarded local `dev` from `aec3f0f` to reviewed PR head `05b66b7`,
  preserved the existing local `AGENTS.md`/`work_log.md` edits, reconciled
  `next_steps.md`, and included the documentation reconciliation in the same
  publish workflow.
- Verification on the exact PR head: GitHub CI green; full pytest ->
  `111 passed`; `run_desktop_app.py --smoke` and `compileall` passed. Direct
  guard reproduction: two peer ports held, a late launcher received no slot,
  and slot 1 became available immediately after guard release.
- Corrected the final author work-log subsection from `2026-07-11` to the
  workstation date `2026-07-10`; remote `dev` and PR closure were verified
  after the push.

### PR #8 Review Round 3: Update Guard Holds Peer Ports (Claude Fable 5, default mode)

- Reviewer confirmed the round-2 fixes but flagged a remaining P1
  time-of-check/time-of-use race: the bind probe released each peer port
  before the updater ran, so a launcher starting mid-update could claim
  slot 1 and import `app_src` while slot 0 was patching it.
- Replaced `any_peer_slot_occupied` with `claim_peer_slots`, which binds
  and holds every peer port; `main` keeps the guard sockets bound for the
  whole `run_startup_update_if_enabled()` call and closes them in a
  `finally`. A launcher starting during the update sees every slot taken
  and shows the session-limit notice; relaunching after the brief update
  window works. Added the requested regression test: with guards held,
  `claim_session_slot` returns no slot; after release it claims slot 1.
- Verification: full pytest x3 -> `111 passed` each run; smoke OK. Direct
  reproduction: gate_passed_before_peer=True, late_peer_slot=None,
  late_launcher_blocked=True (reviewer had late_peer_slot=1), and
  after_release_slot=1.

### GitHub CLI Sandbox Authentication Rule (GPT-5, default mode)

- Confirmed that a sandboxed `gh` command may report invalid authentication
  even when the host keyring login is valid. Consolidated `AGENTS.md` guidance:
  retry an authorized, narrowly scoped Git or GitHub CLI command with
  `sandbox_permissions: "require_escalated"` before changing plans or asking
  the user to reauthenticate; normal approval requirements still apply.
- Used that path to post the PR #8 review findings after the GitHub connector
  rejected review creation with `403 Resource not accessible by integration`;
  the comment identifies the reviewer as GPT-5 (Codex), high reasoning effort.

### PR #8 Follow-Up Review Round 2 (Claude Fable 5, default mode)

- Reviewer confirmed P2 resolved; P1 was improved but incomplete: the
  connect probe in `any_peer_slot_occupied` misses a peer that is still
  starting up, because `claim_session_slot` holds the claimed port with a
  bound socket that is not listening yet and refuses connections.
- Changed the probe from `socket.create_connection` to a bind attempt,
  which fails against any bound port whether or not it listens; dropped
  the now-unused probe timeout constant. Added the bound-but-not-listening
  regression test the reviewer asked for.
- Fixed the flaky `test_no_peer_slots_occupied_when_peer_ports_are_free`
  (it assumed base+1/base+2 were free without checking; failed twice in
  the reviewer's local runs): a helper now reserves and verifies the full
  contiguous 3-port range, then frees only the peer ports while the own
  slot stays bound like a real claim.
- Verification: full pytest x3 -> `110 passed` each run. Direct
  reproduction of the reviewer's scenario now yields peer_port_bound=True,
  accepting_connections=False, detected_occupied=True (was False).

### PR #8 Review Findings Addressed (Claude Fable 5, default mode)

- Addressed both findings from the agent review on PR #8 (GPT-5/Codex).
  `dev` gained no new commits since the rebase, so no second rebase was
  needed despite expectations.
- P1 (updater under live peers): the startup update now requires slot 0
  AND no peer slot port accepting a TCP connection
  (`any_peer_slot_occupied` in `run_desktop_app.py`, 0.5 s probe). A
  relaunch that reclaims slot 0 while slots 1-2 are live skips the update
  instead of patching `app_src` beneath them. Non-app listeners on peer
  ports also suppress the update (conservative direction).
- P2 (stale same-file refusals): the `current-file` endpoint now reads a
  process-local `_current_filepath` in `session.py` (set by
  `initialize_cache`) instead of the filesystem cache, so a restarted
  window reports no file until one is opened. The cached
  `filepath`/`filename`/`sleep_scores_history` keys are untouched, so
  crash-recovery salvage behavior is unchanged.
- Verification: full pytest -> `109 passed` (5 new tests: peer-port
  probe cases; endpoint ignores a persisted cache filepath;
  `initialize_cache` sets process state). Two-process reproduction of the
  P2 scenario on slot 2 confirmed the cache value persists but the
  endpoint reports empty. `run_desktop_app.py --smoke` OK.

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
