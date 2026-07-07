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

## 2026-07-07

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
