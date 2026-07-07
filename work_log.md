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
