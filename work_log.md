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

## 2026-07-22

### v0.16.7 Lightweight Color-Configuration Release Preparation (GPT-5)

- Added a compatibility accessor in `make_figure.py`: configs that define
  `STAGE_COLORS` use the user's four-color list, while preserved pre-v0.16.7
  configs use the original in-code palette without an import failure.
- Restored the source-asset preservation rule that removes
  `app_src/config.py` from both the update ZIP and manifest, so automatic
  updates do not hash-check or overwrite user settings.
- Documented the manual `STAGE_COLORS` block for automatically updated users,
  set runtime and package metadata to v0.16.7, and recorded the semantic
  config-merge design for the next full Windows redistribution.
- Restored multi-lineage baseline handling in the app-specific update-asset
  aligner so released full-package bytes and previously source-patched bytes
  can both be accepted safely.
- Pre-release verification completed so far:
  - Focused config/update packaging tests: `13 passed`.
  - Full pytest: `121 passed, 1 warning`.
  - Repository-pinned Black hook, compile check, source smoke check, and
    `git diff --check`: passed.
  - Downloaded v0.16.5 and v0.16.6 release baselines matched their published
    SHA-256 digests exactly.
- Final asset construction, installed-app fixture checks, tag, push, and GitHub
  publication are pending.

## 2026-07-21

### Configurable sleep-stage colors (Opus 4.8)

- Moved `STAGE_COLORS` (the four sleep-stage colors used by the sleep-score
  heatmap and legend) from `app_src/make_figure.py` into `app_src/config.py` so
  users can recolor stages by editing config, keeping the previous colors as the
  default. Added a comment documenting stage order (Wake/NREM/REM/MA) and
  accepted color formats.
- `make_figure.py` now imports `STAGE_COLORS` from config; the derived
  `COLORSCALE` recomputes from it. `run_inference_stats_model.py` still imports
  `STAGE_COLORS` via `make_figure`, which re-exposes it unchanged.
- Verified config/make_figure/inference all share the same object, `COLORSCALE`
  builds for 3- and 4-class cases, and `run_desktop_app.py --smoke` passes.
- Committed to `dev` only; not released. NOT auto-update-safe as written: this
  changes `config.py`, which the source updater (`allowed_payload_paths=("app_src/",)`)
  would overwrite, wiping user config, and the hard `from app_src.config import
  STAGE_COLORS` would crash on any un-updated config. The reverted July-21
  v0.16.7 solved both (in-code default + `getattr` fallback, plus a
  `--preserve-path app_src/config.py` exclusion); re-adopt that before shipping
  via the Windows auto-updater.
