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

## 2026-07-23

### Public-facing documentation makeover (GPT-5)

- Created `readme-makeover` from a clean `dev` worktree and pushed the
  completed documentation makeover to the matching branch on `origin`.
- Reorganized `README.md` around a compact installation-choice table that
  keeps the private packaged Windows route and the public Windows/macOS source
  route together at the top, with explicit audience, access, update, and
  optional sDREAMER requirements.
- Added a table of contents and limited collapsible sections to secondary
  folder-layout troubleshooting and optional input fields; kept first-run and
  core usage instructions visible.
- Corrected stale user guidance about the supported three-window workflow and
  the `FIX_NE_Y_RANGE` setting.
- Replaced the local-only inventory in `project_overview.md` with an
  architecture map of tracked GitHub content and explicit boundaries for
  private test data, model checkpoints, generated videos, and package outputs.
- Removed maintainer-specific absolute Conda and PowerShell paths from
  `AGENTS.md` while preserving portable Windows guidance.
- Follow-up review standardized README section-title capitalization and
  clarified that **Save Annotations** writes only to the path confirmed in the
  native Save dialog; opening, annotating, and predicting do not overwrite the
  source `.mat` file.
- Verified all relative Markdown link targets in the active docs, balanced the
  README `<details>` tags, and passed `git diff --check`. `treaty validate .`
  could not run because the `treaty` command is not installed or available on
  PATH in this shell.

## 2026-07-22

### Post-release documentation placement (GPT-5)

- Removed the sleep-stage color customization block from `README.md`; like the
  app's other direct config options, it does not need a dedicated user-guide
  section.
- Moved the semantic Python-config merge design, deployment boundary, test
  matrix, and the separable multi-lineage builder follow-up out of this app's
  `next_steps.md` and into upstream updater issue
  `https://github.com/yzhaoinuw/desktop_app_source_updater/issues/2`.
- The GitHub connector could read and search the upstream repository but lacked
  issue-write permission, so the authenticated GitHub CLI created the issue
  after confirming that no open duplicate existed.

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
- Built the v0.16.7 source-update asset from the committed runtime bytes. Its
  manifest accepts v0.16.5 and v0.16.6 and contains only
  `app_src/__init__.py`, `app_src/make_figure.py`, and `app_src/session.py`;
  `app_src/config.py` is absent. Final SHA-256:
  `A874CBB6900762A5C06ECB13C29E1586D51535CA2141A3843031D516DC2DE40F`.
- Applied that asset to three extracted released-app fixtures: fresh v0.16.5,
  fresh full v0.16.6, and v0.16.5 previously patched to v0.16.6. All reported
  an update to v0.16.7 and passed the frozen smoke check. The exact config hash
  remained unchanged in every fixture; an old config used the fallback palette
  while preserving an edited `SLEEP_SCORING_MODEL`, and a config with manually
  added colors exposed those exact colors through `make_figure`.
- Tagged release commit `68a7fd3` as v0.16.7, fast-forwarded both `dev` and
  `main`, pushed the tag, and published the lightweight GitHub Release with
  only the update ZIP and checksum. GitHub reports the expected 6,766-byte ZIP
  and the same SHA-256 digest.
- Restored an edited fixture to the exact v0.16.5 runtime files, removed the
  local update URL override, and confirmed its frozen updater discovered and
  downloaded the public v0.16.7 asset, preserved the config hash, and passed
  the v0.16.7 smoke check. The release is available at
  `https://github.com/yzhaoinuw/sleep_scoring/releases/tag/v0.16.7`.

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
