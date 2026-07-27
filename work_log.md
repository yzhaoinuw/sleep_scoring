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

## 2026-07-27

### v0.16.8 Microarousal Spreadsheet Fix (GPT-5)

- Reproduced the reported zero-microarousal spreadsheet result on the current
  v0.16.7 runtime with an annotation array containing an explicit MA bout
  (`sleep_scores == 3`).
- Confirmed that keyboard `4` correctly writes label `3` and **Save
  Annotations** preserves it in the MAT payload, but `get_sleep_segments()`
  constructs bout rows only for Wake `0`, NREM `1`, and REM `2`. Explicit MA
  intervals are therefore omitted before `get_pred_label_stats()` receives
  the bout table.
- The downstream statistics function counts explicit MA rows correctly when
  given one, so the defect is localized to bout construction. The omission
  also removes MA rows from `Sleep_bouts`, excludes their duration from the
  `Sleep_stats` denominator, and can misclassify transitions across the gap.
- A focused reproduction with 20 s NREM, 5 s MA, and 20 s Wake produced only
  the NREM and Wake bout rows and reported MA time/count as zero. The existing
  postprocessing and save-helper suites still passed (`42 passed`), confirming
  that explicit label-3 export coverage is missing.
- Replaced the hard-coded three-stage bout builder with contiguous-run
  construction that preserves every score label, including explicit MA label
  `3`, without changing the downstream short-Wake-to-MA statistics rule.
- Added direct bout-table coverage and a complete **Save Annotations**
  regression that reads both generated workbook sheets and verifies MA bout,
  duration, count, and percentage data.
- The user manually confirmed the corrected export behavior in the desktop app.
- Updated runtime/package metadata to v0.16.8 and added user-facing changelog
  and README guidance for MA annotation and spreadsheet export behavior.
- Verification passed: focused postprocessing/save tests (`44 passed`), full
  pytest (`123 passed`), repository-pinned Black hook, `compileall`,
  `run_desktop_app.py --smoke`, and `git diff --check`.
- The runtime change is limited to `app_src/postprocessing.py`, so it is
  eligible for the automatic source-update release path. A release still needs
  a multi-baseline update asset, publication, and installed-app update checks
  before packaged users receive it.
- Committed the v0.16.8 candidate as `7d1df7e`, pushed it to `dev`, and
  confirmed the exact commit passed GitHub CI (Python tests, JavaScript tests,
  and formatting) plus all CodeQL jobs.
- Built the lightweight asset after verifying that `setup.py` differs from
  each supported release only by its version assignment. The manifest accepts
  v0.16.5, v0.16.6, and v0.16.7, preserves `app_src/config.py`, and contains
  `app_src/__init__.py`, `app_src/make_figure.py`,
  `app_src/postprocessing.py`, and `app_src/session.py` for direct jump-ahead
  updates. Candidate SHA-256:
  `BCDC6CEB3DA78257750643E79FDF44AC9E77A363F9FE397CA37D1EEA2C0FB5A8`.
- Applied the candidate asset to extracted fresh v0.16.5, fresh v0.16.6, and
  chained v0.16.5 -> v0.16.6 -> v0.16.7 packaged fixtures. Every fixture
  reported an update to v0.16.8, preserved the exact `config.py` hash, and
  passed the frozen v0.16.8 smoke check.
- Fast-forwarded `main` from `dev`, pushed both branches through release commit
  `4fc44c9`, created and pushed annotated tag `v0.16.8` at that commit, and
  published the official non-prerelease GitHub Release:
  `https://github.com/yzhaoinuw/sleep_scoring/releases/tag/v0.16.8`.
- The release contains only the 10,179-byte automatic-update ZIP and its
  checksum file. GitHub reports the ZIP digest as
  `sha256:bcdc6ceb3da78257750643e79fdf44ac9e77a363f9fe397ca37d1eea2c0fb5a8`,
  matching the local artifact exactly.
- GitHub's public latest-release endpoint returns v0.16.8 with both expected
  assets, and the app's real release-API check from an already-v0.16.8 source
  checkout reports `no update available`.

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
- After review, fast-forwarded the completed `readme-makeover` branch into
  `dev` and then `main`, pushed both branches, and verified the local,
  remote-tracking, and remote branch refs were aligned.
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
