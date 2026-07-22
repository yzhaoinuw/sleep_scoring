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

## 2026-07-21

### v0.16.7 User-Config Update Repair (GPT-5, default mode)

- Traced the failed v0.16.6 -> v0.16.7 automatic update to the installed
  `SLEEP_SCORING_MODEL = "sdreamer"` setting. That one legitimate edit changed
  the `app_src/config.py` hash, so the updater refused to replace the file.
- Made figure loading compatible with pre-v0.16.7 config files by falling back
  to the existing sleep-stage palette when `SLEEP_STAGE_COLORS` is absent.
- Marked `app_src/config.py` as user-owned in the source-update packaging step,
  so automatic updates preserve it instead of treating user settings as
  replaceable runtime code.
- Committed the repair as `72a324a`, pushed it directly to `dev`, and
  fast-forwarded `main` without a pull request. The v0.16.7 tag was not moved
  and the version was not bumped.
- Replaced only the existing v0.16.7 update ZIP and checksum assets. The public
  update ZIP is 6,686 bytes with SHA-256
  `DB647CA1C81AB1CEBBCCDD026ECD4E34B7ADA9F3C9B4332DC5BE9357BD78119E`.
- Verification:
  - Focused config/update packaging suite -> `14 passed`.
  - Full pytest -> `122 passed, 1 warning`; source smoke check -> passed.
  - The exact installed v0.16.6 config hash
    `67be73b2c597e84b58f708520bd6a89737d898aa448a500185d798065b2536c6`
    remained unchanged after applying both the local candidate and the public
    GitHub asset; both runs updated to v0.16.7 and retained `"sdreamer"`.

### v0.16.7 Sleep-Stage Color Release Preparation (GPT-5, default mode)

- Classified the change as a lightweight source-update release: runtime edits
  are confined to compatible `app_src` display code and version metadata; no
  dependency, model, launcher, updater, PyInstaller, or packaged-layout change
  requires a new full Windows package.
- Set runtime and package metadata to `v0.16.7`. The v0.16.6 Windows ZIP remains
  the full base for new users, and existing compatible installations will use
  the v0.16.7 automatic source-update asset.
- Updated the lightweight-release roadmap to make this color-configuration
  release the first trial and move the full-path video-association fix to the
  next lightweight patch.
- The first aligned candidate asset updated fresh v0.16.5 and v0.16.6 full
  packages successfully, but the required v0.16.5 -> v0.16.6 -> v0.16.7
  fixture exposed the documented same-version byte-lineage gap: files left
  untouched by the v0.16.6 patch differed from the v0.16.6 full-package bytes.
- Enhanced the app-specific alignment helper to preserve every verified Git,
  full-package, and previously patched baseline hash when one installed version
  has multiple byte lineages. Added regression coverage for multiple package
  lineages instead of silently replacing the earlier hash.
- The shared builder still treats any `setup.py` change as a full-package
  trigger. For this release, independently verified that its only change is the
  required `0.16.6` -> `0.16.7` metadata bump, then used a narrow builder
  override that retained every other default dependency and packaging blocker.
- Built the final automatic source-update candidate with
  `app_src/__init__.py`, `app_src/config.py`, `app_src/make_figure.py`, and
  `app_src/session.py`. SHA-256:
  `25FC95E550BB0520F6832EE8EEF35CACAD1D304BDA2CC4CC7CB58D60D2E529E5`.
- Pre-release verification:
  - Full pytest before the lineage fix -> `119 passed, 1 warning`; after the
    fix -> `120 passed, 1 warning` (existing Flask-Caching deprecation warning).
  - Focused update-asset packaging tests -> `4 passed`.
  - Clientside Jest suite -> `38 passed`.
  - Repository-pinned Black 25.12.0 hook -> passed.
  - `python -m compileall -q app_src run_desktop_app.py`, `git diff --check`,
    and `python run_desktop_app.py --smoke` -> passed; smoke reported
    `Sleep Scoring App v0.16.7 smoke check OK`.
  - Fresh v0.16.5 and v0.16.6 full-package fixtures each applied the v0.16.7
    asset and passed the frozen executable smoke check.
  - A fresh v0.16.5 package applied the published v0.16.6 asset, then the
    v0.16.7 asset, and passed the frozen v0.16.7 smoke check. All three fixtures
    confirmed `SLEEP_STAGE_COLORS` was present after updating.
- Final release verification and delivery:
  - PR #9 (`dev` -> `main`) passed Python, JavaScript, formatting, and CodeQL
    checks and merged as `e413c88`; `dev`, `main`, and annotated tag `v0.16.7`
    all reached that release commit before publication.
  - Published the lightweight release with only the automatic source-update
    ZIP and SHA-256 file:
    `https://github.com/yzhaoinuw/sleep_scoring/releases/tag/v0.16.7`.
  - GitHub reported the update ZIP at 8,438 bytes with remote digest
    `25FC95E550BB0520F6832EE8EEF35CACAD1D304BDA2CC4CC7CB58D60D2E529E5`,
    matching the local artifact; the public latest-release endpoint selected
    `v0.16.7` and exposed both expected assets.
  - With no URL override, a fresh v0.16.6 full-package extraction queried the
    real GitHub latest-release endpoint, printed
    `updated to v0.16.7 (4 changed files)`, and passed the frozen v0.16.7 smoke
    check with `SLEEP_STAGE_COLORS` present.
  - This documentation-only publication follow-up leaves the release tag fixed
    on the exact tested merge commit.

### Configurable Sleep-Stage Colors (GPT-5, default mode)

- Moved the existing Wake, NREM, REM, and MA palette into the named
  `SLEEP_STAGE_COLORS` mapping in `app_src/config.py`; the score heatmaps and
  legend now derive their colors from that mapping while retaining the exact
  previous defaults.
- Documented that users can supply Plotly-compatible hex or RGB values and
  must restart the app after changing the configuration.
- Added regression coverage that verifies the figure palette and four-class
  colorscale are sourced from the configuration mapping.
- Rotated the 2026-07-07 through 2026-07-14 five-date work-log chunk to
  `work_log_archive/work_log_2026-07-07_to_2026-07-14.md`.
- The release work began on `dev`, which was already one documentation commit
  ahead of `origin/dev` before the color change.
- Verification:
  - `python -m pytest tests/test_smoke.py --basetemp
    .pytest_tmp/codex_stage_colors -p no:cacheprovider -q` -> `8 passed`.
  - `python -m pytest --basetemp .pytest_tmp/codex_stage_colors_full -p
    no:cacheprovider -q` -> `119 passed, 1 warning` (existing Flask-Caching
    deprecation warning).
  - `python run_desktop_app.py --smoke` ->
    `Sleep Scoring App v0.16.6 smoke check OK`.
  - `git diff --check` -> clean.
  - Targeted Black check could not start because the installed Black runtime
    raised its internal `AssertionError: LAZY` while loading its grammar.
