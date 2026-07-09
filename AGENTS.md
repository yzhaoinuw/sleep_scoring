# Agent Collaboration Notes For This Project

## Startup Rule

At the start of a new session, read this file first. Do not read every Markdown
file automatically; use the map below to choose only what is relevant.

## Runtime

- Project folder: `sleep_scoring`
- Conda env: `sleep_scoring_dash3.0`
- Conda envs live under `C:\Users\yzhao\miniconda3\envs\`.
- If `conda` is not on PATH, use
  `C:\Users\yzhao\miniconda3\condabin\conda.bat`.

Common commands:

```powershell
conda activate sleep_scoring_dash3.0
python run_desktop_app.py
python run_desktop_app.py --smoke
python -m pytest --basetemp .pytest_tmp\codex -p no:cacheprovider -q
```

## Active App And Packaging

- Desktop entrypoint: `run_desktop_app.py`.
- Active runtime package: `app_src/`.
- Version source of truth: `app_src/__init__.py`; keep `setup.py` aligned.
- Windows packaging docs/scripts: `packaging/windows/`.
- Full Windows package: `packaging/windows/make_full_app_zip.ps1`.
- Source-update asset: `packaging/windows/make_source_update_asset.ps1`.

Startup auto-update lives in `run_desktop_app.py` before importing `app_src`.
Packaged builds check GitHub Release source-update assets. Source runs skip the
check unless update-test env vars are set. Only the first app window (port
slot 0) runs the check; later windows skip it so `app_src/` is never patched
under a running window. Use a full app zip when dependencies,
models, packaging, launcher, or runtime layout changed; use source-update assets
only for compatible `app_src/` changes.

Updater config:

- app: `sleep_scoring`
- version file: `app_src/__init__.py`
- release API: `https://api.github.com/repos/yzhaoinuw/sleep_scoring/releases/latest`
- asset prefix: `sleep_scoring_app_update_`
- allowed payload path: `app_src/`
- env vars: `SLEEP_SCORING_SKIP_UPDATE`, `SLEEP_SCORING_UPDATE_ZIP_URL`,
  `SLEEP_SCORING_UPDATE_RELEASE_API_URL`,
  `SLEEP_SCORING_UPDATE_ASSET_PREFIX`,
  `SLEEP_SCORING_UPDATE_TIMEOUT_SECONDS`

## Worktree And Git

Before editing, run `git status --short --branch`. Preserve unrelated local
changes and untracked files.

This Windows checkout often needs approval/escalation for `git switch`,
`git merge --ff-only`, `git fetch`, `git push`, and tag/ref updates. Known
failure signs include `cannot spawn sh`, auth prompts, `.git/index.lock`,
`FETCH_HEAD`, or `ORIG_HEAD` lock errors. Keep the Git plan narrow; do not
reset, remove locks, or change branches to work around friction. For pushes,
use:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "git push origin <branch>"
```

After branch, merge, commit, tag, or push work, verify with targeted refs:

```powershell
git status --short --branch
git rev-parse <local-ref> <remote-tracking-ref>
git ls-remote --heads origin <branch>
git ls-remote --tags origin <tag>
```

Before leaving an experimental branch, make sure its work is committed, tested,
and either merged/pushed or intentionally parked.

## Release / Tag Gate

Treat any request that combines commit, push, and tag, or asks to publish/cut a
release, as release work. Before creating or pushing a tag:

- verify the local date with `Get-Date -Format yyyy-MM-dd`;
- update version metadata (`app_src/__init__.py`, `setup.py`);
- update release notes/changelog and user-facing docs when behavior changed;
- update `work_log.md` with verification and branch/tag state;
- run the relevant tests/smoke/package checks;
- only then tag, push, and verify pushed refs.

Never write future-dated work-log entries. The current treaty validator rejects
work-log dates after the workstation date.

## Documentation Map

- `work_log.md` / `work_log_archive/`: recent implementation history,
  decisions, verification, and release state. Live log holds at most 5 unique
  dates; archive the oldest 5-date chunk when needed. Read only the relevant
  date slice; find headers with
  `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md work_log_archive/`.
- `next_steps.md`: concrete unfinished work. Remove completed items and add
  real follow-ups only.
- `project_overview.md`: codebase map for unfamiliar areas and active-vs-legacy
  boundaries.
- `README.md`: user-facing setup, packaging, usage, and input-file contracts.
- `CONTRIBUTING.md`: collaboration workflow, branch/test expectations, and doc
  conventions.

Update `work_log.md` after substantive sessions: file edits, meaningful
debugging/validation, technical decisions, branch/release state changes, or
follow-ups future agents need. Skip casual Q&A and trivial one-off commands.

## Commit Messages

Use a short title line. Add a short body with flat bullets only when a commit
contains multiple requested changes. Describe high-level behavior, not internal
implementation details. Do not mention tests/docs/project-memory work unless
that is the main purpose of the commit.
