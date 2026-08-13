# Guidelines and Tips for Agents

This is the first file to read in a session. Keep it lean and project-specific;
generic mechanics live in
[`treaty_docs/treaty_conventions.md`](treaty_docs/treaty_conventions.md).

## Startup Rule

At the start of a session, read this file first. Use the documentation map
below instead of automatically reading every Markdown file.

## Runtime Environment

- Project folder: `sleep_scoring`
- Conda environment: `sleep_scoring_dash3.0`
- Conda environments live under `C:\Users\yzhao\miniconda3\envs\`.
- If `conda` is not on PATH, use the `condabin\conda.bat` under that Miniconda
  installation or open a Miniconda terminal.

Activate the environment before running the app, tests, or one-off scripts:

```powershell
conda activate sleep_scoring_dash3.0
```

## Common Tasks

```powershell
python run_desktop_app.py
python run_desktop_app.py --smoke
python -m pytest --basetemp .pytest_tmp\codex -p no:cacheprovider -q
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_full.ps1 -ValidateOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_lightweight.ps1
```

Before editing, run `git status --short --branch` and preserve unrelated local
changes and untracked files. Before committing, run the checks appropriate to
the touched surface and `git diff --check`.

## When To Update Treaty Docs

At the end of substantive work, prepend the decisions and reusable evidence to
`treaty_docs/work_log.md` and keep `treaty_docs/next_steps.md` accurate. Routine
changes already clear from the diff do not need narration. See
[Work Log Discipline](treaty_docs/treaty_conventions.md#work-log-discipline)
for the exact criteria and entry structure.

## Branch Handoff Discipline

Pull requests normally target `dev`; `main` is the published integration point
for releases. Before leaving an experimental branch, ensure its work is tested,
committed, and either delivered or intentionally parked. See
[Branch Handoff](treaty_docs/treaty_conventions.md#branch-handoff).

## Release / Tag Checklist

Treat any request combining commit, push, and tag—or asking to publish/cut a
release—as release work. Before creating or pushing a tag:

- verify the local date with `Get-Date -Format yyyy-MM-dd`;
- align `app_src/__init__.py`, `setup.py`, and the `version` and
  `date-released` fields in `CITATION.cff`;
- update release notes/changelog and user-facing docs when behavior changed;
- update `treaty_docs/work_log.md` with verification and branch/tag state;
- run the applicable tests, smoke checks, and package gate;
- only then tag, push, and verify pushed refs.

After publishing, confirm Zenodo minted a DOI instead of assuming it did. Query
`https://zenodo.org/api/records?q=sleep_scoring&all_versions=true` rather than
reading webhook status codes: Zenodo returns 403 on the `published` action and
202 on `released`, so a non-2xx there does not mean the deposit failed. A
deposit genuinely stuck on "processing" cannot be repaired by redelivering the
webhook—Zenodo rejects the reused delivery GUID with 409—so delete and recreate
the GitHub release, which keeps the tag unless `--cleanup-tag` is passed. Zenodo
archives only the source zipball, never release assets. See the 2026-08-01
work-log entry.

Both release gates reject stale citation metadata. Zenodo reads `CITATION.cff`
when a release is published; later repository corrections do not propagate to
an existing Zenodo record. Use `release_full.ps1` or `release_lightweight.ps1`
as the candidate gate, and do not rerun individual checks after a passing gate
unless the candidate commit changes. Never write future-dated work-log entries.
The full procedure is in [Release Gate](treaty_docs/treaty_conventions.md#release-gate).

## Updating The Treaty

Treaty updates are a maintainer decision, distinct from work-log maintenance.
Use `treaty diff`, then `treaty update --dry-run`, then `treaty update`.
Preserve project guidance and resolve every unmerged file before committing;
see [Updating The Treaty](treaty_docs/treaty_conventions.md#updating-the-treaty).

## Documentation

- `treaty_docs/treaty_conventions.md`: upstream-maintained work-log, branch,
  release, and update mechanics; avoid local edits.
- `treaty_docs/work_log.md` / `treaty_docs/work_log_archive/`: decisions,
  evidence, and delivery state.
  The live log holds at most five unique dates; find date anchors with
  `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' treaty_docs/work_log.md`
  `treaty_docs/work_log_archive/`.
- `treaty_docs/next_steps.md`: unfinished work; `Currently Hot` names active threads.
- `project_overview.md`: codebase map and active-versus-legacy boundaries.
- `README.md`: user-facing installation, usage, and input-file contracts.
- `CONTRIBUTING.md`: collaboration and verification workflow.

## Commit Message Guidelines

Use a short title line. Add a short body with flat bullets only when a commit
contains multiple requested changes. Describe high-level behavior, not internal
implementation details. Do not mention tests, docs, or project-memory work in a
feature commit unless that internal work is the commit's main purpose.

## Git Ownership Note

This Windows checkout may need approval for Git metadata or host credentials.
If an authorized `git` or `gh` command fails with a metadata-lock, sandbox, or
credential-boundary error, retry the same narrow command with the required
approval before changing the plan. Do not remove locks, reset state, or broaden
the command. For pushes, use `git push origin <branch>`.

After branch, merge, commit, tag, or push work, verify with targeted refs:

```powershell
git status --short --branch
git rev-parse <local-ref> <remote-tracking-ref>
git ls-remote --heads origin <branch>
git ls-remote --tags origin <tag>
```

For `detected dubious ownership`, use a repository-scoped safe-directory
override or mark only this repository safe; do not change OS ownership.

## Pre-commit Note

The reliable Black check on this Windows environment is:

```powershell
$env:PYTHONUTF8='1'; conda run -n sleep_scoring_dash3.0 pre-commit run black --all-files
```

If pre-commit cannot write its default cache, point `PRE_COMMIT_HOME` at a
repo-local ignored directory rather than disabling the hook.

## Project-Specific Reminders

- Desktop entrypoint: `run_desktop_app.py`; active runtime package: `app_src/`.
- Version source of truth: `app_src/__init__.py`; keep `setup.py` and
  `CITATION.cff` aligned for releases.
- Windows packaging lives in `packaging/windows/`; full packages use
  `make_full_app_zip.ps1`, while compatible code-only updates use
  `make_source_update_asset.ps1`.
- Startup update logic runs in `run_desktop_app.py` before importing `app_src`.
  Source runs skip it unless test overrides are set, and only window slot 0
  checks so `app_src/` is never patched under another running window.
- Use a full package when dependencies, models, packaging, the launcher, or
  runtime layout changes; source-update assets may contain only compatible
  `app_src/` changes.
- Updater contract: app `sleep_scoring`, version file `app_src/__init__.py`,
  latest release `https://github.com/yzhaoinuw/sleep_scoring/releases/latest`,
  asset prefix `sleep_scoring_app_update_`, allowed payload `app_src/`.
- Update tests use `SLEEP_SCORING_UPDATE_*`, `SLEEP_SCORING_SKIP_UPDATE`, and
  `SLEEP_SCORING_FORCE_UPDATE_CHECK`.
