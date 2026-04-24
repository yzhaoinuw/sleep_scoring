# Codex Memory For This Project

## Runtime Environment

When running code, tests, or the desktop app for this repository, use the conda environment:

- `sleep_scoring_dash3.0`

Typical startup:

```powershell
conda activate sleep_scoring_dash3.0
```

After activation, use that environment for commands such as:

- `pytest`
- `python run_desktop_app.py`
- package import checks
- one-off scripts

## Git Ownership Note

If Git reports a "detected dubious ownership" warning for this repo, mark this repository as safe:

```powershell
git config --global --add safe.directory C:/Users/yzhao/python_projects/sleep_scoring
```

This is the preferred fix unless the repository ownership itself needs to be changed at the OS level.

## Pre-commit Note

If `pre-commit` cannot write to its default cache location, set a repo-local cache before running it:

```powershell
$env:PRE_COMMIT_HOME = "C:\Users\yzhao\python_projects\sleep_scoring\.pre-commit-cache"
```

Then run:

```powershell
C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pre_commit run --all-files
```

## Commit Message Guidelines

Commit messages should use:

- a short title line
- a short body with flat bullet points for additional requested changes when a commit contains multiple user-requested updates

Commit message bullets should describe high-level added or changed behavior, not implementation details.

For feature commits, mention only the user-facing behavior that was added or changed.

Do not mention tests, docs, project memory updates, or behind-the-scenes implementation details in a feature commit message unless that internal work is itself the main purpose of the commit.

## Pipeline Iteration Reminder

When iterating on experimental scoring pipelines, do not rush to remove useful traces such as:

- parameter-rich output filenames
- debug visualizations
- intermediate diagnostics
- comparison-friendly breadcrumbs that make it easy to match one run against another

These traces are often what makes it possible to explain behavior regressions later.
Prefer keeping them until the behavior is stable and the comparison value is clearly gone, then clean them up deliberately.
