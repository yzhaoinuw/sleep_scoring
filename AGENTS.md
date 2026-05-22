# Agent Collaboration Notes For This Project

## Local Path Conventions

These notes assume collaborators keep the project folder named `sleep_scoring`
and use the conda environment named `sleep_scoring_dash3.0`.

Any absolute path shown below is only an example from the original development
machine. Agents and collaborators should adapt the user/home prefix and clone
location to their local workstation.

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

## Work Log Reading Guidance

`work_log.md` is prepended with new session notes and can become long. At the
start of a session, read only the two most recent dated entries unless the task
needs deeper history. Search the full work log with targeted terms when older
context is needed instead of loading the whole file.

## Git Ownership Note

If Git reports a "detected dubious ownership" warning for this repo, mark this repository as safe:

```powershell
git config --global --add safe.directory (Get-Location).Path
```

This is the preferred fix unless the repository ownership itself needs to be changed at the OS level.
If running the command outside the repository root, replace `(Get-Location).Path` with the absolute path to the local `sleep_scoring` clone.

## Pre-commit Note

If `pre-commit` cannot write to its default cache location, set a repo-local cache before running it:

```powershell
$env:PRE_COMMIT_HOME = Join-Path (Get-Location).Path ".pre-commit-cache"
```

Then run:

```powershell
& "$env:USERPROFILE\miniconda3\envs\sleep_scoring_dash3.0\python.exe" -m pre_commit run --all-files
```

If Miniconda is installed somewhere else, adjust the Python executable path while keeping the environment name `sleep_scoring_dash3.0`.

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
