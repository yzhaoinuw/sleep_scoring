# Agent Collaboration Notes For This Project

## Startup Rule

At the beginning of a new chat or agent session for this project, read this
file first and do not automatically read every markdown file in the repository.
Use the documentation map below to decide which other files are relevant to the
current task.

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

## Worktree Hygiene

Before editing, inspect the current worktree with `git status`. Preserve
unrelated local changes and untracked files. If a task touches files that
already have user changes, work with those changes instead of reverting them.

## Branch Handoff Discipline

Before switching away from an experimental or feature branch, fully resolve the
work on that branch. Confirm whether the branch contains all intended changes,
whether those changes are committed, and whether the user expects them merged,
pushed, or intentionally left parked.

Do not switch to `dev` or start new work on another branch while important
experimental-branch changes are only local, unmerged, or unverified. If related
work accidentally lands on `dev`, move that work back onto the experimental
branch first and retest the combined behavior there before updating `dev`.

Useful checks before switching or merging:

```powershell
git status --short --branch
git log --oneline --left-right --cherry-pick dev...HEAD
git merge-base --is-ancestor dev HEAD
```

## Documentation
Read these documents only as needed:

- `work_log.md`
  - Use when the task needs recent implementation history, experiment outcomes, or verification breadcrumbs.
  - This file is prepended each session and can become long. Read only the two most recent dated entries by default.
  - Find date anchors with ripgrep and read only the slice you need:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md`
  - Search the full log with targeted terms when older context is needed instead of loading the whole file.
  - Prepend a dated entry when a task creates a durable result future collaborators should know about.

- `next_steps.md`
  - Use when planning or continuing unfinished work from previous sessions.
  - Remove items after they are completed. Add new planned follow-ups when they become concrete.

- `project_overview.md`
  - Use when onboarding to the codebase structure or when a task touches an unfamiliar area.

- `README.md`
  - Use when changing user-facing setup, packaging, usage, or input-file expectations.

- `CONTRIBUTING.md`
  - Use when changing collaboration workflow, branch/test expectations, or documentation conventions.

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
