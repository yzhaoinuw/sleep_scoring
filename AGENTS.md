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

Typical startup (the same command works in PowerShell, bash, and zsh):

```
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

## Windows Git Friction

On this Windows workstation, some Git operations may fail even when the plan is
correct. Common symptoms include:

- `cannot spawn sh: No such file or directory`
- `could not read Username for 'https://github.com'`
- `Unable to create .../.git/index.lock: Permission denied`
- `cannot lock ref 'ORIG_HEAD'`

When this happens, do not change branches, reset history, remove lock files, or
change the Git plan just to work around the error. If the worktree is clean or
the intended staged set is already verified, rerun the same narrow Git operation
with the required approval/escalation. For pushes, use the known-good PowerShell
shape:

```
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "git push origin <branch>"
```

After any escalated branch, merge, commit, tag, or push operation, verify the
result with targeted commands such as:

```
git status --short --branch
git rev-parse <local-ref> <remote-tracking-ref>
git ls-remote --heads origin <branch>
```

## Branch Handoff Discipline

Before switching away from an experimental or feature branch, fully resolve the
work on that branch. Confirm whether the branch contains all intended changes,
whether those changes are committed, and whether the user expects them merged,
pushed, or intentionally left parked.

Do not switch to `dev` or start new work on another branch while important
experimental-branch changes are only local, unmerged, or unverified. If related
work accidentally lands on `dev`, move that work back onto the experimental
branch first and retest the combined behavior there before updating `dev`.

Useful checks before switching or merging (portable git commands; run in any shell):

```
git status --short --branch
git log --oneline --left-right --cherry-pick dev...HEAD
git merge-base --is-ancestor dev HEAD
```

## Documentation
Read these documents only as needed:

- `work_log.md` and `work_log_archive/`
  - Use when the task needs recent implementation history, experiment outcomes, or verification breadcrumbs.
  - The live `work_log.md` holds at most the 5 most recent unique calendar dates. Default to reading only the two most recent dated entries.
  - Find date anchors with ripgrep and read only the slice you need:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md`
  - When older context is needed, open the matching file under `work_log_archive/` by its date-range filename, or grep across both at once:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md work_log_archive/`
  - When prepending a dated entry, if today's calendar date already has a `## YYYY-MM-DD` header at the top, add a new `###` session subsection under it. Do not start a second `## YYYY-MM-DD` header for the same date.
  - When prepending a new date would push the live log past 5 unique calendar dates, move the oldest 5 dates as a chunk into a new file at `work_log_archive/work_log_<earliest>_to_<latest>.md`. The live file always holds at most 5 unique dates; each archive file always holds exactly 5.

- `next_steps.md`
  - Use when planning or continuing unfinished work from previous sessions.
  - Remove items after they are completed. Add new planned follow-ups when they become concrete.

- `project_overview.md`
  - Use when onboarding to the codebase structure or when a task touches an unfamiliar area.

- `README.md`
  - Use when changing user-facing setup, packaging, usage, or input-file expectations.

- `CONTRIBUTING.md`
  - Use when changing collaboration workflow, branch/test expectations, or documentation conventions.

## Commit Message Guidelines

Commit messages should use:

- a short title line
- a short body with flat bullet points for additional requested changes when a commit contains multiple user-requested updates

Commit message bullets should describe high-level added or changed behavior, not implementation details.

For feature commits, mention only the user-facing behavior that was added or changed.

Do not mention tests, docs, project memory updates, or behind-the-scenes implementation details in a feature commit message unless that internal work is itself the main purpose of the commit.
