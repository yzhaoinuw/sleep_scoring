# Codex Memory For This Project

## Runtime Environment

When running code, tests, or the desktop app for this repository, use the conda environment:

- `sleep_scoring_dash3.0`

Typical startup:

```powershell
conda activate sleep_scoring_dash3.0
```

In the Codex desktop PowerShell shell, `conda` may not be on `PATH` even though Miniconda is installed.
If `conda` is not recognized, use the environment's Python directly instead of spending time on shell activation:

```powershell
C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest
```

After activation, use that environment for commands such as:

- `pytest`
- `python run_desktop_app.py`
- package import checks
- one-off scripts

## Shell Notes

The Codex PowerShell shell in this project does not support Bash-style `&&`.
Use separate commands or PowerShell separators instead of `&&`.

## Git / Sandbox Notes

`git push` needs network access and should be rerun with escalation if it fails in the sandbox.

`git commit` may also need escalation in this environment when Git hooks invoke shell tooling.

If these commands repeatedly need escalation, prefer approving persistent command prefixes for `git commit` and `git push` so future sessions can run them without extra friction.

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

## Work Log

For future Codex sessions, also read [`codex_work_log.md`](C:\Users\yzhao\python_projects\sleep_scoring\codex_work_log.md) at the start of work.

When adding new entries, prepend the latest session at the top so the freshest context is visible first.
