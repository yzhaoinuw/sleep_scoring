# Guidelines and Tips for Agents

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

## Common Tasks

Short recipes for the things you'll usually do in a session. All commands assume the conda env above is active.

Run the desktop app for manual testing:

```
python run_desktop_app.py
```

Run the test suite (the CI-equivalent subset — skips `ml`-marked tests that need Torch):

```
pytest -v -m "not ml"
```

Run everything, including the ML-dependent tests, when Torch is installed:

```
pytest -v
```

Pre-flight checklist before committing:

- Black formatting passes: `python -m black --check --diff .` (matches the CI `format` job)
- `pytest -m "not ml"` is green (matches the CI `test` job)
- A new entry has been prepended to `codex_work_log.md` describing what was done, intended profiling signal if any, and the verification commands that were actually run

If the change touches active modules (`app_src/app_dev.py`, `components_dev.py`, `make_figure_dev.py`, `preprocessing.py`, `postprocessing.py`, `get_fft_plots.py`), confirm `from app_src import <module>` still imports — the smoke tests in `tests/test_smoke.py` cover this.

Fetch recent work log context efficiently — instead of reading the full file, grep the date anchors and read just the slice you need:

```
grep -nE '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' codex_work_log.md
```

Take the line numbers of the first N entries you want, then read from the first one through just before entry N+1. `codex_work_log_archive.md` uses the same convention for older entries. The same anchor-grep pattern works for any structured Markdown doc in the repo (`project_overview.md`, `next_steps.md`) — `grep -n '^## ' <file>` for the section map, then a targeted slice read rather than loading the whole file.

See [`project_overview.md`](project_overview.md) for the active vs. legacy code map before touching anything under `app_src/`.

## Git Ownership Note

If Git reports a "detected dubious ownership" warning for this repo, mark this repository as safe:

```powershell
git config --global --add safe.directory C:/Users/yzhao/python_projects/sleep_scoring
```

macOS / Linux equivalent (substitute your own repo path):

```bash
git config --global --add safe.directory "$HOME/python_projects/sleep_scoring"
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

macOS / Linux equivalent (run from inside the activated env):

```bash
export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
python -m pre_commit run --all-files
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
