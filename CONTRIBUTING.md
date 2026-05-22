# Contributing

This project is increasingly worked on by a mix of human collaborators and agent-assisted sessions. Keep changes easy to review, easy to reproduce, and easy for the next collaborator to continue.

## Before You Start

- Read `AGENTS.md`, `project_overview.md`, and `next_steps.md`.
- For `work_log.md`, read only the two most recent dated entries by default. Search the full log only when older context is relevant.
- Use the `sleep_scoring_dash3.0` conda environment unless the task explicitly needs a different environment.
- Check `git status` before editing. Preserve unrelated local changes and untracked files.
- Treat `app_src/app_dev.py`, `components_dev.py`, and `make_figure_dev.py` as the current active UI path unless the task says otherwise.

## Branches

- Start from the branch requested by the maintainer. If none is specified, use the current working branch and avoid retargeting history.
- Use focused branch names such as `feature/...`, `fix/...`, `docs/...`, or `agent/...`.
- Keep experimental branches small when possible. If an experiment grows, document what changed and what should be kept or discarded.

## Tests And Verification

- Activate the project environment before running checks:

```powershell
conda activate sleep_scoring_dash3.0
```

- For Python edits, run targeted import or compile checks for the touched modules.
- For UI/navigation JavaScript edits, run a JavaScript syntax check when Node is available.
- For general smoke coverage, run:

```powershell
python -m pytest tests/test_smoke.py -q
```

- Broaden tests when touching preprocessing, postprocessing, FFT, inference, or shared app behavior.
- If a check cannot be run locally, record why in the final handoff or work log.

## Documentation

- Keep `next_steps.md` forward-looking. Put completed experiments, measurements, and outcomes in `work_log.md`.
- Use repo-relative paths in markdown whenever possible.
- If an example command contains a local absolute path, add a note that collaborators should adapt it to their own clone location.
- Update `project_overview.md` when the active runtime path or active-vs-legacy file map changes.

## Commits

- Use a short title line.
- Add a short body with flat bullets when a commit contains multiple requested changes.
- Commit message bullets should describe high-level added or changed behavior, not implementation details.
- For feature commits, do not mention tests, docs, project memory updates, or behind-the-scenes implementation details unless that internal work is the main purpose of the commit.

## Agent Collaboration Notes

- Prefer small, inspectable edits over broad rewrites.
- Do not remove useful experimental traces, output filenames, diagnostics, or comparison breadcrumbs until the behavior is stable and their value is clearly gone.
- When resuming work from another agent or collaborator, first read the current docs and inspect the worktree rather than assuming the previous mental model is still current.
