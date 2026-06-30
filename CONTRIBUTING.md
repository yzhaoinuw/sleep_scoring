# Contributing

This project is increasingly worked on by a mix of human collaborators and agent-assisted sessions. Keep changes easy to review, easy to reproduce, and easy for the next collaborator to continue.

## If You Are Not A Programmer

You can still make a useful contribution. Start by opening a GitHub issue and
choose the form that best matches what you are doing: app bug, data file
problem, feature request, or question.

- Describe what you were trying to do and what happened instead.
- Include screenshots or copied error text when possible.
- Mention your app version, zip name, branch, or operating system if you know it.
- Do not upload sensitive/raw lab data publicly. If an example file is needed,
  the maintainer can help choose a safe way to share it.
- If you are not sure which form to use, choose the question form.

If you want to help test a fix, improve wording, or review a workflow, say so
in the issue. A maintainer can suggest the next step.

## Pull Requests

Pull requests are welcome, but starting with an issue is usually the easiest
path unless a maintainer has already asked for a specific change.

- Keep each pull request focused on one bug, workflow, or documentation topic.
- Use the pull request template to summarize what changed and how you checked it.
- Do not add private lab data, raw recordings, or machine-specific files.
- If you cannot run checks locally, say what blocked you.

## Before You Start

- Agents should start with `AGENTS.md`; it is the canonical project-specific instruction file and points to everything else.

## Branches

- Start from the branch requested by the maintainer. If none is specified, use the current working branch and avoid retargeting history.
- Use focused branch names such as `feature/...`, `fix/...`, `docs/...`, or `agent/...`.
- Keep experimental branches small when possible. If an experiment grows, document what changed and what should be kept or discarded.

## Tests And Verification

- Activate the project conda environment first; see the Runtime Environment section of `AGENTS.md` for the exact command.
- For Python edits, run targeted import or compile checks for the touched modules.
- For UI/navigation JavaScript edits, run a JavaScript syntax check when Node is available.
- For general smoke coverage, run:

```
python -m pytest tests/test_smoke.py -q
```

- Broaden tests when touching preprocessing, postprocessing, FFT, inference, or shared app behavior.
- If a check cannot be run locally, record why in the final handoff or work log.

## Local Troubleshooting

- If Git reports a "detected dubious ownership" warning, mark the repository as safe:

```
git config --global --add safe.directory (Get-Location).Path
```

- If `pre-commit` cannot write to its default cache location, set a repo-local cache before running it:

```
$env:PRE_COMMIT_HOME = Join-Path (Get-Location).Path ".pre-commit-cache"
& "$env:USERPROFILE\miniconda3\envs\sleep_scoring_dash3.0\python.exe" -m pre_commit run --all-files
```

- If Miniconda is installed somewhere else, adjust the Python executable path while keeping the environment name `sleep_scoring_dash3.0`.

## Documentation

- Use repo-relative paths in markdown whenever possible.
- If an example command contains a local absolute path, add a note that collaborators should adapt it to their own clone location.
- Update `project_overview.md` when the active runtime path or active-vs-legacy file map changes.
- For doc-map, work-log rotation, and `next_steps.md` conventions, follow `AGENTS.md`.

## Commits

- Follow the Commit Message Guidelines in `AGENTS.md`.

## Research Practices

When iterating on experimental scoring pipelines, do not rush to remove useful traces such as:

- parameter-rich output filenames
- debug visualizations
- intermediate diagnostics
- comparison-friendly breadcrumbs that make it easy to match one run against another

These traces are often what makes it possible to explain behavior regressions later. Prefer keeping them until the behavior is stable and the comparison value is clearly gone, then clean them up deliberately.
