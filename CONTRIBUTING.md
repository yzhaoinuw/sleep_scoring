# Contributing

This project welcomes issue reports, workflow feedback, documentation fixes, and
code changes from both human collaborators and agent-assisted sessions. Issues
are the easiest starting point; pull requests are welcome when you already know
what change you want to propose.

Do not upload sensitive/raw lab data publicly. If a sample file is needed, the
maintainer can help choose a safe way to share it.

## Human Contributors

### Report A Problem Or Idea

Use GitHub's **New issue** button and choose the form that best matches what you
are doing: app bug, data file problem, feature request, or question.

- Describe what you were trying to do and what happened instead.
- Include screenshots or copied error text when possible.
- Mention your app version, zip name, branch, or operating system if you know it.
- If you are not sure which form to use, choose the question form.

### Make A Pull Request

Start with an issue unless a maintainer has already asked for a specific change.
Pull requests should usually target `dev`, not `main`.

- If you do not have write access, fork the repository and work from your fork.
- On GitHub, click **Fork**, clone your fork, then create a focused branch from
  `dev`.
- Create a focused branch such as `fix/...`, `feature/...`, or `docs/...`.
- Keep each pull request focused on one bug, workflow, or documentation topic.
- Use the pull request template to summarize what changed and how you checked it.
- Do not add private lab data, raw recordings, or machine-specific files.

### Set Up The Project

For full source-install instructions, see the README. The short path is:

```bash
conda env create -f environment.yml
conda activate sleep_scoring_dash3.0
```

If you already have the environment, update it from the repository root:

```bash
conda env update -f environment.yml
conda activate sleep_scoring_dash3.0
```

### Check Your Change

For code changes, run the checks that match your edit. The GitHub CI currently
runs formatting and non-ML tests:

```bash
python -m black --check --diff .
python -m pytest -v -m "not ml" --tb=short
```

For documentation-only changes, checking the rendered Markdown is usually
enough. If you cannot run a check locally, say what blocked you in the pull
request.

### Working With An AI Agent

If you use an AI coding agent, ask it to read `AGENTS.md` and this file before
editing. Before you open a pull request, ask the agent to show:

- which files changed
- what checks it ran
- anything it could not verify

Review the changes yourself, especially any files containing data paths, sample
data, credentials, or machine-specific settings.

## Agent Collaborators

This repository follows the Agent Collab Treaty style: keep local truth visible,
preserve user work, document handoffs, and make verification easy to repeat.

- Start every session with `AGENTS.md`; it is the canonical project instruction
  file and points to the relevant documentation.
- Follow the `AGENTS.md` documentation map instead of reading every Markdown
  file by default.
- Inspect `git status --short --branch` before editing, and preserve unrelated
  local changes.
- Respect branch handoff discipline. Do not switch away from branches with
  unresolved local work, and generally aim PR/publish work through `dev` unless
  the maintainer asks otherwise.
- Update `work_log.md` for material implementation, verification, release, or
  workflow changes; follow the rotation rules in `AGENTS.md`.
- Use `project_overview.md` when the active runtime path or active-vs-legacy map
  matters, and `next_steps.md` for concrete unfinished follow-ups.
- Run focused checks for touched code paths and record any checks that could not
  be run.
- Follow the Windows Git friction guidance in `AGENTS.md` when credential-helper
  or Git lock errors interrupt an otherwise-correct Git operation.
- When iterating on experimental scoring pipelines, preserve useful diagnostics
  until the behavior is stable and their comparison value is clearly gone.
