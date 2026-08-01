# Treaty Conventions

The generic mechanics of the Agent Collab Treaty: what belongs in the work log, how dated entries are written and rotated, how branches are handed off, how releases are gated, and how `treaty update` behaves.

**This file is maintained upstream.** `treaty update` keeps it current, which also makes it the file where local edits are most likely to collide. Project-specific answers — the runtime, the common tasks, the reminders, the documentation map — belong in [`AGENTS.md`](AGENTS.md), which upstream does not revise.

Read this when you are about to *perform* one of these procedures. `AGENTS.md` is what you read at the start of a session.

## Work Log Discipline

**The work log records decisions about the project — not the content of the work produced.** The work itself is already in version control; the reasoning behind it is not. "Implemented function X" and "drafted chapter 4" are noise for the same reason: the diff already says that.

A session is worth logging when it produced any of:

- a decision, a reversal, or an approach that was tried and discarded — and why
- evidence a future agent would otherwise have to rediscover: a measurement, a root cause, a constraint that turned out to be real
- a change to shared state: branch, PR, release, deployment, environment, or an external service
- unfinished follow-up that belongs in `next_steps.md`

Skip: casual Q&A, explanation-only exchanges that leave no lasting project state, routine changes the diff already explains, and one-off commands with no future coordination value.

Log experiments even when the change is reverted, as long as they leave reusable evidence, a decision, or a warning. Skip pure scratch work, and skip anything the user asks to keep off the book.

Each session entry records the model + version, effort/thinking mode, and token budget if known, and ends with a `- Verification:` subsection naming the commands actually run.

When a session creates or changes future work, update `next_steps.md` in the same pass: add concrete follow-ups, remove completed items, and keep "Currently Hot" accurate.

## Work Log Rotation And Dating

- The live `work_log.md` holds at most the 5 most recent unique calendar dates. Each file under `work_log_archive/` holds exactly 5.
- When prepending a new date would push the live log past 5 unique dates, move the oldest 5 dates as a chunk into `work_log_archive/work_log_<earliest>_to_<latest>.md`.
- If today's date already has a `## YYYY-MM-DD` header at the top, add a new `###` session subsection under it. Never open a second `## YYYY-MM-DD` header for the same date.
- Before writing any dated entry, verify the local date (`date +%F` on macOS/Linux, `Get-Date -Format yyyy-MM-dd` on Windows) and use that. When the model-context date and the local environment disagree — which happens across a UTC midnight boundary — trust the local date. Never write a future-dated entry; `treaty validate` fails with `work-log-future-date`.

## Reading Structured Docs

Find the anchors first, then read only the slice you need instead of loading whole files:

```
rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md
rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md work_log_archive/
grep -n '^## ' <any-structured-doc>
```

## Branch Handoff

Before switching away from an experimental or feature branch, confirm the branch holds all intended changes, that they are committed, and whether the user expects them merged, pushed, or intentionally parked.

Useful checks before switching or merging (portable git commands; run in any shell):

```
git status --short --branch
git log --oneline --left-right --cherry-pick dev...HEAD
git merge-base --is-ancestor dev HEAD
```

If related work accidentally lands on `dev`, move it back onto the feature branch first and retest the combined behavior there before updating `dev` again.

## Release Gate

Any request that combines **commit + push + tag** — or "cut a release" / "publish version X" — is a release. The documentation gate clears *before* the tag is created or pushed, not after.

Verify and update every release artifact that exists in this repo:

- Version metadata bumped and consistent with the tag [e.g. `pyproject.toml`, `package.json`, `Cargo.toml`].
- Changelog / release notes updated, if the repo has one [e.g. `CHANGELOG.md`, a release-notes doc].
- User-facing docs (`README.md`, setup / usage docs) updated when behavior changed.
- `work_log.md` updated with the implementation summary, the verification commands actually run, and the release / branch / tag state.
- Every dated artifact uses the verified local date (see [Work Log Rotation And Dating](#work-log-rotation-and-dating)).

Only after every applicable item is done: create the tag and push. Then confirm the pushed refs landed where you expect:

```
git ls-remote --tags origin
git ls-remote --heads origin dev
```

If a ref is missing or points at the wrong commit after the push, fix it before treating the release as complete.

## Updating The Treaty

Pulling upstream treaty refinements into this project is a maintainer's call — only do it when asked. This is distinct from updating `work_log.md` / `next_steps.md` content.

1. Commit or stash local changes first. `treaty update` requires a clean, git-tracked working tree and refuses to run on a dirty one.
2. `treaty diff` shows, section by section, how far this project has drifted from the template version it is pinned to — that is the conflict exposure before you touch anything.
3. `treaty update --dry-run` previews the merge without writing.
4. `treaty update` performs a **three-way merge**, so edits that do not overlap upstream changes are kept automatically. Where a local edit overlaps a changed region, it leaves conflict markers (`<<<<<<< before updating` / `=======` / `>>>>>>> after updating`) and an unmerged file, exactly like `git merge`. Resolve each one — keep your content, fold in the new template material — then `git add` the file.
5. Review `git diff`, confirm no conflict markers survive, then commit.

## Customizing These Docs

How much an edit will cost you at update time depends on *what* you edit, not how much:

- **Bracket placeholders** (`[path/to/entrypoint]`) are conflict-free by construction. Upstream will never ship a revision to them, so replacing them costs nothing later. Most of `project_overview.md` is this.
- **Maintained guidance** — the prose in this file — is actively revised upstream, so local edits here are the ones that turn into conflicts. Prefer adding to `AGENTS.md` over editing this file.
- **Renaming a heading is the worst single thing you can do.** A three-way merge reads it as a delete plus an unrelated add, so a later upstream edit to that section conflicts *and* cannot be auto-resolved. If you want different wording, change the body and leave the heading alone. `treaty diff` flags renamed headings specifically.
- **Deleting a section** is the next-worst: upstream edits to it arrive with nothing local to merge into. Where the treaty offers a question for a section you don't want (`has_releases`, `uses_precommit`, `include_git_ownership_note`), answer it instead of deleting — the section then never renders and never conflicts.
- **Adding a section** always merges cleanly. Domain-specific conventions belong in `AGENTS.md` as new sections.
