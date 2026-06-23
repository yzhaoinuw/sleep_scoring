# Work Log

Prepend new session notes to the top of this file. If you log multiple
sessions on the same calendar date, add a new `###` subsection under the
existing `## YYYY-MM-DD` header instead of starting a second header for the
same date.

Historical verification commands may include absolute paths from the original
development machine. When replaying or adapting them, keep the project folder
name `sleep_scoring` and conda environment name `sleep_scoring_dash3.0`, but
replace the user/home prefix and clone location with the collaborator's local
setup.

Reading note for agents: this file holds at most the 5 most recent unique
calendar dates. Older entries are rotated in chunks of 5 dates into
`work_log_archive/work_log_<earliest>_to_<latest>.md`. Default to reading the
two most recent dated entries; search older entries with targeted terms using
the `^## [0-9]{4}-[0-9]{2}-[0-9]{2}` anchor, or open the relevant archive file
by its date range. See `AGENTS.md` for the full rotation policy.

## 2026-06-20

### Live-Log Rotation

- Rotated the previous five dates (2026-06-06 through 2026-06-15) into
  `work_log_archive/work_log_2026-06-06_to_2026-06-15.md` per the live-log
  size policy, so today's entry does not push the live log past five unique
  dates.

### JOSS Paper Draft (Fresh Pass On `publication`)

- Drafted `paper/paper.md` (~860 words) and `paper/paper.bib` from scratch in
  this session. Note for future agents: an earlier 2026-06-15 session already
  produced `paper/paper.md`, `paper/paper.bib`, and `CITATION.cff` on the
  `publication` branch (see the archived 2026-06-15 entry); the working tree
  at the start of this session did not contain `paper/` (the user was on
  `dev`, where `paper/` is gitignored), and `git checkout -b publication`
  succeeded as if the branch did not exist locally. The local `publication`
  branch built in this session therefore diverges from any prior
  `origin/publication`; reconciling the two is up to the maintainer. If a
  prior remote `publication` exists with the 2026-06-15 paper work, decide
  whether to keep that version, merge, or force-push the new draft before
  treating either as canonical.
- Statement of need is now explicitly scoped to the U19 BrainFlowZZZ research
  program: leads with Viewpoint (EEG/EMG) + TDT (NE photometry) hardware,
  the companion preprocessing pipeline that produces the fixed `.mat` field
  layout, and 1-second epoch scoring chosen to match NE photometry
  resolution. Generality is bounded honestly — the input format and epoch
  length are opinionated, and external adopters need recordings shaped to
  match the BrainFlowZZZ pipeline (or a thin adapter producing the same
  field layout).
- The contributions the paper does claim are (i) the interactive Dash/Plotly
  + `pywebview` annotation UI and (ii) the integrated behavior-video clip
  extraction synchronized to annotation selection. The sDREAMER scorer is
  framed throughout as an externally-developed upstream model integrated by
  the app, not as a contribution of this paper. Summary, Statement of need,
  and Implementation all carry this framing; a placeholder
  `@TODO_sdreamer_citation` BibTeX key marks where the real sDREAMER
  citation must be filled in before submission.

### Repo Hygiene For Public-Facing Push (On `dev`)

- Added root `LICENSE` (MIT) on `dev` so the repo carries an OSI-approved
  license alongside the upcoming JOSS submission. Easy to swap later to
  Apache-2.0 (adds an explicit patent grant) or BSD-3-Clause by replacing
  the body; `pyproject.toml` does not yet declare `license =` / `license-files
  =`, worth adding so wheels carry the metadata.
- Tightened `CONTRIBUTING.md` on `dev` (76 -> 60 lines) by removing content
  duplicated with `AGENTS.md` and the Agent Collab Treaty: dropped the
  "active path" pointer (already in `project_overview.md`), the
  `next_steps.md` / `work_log.md` split bullet (covered by treaty rotation
  policy + `AGENTS.md` doc map), the verbatim "Commits" body (already in
  `AGENTS.md` Commit Message Guidelines — replaced with a one-line pointer),
  and the generic "Collaboration Notes" section. Kept all Local
  Troubleshooting, Tests And Verification, Research Practices, Branches, and
  the project-specific Documentation bullets.
- Added `paper/` to `.gitignore` on `dev` so the JOSS draft can live on disk
  for editing across machines without leaking into `dev`-branch commits or
  PR diffs. Remove the line on `publication` when the paper is ready to be
  treated as canonical there.

### Sandbox Git Limitation Hit (Carryover)

- `git commit` on the `publication` branch failed inside the sandbox with
  `Operation not permitted` on `.git/index.lock` (same limitation already
  recorded in agent memory under `reference_sandbox_git_unlink`). The
  branch was created and `paper/paper.md` + `paper/paper.bib` were staged
  successfully; the commit and `git push -u origin publication` were
  handed off to the host. No verification of the push outcome was
  performed from inside the sandbox.

### Forward Pointers

- New `next_steps.md` Publication open items added today: replace the
  `@TODO_sdreamer_citation` placeholder; name external adopters for JOSS
  reviewer credibility once any exist; add an "Adapting input data"
  subsection to `README.md` documenting the `.mat` field contract for
  outside labs writing a thin converter.
