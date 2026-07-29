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

## 2026-07-29

### v0.17.0 full-release verification (GPT-5)

- Prepared v0.17.0 as the full Windows compatibility boundary for the
  rate-limit-safe shared updater, exact startup version reporting, stable
  v0.17 package-folder naming, and the canonical configurable sleep-stage
  colors.
- The first package attempt exposed and fixed an exact-version PyInstaller
  staging-folder mismatch before ZIP creation. The corrected candidate at
  `06d31916b8174e611d1e41b10d5d4dbfa12a3108` passed the standard full-release
  gate: 158 Python tests, repository-pinned Black, compileall, 38 JavaScript
  tests, source smoke, PyInstaller, full-package structure/config smoke,
  packaged executable smoke, and a forced live update check.
- The verified full ZIP used the single top-level
  `sleep_scoring_app_v0.17` folder, embedded exact version v0.17.0 and
  `STAGE_COLORS`, and had SHA-256
  `3E594456A08364A267D34DB5593CEE4C6A3B95C080A1AFF608C16429D8B75E43`.
  The optional Torch 2.9.1 CPU runtime had SHA-256
  `CB283F8F1EEF782F44471F1AFC0A42A6A302B3F7F77D09570018D97AB07CB12E`.
- Local `dev`, `origin/dev`, and remote `dev` matched that candidate, and all
  CI plus CodeQL jobs passed. `main` remained at the published v0.16.8 line;
  no v0.17.0 tag or GitHub Release existed when the user authorized the final
  fast-forward, tag, and publication.
- This release record is the only tracked change after the verified candidate.
  The full gate must therefore rebuild once from its final commit before
  `main`, the v0.17.0 tag, and the published artifacts are advanced together.
