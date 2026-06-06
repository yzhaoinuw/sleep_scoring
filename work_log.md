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

## 2026-06-06

### Installation Packaging Docs

- User manually confirmed the generated no-Torch full app zip can be unzipped and launched on Windows.
- Updated `README.md` so Windows installation and Automatic Sleep Scoring explain the current distribution model: the app zip includes sDREAMER code/checkpoints but not the optional Torch runtime, and users who need automatic scoring place the unzipped `torch` folder directly inside `_internal/`.
- Updated `packaging/windows/README.md` to clarify that the full app zip is still the file shared with new Windows users, while generated build-env requirement snapshots are release/debugging records.
- Updated `next_steps.md` to mark the full app zip manual launch as validated and to use `app_src` update wording consistently.
- Added `.pytest_tmp` parent-directory creation inside both packaging scripts so clean builds can run the repo-local pytest basetemp path on Windows.
- Rotated the previous five live work-log dates into `work_log_archive/work_log_2026-05-25_to_2026-06-05.md` per the live-log size policy.
