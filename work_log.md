# Work Log

Prepend new session notes to this file. Record project decisions, reusable
evidence, and shared-state changes—not routine content already explained by the
diff. Each session ends with a `- Verification:` subsection. See
[`treaty_conventions.md`](treaty_conventions.md#work-log-discipline).

If today's date is already at the top, add a new `###` subsection beneath it.
The live log holds at most five unique dates; when a new date would exceed that
limit, move the oldest five-date chunk to
`work_log_archive/work_log_<earliest>_to_<latest>.md`.

Historical commands can contain machine-specific paths. When replaying them,
keep the `sleep_scoring` folder and `sleep_scoring_dash3.0` environment names
but adapt the user prefix and clone location. Default to the two newest dates;
search older entries by date anchor rather than reading every archive.

## 2026-08-05

### Triage the open Dependabot alerts (Claude Opus 5)

- Six open alerts, none reachable in the shipped app. Split them by whether a
  fix needs a release: `black`, `pytest`, and `brace-expansion` are dev/test
  only, sit outside `[project.dependencies]`, and are never bundled by
  `packaging/windows/app.spec`, so they land as an ordinary `dev` PR with no
  version bump, citation alignment, packaging gate, or Zenodo step. Only
  `torch` is a runtime pin, and that one is deferred.
- Reachability evidence, so this is not re-derived next time. The `black`
  advisory (GHSA-3936-cmfr-pm3m, "high") needs an attacker-controlled
  `--python-cell-magics` value; the hook in `.pre-commit-config.yaml` passes no
  such flag. `brace-expansion` (GHSA-3jxr-9vmj-r5cp) is transitive under
  `jest ^30` in `tests/js` and only expands our own test globs. The `pytest`
  advisory (GHSA-6w46-j5rx-g56g) is a `/tmp/pytest-of-{user}` race on
  multi-user UNIX, and the documented invocation already passes `--basetemp`.
  Both `torch` advisories are local memory corruption in `torch.lstm_cell` and
  `torch.jit.script`; neither function appears anywhere in the tree.
- Deferred `torch` 2.9.1 rather than bumping it. Clearing both alerts requires
  2.13.0, because GHSA-rrmf-rvhw-rf47 is patched only there while 2.10.0 clears
  just GHSA-qfhq-4f3w-5fph. The pin is duplicated across `requirements.txt`,
  the `ml` extra in `pyproject.toml`, and the README's sDREAMER instructions,
  and `make_full_app_zip.ps1:99` stamps the version into the `torch.zip`
  manifest — so any bump forces a full package plus a rebuilt optional runtime,
  and risks sDREAMER/timm compatibility. Two unreachable lows do not justify
  that on their own; folded into the next full release instead. Recorded in
  `next_steps.md`.
- Verified the black 26 stable style before adopting it: the whole `--diff`
  across the repo was 195 lines with zero non-blank added or removed lines. The
  new style only drops one redundant blank line in 14 files, so the reformat is
  textually whitespace-only and was kept in its own commit.
- `.github/dependabot.yml` does not exist, so the repository receives alerts but
  no automated fix PRs. Adding one would let Dependabot carry this kind of
  dev-dependency churn by itself; left as a follow-up in `next_steps.md`.
- Branch `security-dep-bumps` off `dev`, two commits, not yet pushed: the
  pytest/lockfile bump, then the black bump with its reformat.
- Verification:
  - `gh api repos/:owner/:repo/dependabot/alerts --paginate` for the alert set,
    scopes, and manifests; `gh api /advisories/<ghsa>` for each advisory.
  - `npm audit` before and after `npm audit fix` in `tests/js`: 1 high to 0
    vulnerabilities, `package.json` untouched and `jest` not bumped;
    `brace-expansion` 2.1.1 to 2.1.4 and 1.1.15 to 1.1.18.
  - `npm test` in `tests/js`: 38 passed.
  - `python -m pytest --basetemp .pytest_tmp/claude -p no:cacheprovider -q`
    under pytest 9.1.1, before and after the reformat: 173 passed, 2 skipped.
  - `python run_desktop_app.py --smoke` with `SLEEP_SCORING_SKIP_UPDATE=1`:
    "Sleep Scoring App v0.17.1 smoke check OK".
  - `black --check .` after formatting: 54 files unchanged. Note for this Mac
    checkout: `pre-commit` is not installed in `sleep_scoring_dash3.0`, so black
    was run directly at the pinned 26.5.1 rather than through the hook. Rerun
    the documented `pre-commit run black --all-files` on Windows before merge.
