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

## 2026-07-30

### Release gates now enforce the CITATION.cff bullet (Claude)

- The 2026-07-29 session added `CITATION.cff` to the AGENTS.md version-metadata
  bullet after finding it five releases behind. Checking what enforces that
  bullet found nothing: `CITATION` appeared nowhere in `packaging/`, `.github/`,
  or `tests/`. `full_release.py` compared `app_src/__init__.py` against
  `setup.py` and required a `change_log.txt` heading, and stopped there. So the
  instruction gap was closed and the enforcement gap was not, which is how the
  drift reached five releases instead of one.
- This matters more now than it did before: Zenodo is enabled and scrapes
  `CITATION.cff` when a release is published, so a stale version is what gets
  published as the record's metadata.
- Correction, from review: an earlier draft of this entry and of the gate
  docstrings claimed the published metadata could not be fixed afterwards.
  That is wrong, and checked against Zenodo's own docs rather than conceded:
  "Metadata CAN be modified. Files and the persistent identifier CANNOT be
  modified", and editing metadata does not affect the DOI. The real cost is
  narrower — the correction is manual, on Zenodo, per record, and editing the
  repository file later does not propagate to records already published. The
  2026-07-29 entry below carries the same imprecision; it is left as written
  rather than rewritten, since it records what that session believed.
- Added `validate_citation_metadata` to both `full_release.py` and
  `lightweight_release.py`: `version` must equal the app version with `v`
  stripped, and `date-released` must parse and not be in the future. The
  version equality is the load-bearing half, since it forces the file to be
  edited every release; the date check only catches typos.
- Included the lightweight gate deliberately. Lightweight releases are still
  tags and Zenodo archives every published release, so a stale citation costs
  the same manual correction there. The file is not shipped inside the
  source-update asset, which carries only `app_src/`; it is the tagged repo
  state Zenodo reads.
- Duplicated the check across both scripts rather than sharing a module. The
  tests load each script standalone through `spec_from_file_location`, which
  does not put the package directory on `sys.path`, so a sibling import would
  not resolve. `VERSION_RE` and `SETUP_VERSION_RE` are already duplicated for
  the same reason; the docstring records this so it does not read as careless
  copy-paste.
- The first implementation pattern-matched the two lines. Review showed the
  quotes were independently optional, so `version: "0.17.0'` — an unterminated
  YAML scalar Zenodo could not read — matched and passed; verified that the old
  pattern did extract `0.17.0` from it. Matching two lines also could not see a
  document broken anywhere else. Now parsed with `yaml.safe_load`, which
  rejects both classes and removes the top-level-versus-`references:` key
  ambiguity entirely, since parsing returns the top-level mapping directly.
  Added malformed-input tests to both gate suites: mismatched quotes, a
  document broken away from these two keys, and an unquoted `date-released`
  that PyYAML resolves to a date object rather than a string.
- `pyyaml` declared in the `dev` extra, not the runtime dependencies, so it
  stays out of the packaged Windows app. It was already importable in the env
  as a transitive dependency; the gate should not rely on that silently.
- Fixing the fixtures exposed a test-integrity problem:
  `test_validate_candidate_rejects_forgotten_setup_version_bump` left
  `CITATION.cff` stale too, and the citation error fires first with a message
  that also matched its `does not match 'v0.17.1'` regex. It would have kept
  passing while no longer testing setup.py at all. That test now bumps the
  citation and matches a setup-specific message.
- Verified the real checkout passes its own new gate:
  `python packaging/windows/full_release.py --repo .` prints
  `v0.17.0	5eab40b...`.
- Verification: 173 passed, 2 skipped (8 new tests), Black clean,
  `git diff --check` clean.

### Full-package-only releases no longer look like failed updates (Claude)

- Pre-release review of the updater found a redirect-discovery gap. Redirect
  mode composes the asset filename from the release tag instead of reading a
  release asset list, so it always believes an asset exists. A newer release
  without a `sleep_scoring_app_update_<tag>.zip` therefore announced
  "updating from version X to version Y..." and then failed with HTTP 404,
  once per 24-hour check window, on a `console=True` window the user sees.
- This was reproduced against the live repo, not just in tests: v0.17.0 is
  full-package only, so a simulated 0.16.8 install reported
  `failed / could not download update asset: HTTP 404`. It is the state every
  installed app would have entered after the next full release.
- Fixed upstream in `desktop_app_source_updater` on branch
  `fix/redirect-missing-asset`, merged through PR #3: a composed asset URL is
  probed with a no-redirect HEAD before the update-available callback fires,
  and a definitive 404/410 reports `up-to-date` with "no matching source update
  asset". Also added `failure_retry_seconds` so a failed or offline check
  retries in an hour instead of a full day.
- App side: `format_startup_update_console_message` now recognizes the case
  where the check found a newer release it cannot install, names it, and prints
  the Releases URL, rather than saying "no update available". It compares
  versions instead of matching the updater's message text, because this
  launcher ships frozen and cannot be corrected by a source update.
- The message originally said "see the README installation steps". Review
  caught that the generated full package contains only `app_src/`, `models/`,
  the launcher, and `unblock_app.cmd` — no `README.md` — so a packaged user,
  the only user who ever sees this message, has nothing to open. It now carries
  the Releases URL. Used the public `LATEST_RELEASE_URL` constant rather than
  the configured check URL, which an env override can repoint at a test
  endpoint. Shipping the README inside the package was the alternative, but
  that changes the documented folder contract and the frozen fixture for no
  gain over a URL; noted as a possible follow-up instead.
- Verification: 26 launcher tests passed; upstream 41 tests passed (34 before
  this work), `compileall` and the builder `--help` gate passed.
- Upstream PR #3 merged to `dev` and `main` as `5eab40b`, including two review
  rounds on `UpdateConfig` field ordering. Bumped the `requirements.txt` pin
  from `85bb68e` to `5eab40b`, which clears the blocker noted above; a packaged
  build now carries the fix. `pyproject.toml` tracks `@main` and needed no
  change.
- Verified the pin rather than assuming it resolves: installed
  `git+...@5eab40b` into a throwaway venv (not the project env) exactly as a
  packaged build would, confirmed the installed module has `_asset_is_available`
  and the appended `failure_retry_seconds`, then re-ran the live production
  config against the real repo. A simulated 0.16.8 install now reports
  `up-to-date / release v0.17.0 has no matching source update asset` with no
  update-available callback, where the old pin reported
  `failed / HTTP 404` after announcing an update.
- Also confirmed `packaging/windows/full_release.py` parses the new pin, since
  that gate is what would reject a malformed SHA at release time.
- Verification at the final branch head: `pytest` collected 166, with 164
  passed and 2 skipped (both skips pre-existing). Smoke check passed,
  release-gate pin parse passed, Black clean, `git diff --check` clean.
  Earlier entries in this section quote lower counts because they were run
  before later commits added tests; each was accurate when written.

### Public GitHub Windows installation (GPT-5)

- Replaced the private SharePoint/OneDrive package path in `README.md` with the
  public GitHub Releases page. Windows users are now told to find the newest
  release containing `sleep_scoring_app_vX.Y_full.zip`, avoid GitHub's
  source-code archives, and install that full Windows base.
- Clarified that compatible code-only releases update automatically after the
  full package is installed, while dependency, model, launcher, or packaged
  runtime changes require downloading a newer full Windows release.
- Pointed the optional packaged sDREAMER setup to `torch.zip` on the same
  GitHub Release and replaced the OneDrive-specific runtime warning with a
  general warning against cloud-synced and network folders.
- Audited the README for duplicate installation/update instructions. It retains
  one packaged-install section and one automatic-update section; source
  installation updates and optional sDREAMER setup remain separate because
  they serve different workflows. Internal anchors, relative links, balanced
  details blocks, stale distribution references, and `git diff --check` all
  passed.

### Explicit full-package asset naming (GPT-5)

- Adopted `sleep_scoring_app_vX.Y_full.zip` as the full-package convention.
  The version remains immediately after the app name, `_full` makes the
  installable asset obvious, and the extracted folder remains
  `sleep_scoring_app_vX.Y` so later patch updates do not make its name stale.
- Updated the full builder, full-release gate, frozen v0.17.0 fixture, tests,
  public README, packaging documentation, and v0.17.0 change log. The
  `windows` filename suffix is no longer used.
- Preserved `sleep_scoring_app_update_vX.Y.Z.zip` for v0.17.x because the
  already distributed updater constructs that exact prefix-style name. Added
  a next-full-base task to extend the shared updater with configurable
  suffix-style discovery before adopting
  `sleep_scoring_app_vX.Y.Z_update.zip`.
- Renamed the retained and published v0.17.0 archive to
  `sleep_scoring_app_v0.17_full.zip` without rebuilding it. Its SHA-256 remains
  `D7F139F1390FFF2F036F6758043F711B11F8920015D489990649190B4268D461`.
  Corrected manifest and checksum sidecars were uploaded, the stale old-name
  sidecars were removed, and the GitHub Release notes now use the new name.
- Verification passed: repository-pinned Black, 29 focused packaging tests,
  all three edited PowerShell scripts parsed, local archive/manifest/checksum
  consistency passed, GitHub reported seven expected custom assets with no
  old-name copies, and every new remote digest matched its local file.

## 2026-07-29

### Citation metadata and sDREAMER attribution (Claude)

- Prepared `CITATION.cff` for a Zenodo archive DOI. Zenodo scrapes this file at
  archive time, so the stale fields would have become the permanent record
  metadata. Corrected `version` 0.16.2 to 0.17.0 and `date-released`
  2026-06-15 to 2026-07-29 (the v0.17.0 tag date), and added an `abstract` so
  the record carries a real description instead of the GitHub blurb.
- Resolved the co-author TODO as sole authorship. The application code is the
  repository author's; sDREAMER is a dependency relationship, not authorship,
  so the original authors are cited rather than listed.
- Added a `references:` block citing the original sDREAMER paper (Chen et al.,
  IEEE ICDH 2023, doi 10.1109/ICDH60066.2023.00028) and the `sdreamer_flow`
  training pipeline that produced the shipped checkpoints. Author list, venue,
  pages, and DOI were confirmed against Crossref, not transcribed by hand.
- Added a "Citing sDREAMER" README subsection with APA and BibTeX forms, a
  pointer from the sDREAMER setup section, and a provenance docstring in
  `models/sdreamer/__init__.py`. All three state that the code is adapted
  rather than copied and that the shipped checkpoints are not the original
  authors' weights.
- Added `CITATION.cff` to the version-metadata bullet of the AGENTS.md release
  gate. Its absence there is why the file drifted five releases behind.
- Verified: `cffconvert --validate` passes against CFF schema 1.2.0 (run from a
  scratch venv, not the project env), CITATION.cff `version` matches
  `app_src.VERSION`, Black clean, `tests/test_smoke.py` 8 passed.
- Added `NOTICE` recording third-party provenance for `models/sdreamer/`. NIH
  policy was checked directly rather than assumed: GPS 8.2.1 defines "data" to
  include software and permits copyrighting it without NIH approval, reserving
  only a federal-purposes license, so U19 funding does not by itself place the
  adapted code in the public domain or pick a license. NIH recommends
  OSI-approved permissive licenses, which makes MIT the compliant choice, and
  NIH explicitly defers ownership questions to the institution. UR's IP policy
  covers software and requires an IP agreement from everyone in sponsored
  research, so copyright most likely sits with the University rather than with
  the departed original developers.
- Merged `publication` into `dev` to consolidate the paper draft. Resolving the
  merge recovered the 2026-06-20 work-log entry, which existed only on
  `publication` and had fallen through the archive rotation; it is now at
  `work_log_archive/work_log_2026-06-20_to_2026-06-20.md`. That entry documents
  who drafted `paper/paper.bib` and is the provenance for the fabricated
  bibliography entries found below.
- Checked every `paper.bib` entry carrying a DOI against Crossref. Six verified
  clean. Three did not: the sDREAMER placeholder, the somnotate entry (keyed to
  the SPINDLE first author, wrong year, no metadata), and the AccuSleep entry,
  whose title did not match its DOI. Fixed on `publication` and routed to `dev`
  through PR #11 rather than committed directly, keeping the deferred JOSS work
  behind review.
- Open follow-ups: confirm with the PI and UR Ventures that `models/sdreamer/`
  may be redistributed under MIT, and fill the grant-number TODO in `NOTICE`.
  Zenodo still needs the repository enabled and a release cut afterward; the
  webhook does not archive releases retroactively, so v0.17.0 will not be
  picked up.

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

### v0.17.0 publication and Schema 2 launchpad (GPT-5)

- Rebuilt the full Windows candidate from final release commit
  `d25414d2f31dc9a6eec0034a54c587f77a027841`. The complete gate passed with
  158 Python tests, repository-pinned Black, compileall, 38 JavaScript tests,
  source smoke, PyInstaller, packaged structure/config smoke, packaged
  executable smoke, and a forced update check.
- Fast-forwarded `main`, created and pushed annotated tag `v0.17.0`, and
  published the GitHub Release with all seven custom release assets. Local
  `main`/`dev`, remote `main`/`dev`, and the peeled tag matched the release
  commit. The published full ZIP SHA-256 was
  `D7F139F1390FFF2F036F6758043F711B11F8920015D489990649190B4268D461`;
  the optional Torch runtime SHA-256 was
  `DC9EDC03EFB98E9026F06B0F86E9E2514E71382EA01C96323D51EBE9A4A10C06`.
  GitHub's reported digests matched both local artifacts.
- Exported the published v0.17.0 full ZIP as the new compact installed
  baseline with exact hashes for all 34 packaged `app_src` files. Future
  lightweight candidates and frozen-app fixtures now begin at v0.17.0.
- Enabled post-v0.17 Schema 2 configuration merging. A source update remains
  Schema 1 when `app_src/config.py` is unchanged; when it changes, every
  compatible installation must be v0.17.0 or newer and the builder declares
  the approved 12 user-facing literal assignments. Runtime-derived and
  profiling assignments remain authoritative from the downloaded template.
- The release helper now customizes a freshly extracted config before applying
  an update, verifies the exact recursive merge result, checks the installed
  version, and runs the frozen executable smoke test. The repository-pinned
  Black hook, all 163 Python tests, and both PowerShell parsers passed.
- An isolated v0.17.1 trial built a real three-file Schema 2 asset through the
  shared updater, including a new nested `WINDOW_CONFIG` default and an
  ordinary replacement file. The frozen published v0.17.0 executable applied
  it successfully, preserved the customized config values, advanced to
  v0.17.1, and passed smoke. The disposable worktree and artifact were removed.
- After reviewing the branch policy, the maintainer chose to advance `main`
  with the completed Schema 2 release tooling rather than keep it exclusively
  on `dev`. The published `v0.17.0` tag and artifacts remain tied to the
  original release commit; `main` now represents the latest tested stable code.
