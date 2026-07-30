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

### Public GitHub Windows installation (GPT-5)

- Replaced the private SharePoint/OneDrive package path in `README.md` with the
  public GitHub Releases page. Windows users are now told to find the newest
  release containing `sleep_scoring_app_vX.Y-windows.zip`, avoid GitHub's
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
