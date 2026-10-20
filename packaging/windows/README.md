# Windows Packaging

These scripts keep the current zip-based user workflow, while making the build
steps repeatable and easier to audit.

## Release Naming

Use the version alone as the GitHub release title, exactly matching its tag.
This is the project convention for both full packages and compatible updates.

| Item | Convention | Example |
| --- | --- | --- |
| Git tag | Exact app version, including `v` | `v0.17.6` |
| GitHub release title | Identical to the tag | `v0.17.6` |
| Release notes | App name/version, then the matching changelog entry | `Sleep Scoring App v0.17.6` followed by its user-facing changes |
| Full Windows ZIP | App name, major/minor line, and `_full` suffix | `sleep_scoring_app_v0.17_full.zip` |
| Automatic update ZIP | Existing updater-compatible name and exact version | `sleep_scoring_app_update_v0.17.6.zip` |

Keep descriptive subtitles, codenames, and full/partial labels out of the release
title. Describe changes, compatibility, and installation in the release notes.
The download filenames retain the app name so users can identify files outside
GitHub. Keep the current automatic-update filename pattern: it is part of the
installed updater's discovery contract. Full-ZIP and checksum details are
documented below.

When publishing with `gh release create`, explicitly pass the same version to
the tag argument and `--title`, using `--title $Version` even when generating
release notes automatically. Verify `tagName` and `name` agree with
`gh release view $Version --json tagName,name` after publishing.

This naming rule concerns GitHub releases. Software titles in `CITATION.cff`
and Zenodo remain descriptive citation metadata. Existing release-title cleanup
is separate from creating or moving tags or renaming assets.

## Release Documentation

`CHANGELOG.md` is the authority for user-facing release changes, including
material limitations. Copy the matching version's entry into the GitHub notes;
do not add manuscript revisions, internal refactoring, branch history, or a
second summary that repeats the same changes. Maintainer evidence belongs in
`treaty_docs/work_log.md`.

Keep the notes short: app name/version and changelog text, then **Installation**
(existing users and new users) and **Verification** (one sentence stating the
gate and important packaged-app checks). Link directly to the full Windows ZIP.
Keep meaningful limitations; leave test counts, commands, and detailed evidence
in the work log. Do not use GitHub's automatically generated commit list.

### Prepare the Windows download

The README has one actual full-package download URL. Its prominent top link
points to the installation section. `release_metadata.json` records the current
full-package tag; the ZIP name is derived using `package_naming.py`.

For a **new full release**, first align version/citation metadata and add the
user-facing changelog entry, then activate the test environment and run:

```powershell
conda activate sleep_scoring_dash3.0
python packaging/windows/release_docs.py prepare-full
```

This updates the README link and full-package metadata together. Review and
commit them with the candidate before the full gate. The gate rejects stale
metadata or a stale/duplicate README download link, even with quality checks
skipped. The URL need not exist before publication; this check is local.

For a **compatible update**, leave the full-package tag/link unchanged. The
lightweight gate checks that the advertised full base is among the supported
versions. Continue maintaining the installed baselines and fixtures required by
the packaging gate when changing the full base.

### Generate and publish notes

Both gates generate `release_artifacts/release_notes_vX.Y.Z.md` after their
checks. This is ignored local output beside the ZIPs, not committed content.
The lightweight gate reads documentation from `-ToRef`, matching the code being
shipped. Full validation-only runs produce preflight notes; skipped checks
produce development notes, never a claim that the complete release gate passed.
A gate removes an older same-version notes file before its documentation check.

Review the generated file, then pass it explicitly when publishing:

```powershell
gh release create $Version --title $Version --notes-file "release_artifacts/release_notes_$Version.md" <assets>
```

`<assets>` stands for the appropriate full-package or compatible-update assets.
Publishing/tagging remains an explicit maintainer action. After publication,
verify the title/tag, expected assets, and the README's full-package download
with an HTTP HEAD request that follows redirects. Continue the Zenodo/ref checks
in the project release checklist.

To preview notes without running a release gate, use `release_docs.py notes
--kind full` (after preparing a full candidate), or `notes --kind update
--from-ref <supported-tag>` repeated for every supported version. Preview notes
are marked unverified by default. Only the gates should supply `--verification`.

## Full App Zip

Use this when dependencies, `_internal/`, `run_desktop_app.exe`, `models/`, the
PyInstaller runtime layout, or the bundled auto-updater changed.

The default full app zip intentionally removes Torch, which is the largest
runtime dependency. It keeps the sDREAMER code and checkpoint files, so users
who need sDREAMER can enable it by copying the optional sDREAMER Torch runtime
contents directly into `_internal/`.

This full zip is still the file to share with new Windows users. The generated
`build_env_requirements` sidecar is for release/debugging records, not a user
install step.

The generated app folder includes `unblock_app.cmd`, a double-click starter
that contains the unblock step and then launches `run_desktop_app.exe`. It is
included only in the full app zip, not in the small `app_src` update zip.

Use the standard one-command gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_full.ps1
```

During development, run the same preflight and quality checks without invoking
PyInstaller:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_full.ps1 -ValidateOnly -AllowDirty
```

The default build environment is `sleep_scoring_dash3.0_dist`; the default test
environment is `sleep_scoring_dash3.0`.

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_vX.Y_full.zip
sleep_scoring_app_vX.Y_full.zip.manifest.json
sleep_scoring_app_vX.Y_full.zip.sha256.txt
sleep_scoring_app_vX.Y_full.zip.build_env_requirements.txt
torch.zip
torch.zip.manifest.json
torch.zip.sha256.txt
```

The `_full.zip` suffix distinguishes the installable package from automatic
source-update assets. The version token in the full ZIP and its extracted
top-level folder use only the major/minor release line, for example
`sleep_scoring_app_v0.17_full.zip` and `sleep_scoring_app_v0.17`. The exact
patch version, such as `v0.17.0`, remains authoritative in the manifest,
startup terminal message, and app window title. This prevents an installation
folder name from becoming a stale version report after source-only updates.

Before creating the zip, the script checks that the release folder contains the
expected files, including the double-click starter. It runs
`run_desktop_app.exe --smoke` to verify that the built launcher can import the
side-by-side `app_src/` folder, verifies that the packaged config exposes
`STAGE_COLORS`, then runs `run_desktop_app.exe --check-update` in forced-check
mode with an isolated temporary state file and a 30-second process deadline.
The isolated state keeps build sandboxes from redirecting or denying the
packaged app's normal `%LOCALAPPDATA%` write. A discovery failure, updater
failure, or hung packaged check stops the package build instead of shipping a
broken automatic update check. The build also refuses to run when the updater
installed in the PyInstaller environment does not match the exact commit
pinned in `requirements.txt`.
The packaged `app_src/` files are written directly from the release commit's
Git blobs, without checkout or archive transformations, so their bytes match
the automatic-update manifests on Windows as well as source runs.

`release_full.ps1` is the complete candidate gate. Do not rerun its individual
pytest, Black, JavaScript, compile, source-smoke, build, packaged-smoke, updater,
or checksum checks by hand after it passes unless the candidate commit changes.
The pushed candidate must still pass CI, but that confirmation does not require
repeating the same local checks. Tagging and GitHub publication remain one
explicit maintainer step after reviewing the artifacts and release notes.

## Optional sDREAMER Torch Runtime Zip

`make_full_app_zip.ps1` builds the app with Torch available so PyInstaller can
discover Torch, TorchVision, and related hidden imports. The script then creates
the optional runtime zip from the built `_internal\torch` folder and removes
that folder before zipping the main app. This keeps the full app zip smaller
without losing imports such as `cProfile` that Torch loads later.

The generated runtime zip does not contain an `_internal/` folder itself. Users
copy its contents directly into the app's existing `_internal/` folder. After
copying, `_internal\torch` should exist.

## Automatic Source Update Asset

Use this for future code-only releases when changes are only in `app_src/` and
the installed full app already includes the auto-updater. Attach the generated
zip to the matching GitHub Release; users do not unzip it manually.

For a release candidate, use the one-command gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\release_lightweight.ps1
```

It discovers every compatible tag from v0.17.1 through the release immediately
before the target version. It then validates the lightweight-release boundary,
runs pytest, Black, JavaScript, compile, and source smoke checks once, builds
and validates the update asset, and tests it against:

- a fresh v0.17.1 Windows package containing the schema-2-capable updater; and
- user-customized values in the approved `config.py` assignments.

The retained full-package ZIPs are fixture inputs in the ignored
`release_artifacts/` folder. Exact installed hashes are tracked compactly under
`installed_baselines/`, so normal asset construction no longer needs to open
the large package ZIPs just to align line-ending variants.

When a future full Windows base is introduced, create its compact baseline with
`lightweight_release.py export-baseline`, review the generated JSON, and update
the fixture policy in `release_lightweight.ps1` in the same change. Retain each
full-package fixture under a version-specific folder, for example:

```text
release_artifacts/fixtures/v0.17.0/sleep_scoring_app_v0.17_full.zip
release_artifacts/fixtures/v0.17.1/sleep_scoring_app_v0.17_full.zip
```

Use `-FixtureArtifactDir` when the version-specific fixture folders live
somewhere other than `release_artifacts/fixtures/`.

The command allows `setup.py` only when its version assignment is the sole
change and it matches `app_src/__init__.py`. It refuses dependency, model,
launcher/updater, PyInstaller, release-helper, runtime deletion, and runtime
rename changes. Those require a new full Windows package.

An update that does not change `app_src/config.py` remains manifest schema 1.
When `config.py` changes, the builder emits schema 2 and requires every
compatible installed version to be v0.17.0 or newer. The downloaded file is the
authoritative template, while these user-facing literal assignments retain
installed values:

- `WINDOW_CONFIG`
- `FIX_NE_Y_RANGE`
- `STAGE_COLORS`
- `SPECTROGRAM_COLORSCALE`
- `THETA_DELTA_RATIO_LINE_COLOR`
- `THETA_DELTA_RATIO_LINE_OPACITY`
- `GAUSSIAN_FILTER_SIGMA`
- `POSTPROCESS`
- `SLEEP_SCORING_MODEL`
- `STATS_MODEL_WAKE_THRESHOLD`
- `STATS_MODEL_MIN_WAKE_DURATION`
- `STATS_MODEL_MIN_REM_DURATION`
- `STATS_MODEL_REM_THRESHOLD_PERCENTILE`
- `STATS_MODEL_REM_THRESHOLD_COMPARISON_PERCENTILE`
- `ENABLE_USAGE_REPORTING`

For `WINDOW_CONFIG`, dictionary values merge recursively: existing keys keep the
installed value, newly introduced keys use the downloaded default, and keys
removed from the downloaded template stay removed. Imports, functions,
comments, ordering, and non-user-facing assignments come from the downloaded
template. Invalid or nonliteral editable values stop the update before any file
is changed.

Output goes to `release_artifacts/`:

```text
sleep_scoring_app_update_vX.Y.Z.zip
sleep_scoring_app_update_vX.Y.Z.zip.sha256.txt
```

`make_source_update_asset.ps1` remains the lower-level builder for focused
packaging work. Pass `-FromRef` more than once for jump-ahead support and
`-InstalledBaselineManifest` more than once for compact package byte lineages.
`-FromPackageZip` remains available as a compatibility fallback.

Neither command publishes a GitHub Release automatically. Tagging and
publication remain an explicit maintainer action after reviewing the candidate
artifact and release notes.

## Manual app_src Update Zip

`make_app_update_zip.ps1` remains a fallback for manually replacing `app_src/`
when needed, but the automatic source update asset is the preferred code-only
release path for auto-update-enabled builds.

## Local Test Builds

Both scripts normally require a clean worktree so release artifacts can be tied
to a commit. For local testing before committing, pass `-AllowDirty`. To skip
tests during lower-level packaging-script development, pass `-SkipTests`.
`release_lightweight.ps1` also has `-SkipQualityChecks` and
`-SkipInstalledAppTests`; do not use either switch for a published release.
