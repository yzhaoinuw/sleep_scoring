import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "full_release.py"
SPEC = importlib.util.spec_from_file_location("full_release", MODULE_PATH)
FULL_RELEASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FULL_RELEASE)
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")
SMOKE_SCRIPT = MODULE_PATH.with_name("smoke_check_release.ps1")
FULL_RELEASE_SCRIPT = MODULE_PATH.with_name("release_full.ps1")
FULL_PACKAGE_SCRIPT = MODULE_PATH.with_name("make_full_app_zip.ps1")


def _write_candidate(
    tmp_path,
    *,
    app_version="v0.17.0",
    setup_version="0.17.0",
    citation_version="0.17.0",
    citation_date="2026-07-29",
):
    app_src = tmp_path / "app_src"
    app_src.mkdir()
    (app_src / "__init__.py").write_text(f'VERSION = "{app_version}"\n', encoding="utf-8")
    (app_src / "config.py").write_text(
        'STAGE_COLORS = ["blue", "red", "green", "yellow"]\n',
        encoding="utf-8",
    )
    (tmp_path / "setup.py").write_text(
        f'setup(\n    version="{setup_version}",\n)\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"## {app_version}\n\n- Candidate.\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "desktop-app-source-updater @ "
        "git+https://github.com/yzhaoinuw/desktop_app_source_updater.git@" + ("a" * 40) + "\n",
        encoding="utf-8",
    )
    # A cited work carries its own indented version/date keys; the validator
    # must read the top-level ones describing this release.
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        'title: "sleep_scoring"\n'
        f"version: {citation_version}\n"
        f'date-released: "{citation_date}"\n'
        "references:\n"
        "  - type: article\n"
        '    title: "An upstream model"\n'
        "    version: 1.2.3\n"
        '    date-released: "2023-01-01"\n',
        encoding="utf-8",
    )


def test_full_candidate_validation_returns_version_and_updater_pin(tmp_path):
    _write_candidate(tmp_path)

    assert FULL_RELEASE.validate_full_candidate(tmp_path) == (
        "v0.17.0",
        "a" * 40,
    )


def test_full_candidate_requires_aligned_setup_version(tmp_path):
    _write_candidate(tmp_path, setup_version="0.16.9")

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="does not match"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_requires_current_citation_version(tmp_path):
    # The drift that actually happened: CITATION.cff left five releases behind
    # while the app moved on. Zenodo scrapes this file at publish time, so the
    # stale value is what gets published, and fixing the repository afterwards
    # does not reach records already published.
    _write_candidate(tmp_path, citation_version="0.16.2")

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="CITATION.cff version"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_rejects_future_release_date(tmp_path):
    _write_candidate(tmp_path, citation_date="2099-01-01")

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="in the future"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_rejects_unparseable_release_date(tmp_path):
    _write_candidate(tmp_path, citation_date="2026-13-45")

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="not a valid date"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_rejects_mismatched_citation_quotes(tmp_path):
    # An unterminated scalar is a YAML error, so Zenodo could not read this
    # file. A pattern with independently optional quotes accepted it.
    _write_candidate(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "0.17.0\'\ndate-released: "2026-07-29"\n',
        encoding="utf-8",
    )

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="not valid YAML"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_rejects_citation_broken_elsewhere_in_the_document(tmp_path):
    # The version and date lines are both correct here; the file is still
    # unusable, which pattern matching those two lines could not detect.
    _write_candidate(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "version: 0.17.0\n"
        'date-released: "2026-07-29"\n'
        "authors: [ {given-names: Yue\n",
        encoding="utf-8",
    )

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="not valid YAML"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_accepts_an_unquoted_release_date(tmp_path):
    # PyYAML resolves this to a date object rather than a string.
    _write_candidate(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nversion: 0.17.0\ndate-released: 2026-07-29\n",
        encoding="utf-8",
    )

    version, _ = FULL_RELEASE.validate_full_candidate(tmp_path)

    assert version == "v0.17.0"


def test_citation_check_reads_top_level_keys_not_cited_works(tmp_path):
    # The fixture's `references:` block carries version 1.2.3 and a 2023 date.
    # Matching either of those instead of the top-level keys would make the
    # check silently meaningless, so prove the happy path still validates.
    _write_candidate(tmp_path)

    version, _ = FULL_RELEASE.validate_full_candidate(tmp_path)

    assert version == "v0.17.0"


def test_full_candidate_requires_changelog_heading(tmp_path):
    _write_candidate(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "## v0.16.9\n\n- Previous.\n",
        encoding="utf-8",
    )

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="release heading"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_requires_customizable_stage_colors(tmp_path):
    _write_candidate(tmp_path)
    (tmp_path / "app_src" / "config.py").write_text(
        'SLEEP_SCORING_MODEL = "stats_model"\n',
        encoding="utf-8",
    )

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="STAGE_COLORS"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_candidate_requires_immutable_updater_pin(tmp_path):
    _write_candidate(tmp_path)
    (tmp_path / "requirements.txt").write_text(
        "desktop-app-source-updater @ "
        "git+https://github.com/yzhaoinuw/desktop_app_source_updater.git@main\n",
        encoding="utf-8",
    )

    with pytest.raises(FULL_RELEASE.FullReleaseError, match="exactly one"):
        FULL_RELEASE.validate_full_candidate(tmp_path)


def test_full_release_gate_does_not_repeat_source_tests_in_builder():
    source = FULL_RELEASE_SCRIPT.read_text(encoding="utf-8")

    assert source.count('"pytest",') == 1
    assert "SkipTests = $true" in source
    assert "AllowDirty = $true" in source
    assert '"full_release_" + [guid]::NewGuid()' in source
    assert "release_artifacts\\sleep_scoring_app_$($ReleaseLine)_full.zip" in source


def test_full_package_builder_moves_exact_version_stage_to_release_line():
    source = FULL_PACKAGE_SCRIPT.read_text(encoding="utf-8")

    assert "[int]$PackagedUpdateCheckTimeoutSeconds = 30" in source
    assert '$PyInstallerDistName = "sleep_scoring_app_$Version"' in source
    assert '$DistName = "sleep_scoring_app_$ReleaseLine"' in source
    assert '$ZipPath = Join-Path $ArtifactDir "$($DistName)_full.zip"' in source
    assert "subprocess.run(sys.argv[2:], check=True, timeout=int(sys.argv[1]))" in source
    assert "$PackagedUpdateCheckTimeoutSeconds.ToString()" in source
    assert '"SLEEP_SCORING_UPDATE_STATE_FILE"' in source
    assert '"packaged_update_check_" + [guid]::NewGuid()' in source
    assert "Remove-Item -LiteralPath $UpdateCheckStateDir -Recurse -Force" in source
    assert "Move-Item -LiteralPath $PyInstallerDistPath -Destination $DistPath" in source
    assert source.index("Move-Item -LiteralPath $PyInstallerDistPath") < source.index(
        '$BundledTorchDir = Join-Path $DistPath "_internal\\torch"'
    )


def _write_full_release_fixture(tmp_path, *, include_stage_colors):
    for relative_path in (
        "_internal",
        "models/sdreamer/checkpoints",
        "app_src/assets",
    ):
        (tmp_path / relative_path).mkdir(parents=True, exist_ok=True)
    for relative_path in (
        "run_desktop_app.exe",
        "unblock_app.cmd",
        "models/sdreamer/checkpoints/model.pt",
        "app_src/__init__.py",
        "app_src/app.py",
        "app_src/assets/app.js",
    ):
        (tmp_path / relative_path).write_text("fixture\n", encoding="utf-8")
    config = 'STAGE_COLORS = ["blue", "red", "green", "yellow"]\n'
    if not include_stage_colors:
        config = 'SLEEP_SCORING_MODEL = "stats_model"\n'
    (tmp_path / "app_src" / "config.py").write_text(config, encoding="utf-8")


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is unavailable")
def test_full_package_smoke_requires_stage_colors(tmp_path):
    _write_full_release_fixture(tmp_path, include_stage_colors=False)

    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SMOKE_SCRIPT),
            "-Path",
            str(tmp_path),
            "-Kind",
            "Full",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing expected STAGE_COLORS assignment" in (result.stdout + result.stderr)


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is unavailable")
def test_full_package_smoke_accepts_stage_colors(tmp_path):
    _write_full_release_fixture(tmp_path, include_stage_colors=True)

    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SMOKE_SCRIPT),
            "-Path",
            str(tmp_path),
            "-Kind",
            "Full",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
