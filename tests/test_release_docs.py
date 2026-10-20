import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "packaging/windows/release_docs.py"
SPEC = importlib.util.spec_from_file_location("release_docs", SCRIPT)
DOCS = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(DOCS)


def candidate(repo, version="v0.17.6", base="v0.17.1"):
    (repo / "app_src").mkdir()
    (repo / "app_src/__init__.py").write_text(f'VERSION = "{version}"\n')
    (repo / "packaging/windows").mkdir(parents=True)
    (repo / DOCS.METADATA).write_text(json.dumps({"full_package_version": base}))
    (repo / "README.md").write_text(f"[Download the Windows app]({DOCS.download_url(base)})\n")
    (repo / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## {version}\n\n- Current user change.\n"
        "  Continued detail.\n\nCalibration still needs evaluation.\n\n"
        "## v0.16.0\n\n- Old change.\n"
    )


def test_full_release_requires_prepare_and_patch_keeps_full_base(tmp_path):
    candidate(tmp_path)
    original = (tmp_path / "README.md").read_bytes()
    DOCS.validate_docs(tmp_path, "update", from_versions=("v0.17.1", "v0.17.5"))
    assert (tmp_path / "README.md").read_bytes() == original
    with pytest.raises(DOCS.ReleaseDocsError, match="prepare-full"):
        DOCS.validate_docs(tmp_path, "full")
    DOCS.prepare_full(tmp_path)
    assert DOCS.validate_docs(tmp_path, "full")[:2] == ("v0.17.6", "v0.17.6")
    assert (
        "download/v0.17.6/sleep_scoring_app_v0.17_full.zip" in (tmp_path / "README.md").read_text()
    )


@pytest.mark.parametrize("problem", ["stale", "duplicate"])
def test_rejects_stale_or_duplicate_downloads(tmp_path, problem):
    candidate(tmp_path)
    readme = tmp_path / "README.md"
    text = readme.read_text()
    readme.write_text(text.replace("v0.17.1/", "v0.17.0/") if problem == "stale" else text * 2)
    with pytest.raises(DOCS.ReleaseDocsError, match="README.md"):
        DOCS.validate_docs(tmp_path, "update", from_versions=("v0.17.1",))


@pytest.mark.parametrize(
    "versions", [(), ("v0.17.5",), ("v0.17.1", "v0.16.9"), ("v0.17.1", "v0.17.6")]
)
def test_rejects_unsupported_update_base(tmp_path, versions):
    candidate(tmp_path)
    with pytest.raises(DOCS.ReleaseDocsError, match="Compatible versions"):
        DOCS.validate_docs(tmp_path, "update", from_versions=versions)


@pytest.mark.parametrize("entry", ["", "## v0.17.6\n\n", "## v0.17.6\n- A\n## v0.17.6\n- B\n"])
def test_rejects_missing_empty_or_duplicate_changelog_entry(entry):
    with pytest.raises(DOCS.ReleaseDocsError, match="CHANGELOG.md"):
        DOCS.changelog_entry(entry, "v0.17.6")


def test_notes_copy_exact_entry_and_material_limitations(tmp_path):
    candidate(tmp_path)
    versions = tuple(f"v0.17.{patch}" for patch in range(1, 6))
    version, base, entry = DOCS.validate_docs(tmp_path, "update", from_versions=versions)
    notes = DOCS.render_notes(version, base, entry, "update", versions, "complete")
    assert f"\n\n{entry}\n\n## Installation" in notes
    assert "Old change" not in notes
    assert "Calibration still needs evaluation." in notes
    assert "Existing v0.17.1–v0.17.5 installations" in notes
    assert "fresh packaged-install update" in notes


def test_noncontiguous_compatibility_does_not_claim_a_range():
    notes = DOCS.render_notes(
        "v0.17.6", "v0.17.1", "- Change.", "update", ("v0.17.1", "v0.17.5"), "partial"
    )
    assert "Existing v0.17.1, v0.17.5 installations" in notes
    assert "complete release gate has not passed" in notes
    assert "Release gate passed" not in notes


@pytest.mark.parametrize("verification", ["not-run", "preflight", "partial"])
def test_incomplete_checks_never_claim_a_passing_release(verification):
    notes = DOCS.render_notes("v0.18.0", "v0.18.0", "- Change.", "full", (), verification)
    assert "Release gate passed" not in notes
    assert "download/v0.18.0/sleep_scoring_app_v0.18_full.zip" in notes


def test_cli_defaults_to_draft_and_reads_exact_git_candidate(tmp_path):
    candidate(tmp_path, version="v0.18.0", base="v0.18.0")
    for args in (
        ("init",),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Candidate"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    (tmp_path / "CHANGELOG.md").write_text("Uncommitted unrelated edits")
    output = tmp_path / "release_artifacts/notes.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "notes",
            "--repo",
            str(tmp_path),
            "--ref",
            "HEAD",
            "--kind",
            "full",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    notes = output.read_text(encoding="utf-8")
    assert "Current user change" in notes
    assert "Draft: release verification has not run" in notes
    assert "Uncommitted unrelated edits" not in notes
