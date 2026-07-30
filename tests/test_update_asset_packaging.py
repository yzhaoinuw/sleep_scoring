import hashlib
import importlib.util
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "align_update_asset_with_package.py"
)
SPEC = importlib.util.spec_from_file_location("align_update_asset_with_package", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXPORT_SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "export_runtime_from_git.py"
)
EXPORT_SPEC = importlib.util.spec_from_file_location("export_runtime_from_git", EXPORT_SCRIPT_PATH)
EXPORT_MODULE = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORT_MODULE)

LIGHTWEIGHT_SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "lightweight_release.py"
)
LIGHTWEIGHT_SPEC = importlib.util.spec_from_file_location(
    "lightweight_release", LIGHTWEIGHT_SCRIPT_PATH
)
LIGHTWEIGHT_MODULE = importlib.util.module_from_spec(LIGHTWEIGHT_SPEC)
LIGHTWEIGHT_SPEC.loader.exec_module(LIGHTWEIGHT_MODULE)


def _write_update_zip(path):
    manifest = {
        "from_versions": ["v0.16.5"],
        "files": [
            {
                "path": "app_src/session.py",
                "sha256": hashlib.sha256(b"new\n").hexdigest(),
                "previous_sha256_by_version": {"v0.16.5": "git-blob-hash"},
            }
        ],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        update.writestr("app_src/session.py", b"new\n")


def test_align_update_asset_uses_exact_packaged_baseline_bytes(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("sleep_scoring_app_v0.16.5/app_src/session.py", b"previous\r\n")

    MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])

    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
        assert update.read("app_src/session.py") == b"new\n"
    assert set(manifest["files"][0]["previous_sha256"]) == {
        "git-blob-hash",
        hashlib.sha256(b"previous\r\n").hexdigest(),
    }
    assert "previous_sha256_by_version" not in manifest["files"][0]


def test_align_update_asset_preserves_multiple_package_lineages(tmp_path):
    update_zip = tmp_path / "update.zip"
    full_package_zip = tmp_path / "full.zip"
    patched_package_zip = tmp_path / "patched.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(full_package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("full/app_src/session.py", b"full-package\r\n")
    with zipfile.ZipFile(patched_package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("patched/app_src/session.py", b"previous-source-update\n")

    MODULE.align_update_asset(
        update_zip,
        [
            ("v0.16.5", full_package_zip),
            ("v0.16.5", patched_package_zip),
        ],
    )

    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
    assert set(manifest["files"][0]["previous_sha256"]) == {
        "git-blob-hash",
        hashlib.sha256(b"full-package\r\n").hexdigest(),
        hashlib.sha256(b"previous-source-update\n").hexdigest(),
    }


def test_align_update_asset_rejects_missing_packaged_runtime_file(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w"):
        pass

    with pytest.raises(ValueError, match="expected exactly one"):
        MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])


def test_align_update_asset_reads_windows_package_member_separators(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            r"sleep_scoring_app_v0.16.5\app_src\session.py",
            b"packaged\r\n",
        )

    MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])

    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
    assert hashlib.sha256(b"packaged\r\n").hexdigest() in set(
        manifest["files"][0]["previous_sha256"]
    )


def test_align_update_asset_preserves_builder_multi_hash_representation(tmp_path):
    update_zip = tmp_path / "update.zip"
    manifest = {
        "from_versions": ["v0.16.5"],
        "changed_files": ["app_src/config.py", "app_src/session.py"],
        "files": [
            {
                "path": "app_src/config.py",
                "sha256": "new-config",
                "previous_sha256": ["old-config-a", "old-config-b"],
            },
            {
                "path": "app_src/session.py",
                "sha256": hashlib.sha256(b"new\n").hexdigest(),
                "previous_sha256": ["old-session-a", "old-session-b"],
            },
        ],
    }
    with zipfile.ZipFile(update_zip, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        update.writestr("app_src/config.py", b"new config\n")
        update.writestr("app_src/session.py", b"new\n")

    MODULE.align_update_asset(update_zip, [], ("app_src/config.py",))

    with zipfile.ZipFile(update_zip) as update:
        updated_manifest = json.loads(update.read("manifest.json"))
        assert "app_src/config.py" not in update.namelist()
    assert updated_manifest["changed_files"] == ["app_src/session.py"]
    assert updated_manifest["files"][0]["previous_sha256"] == [
        "old-session-a",
        "old-session-b",
    ]


def test_align_update_asset_omits_preserved_user_config(tmp_path):
    update_zip = tmp_path / "update.zip"
    manifest = {
        "from_versions": ["v0.16.6"],
        "changed_files": ["app_src/config.py", "app_src/make_figure.py"],
        "files": [
            {
                "path": "app_src/config.py",
                "sha256": "new-config",
                "previous_sha256_by_version": {"v0.16.6": "old-config"},
            },
            {
                "path": "app_src/make_figure.py",
                "sha256": "new-figure",
                "previous_sha256_by_version": {"v0.16.6": "old-figure"},
            },
        ],
    }
    with zipfile.ZipFile(update_zip, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        update.writestr("app_src/config.py", b"new config\n")
        update.writestr("app_src/make_figure.py", b"new figure\n")

    MODULE.align_update_asset(update_zip, [], ("app_src/config.py",))

    with zipfile.ZipFile(update_zip) as update:
        updated_manifest = json.loads(update.read("manifest.json"))
        assert "app_src/config.py" not in update.namelist()
        assert update.read("app_src/make_figure.py") == b"new figure\n"
    assert updated_manifest["changed_files"] == ["app_src/make_figure.py"]
    assert [entry["path"] for entry in updated_manifest["files"]] == ["app_src/make_figure.py"]


def test_export_runtime_writes_exact_git_blob_bytes(tmp_path):
    repo = Path(__file__).parents[1]

    exported = EXPORT_MODULE.export_runtime(repo, "HEAD", "app_src", tmp_path)

    assert "app_src/__init__.py" in exported
    expected = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:app_src/__init__.py"])
    assert (tmp_path / "app_src" / "__init__.py").read_bytes() == expected


def test_export_installed_baseline_hashes_exact_windows_zip_bytes(tmp_path):
    package_zip = tmp_path / "full.zip"
    output = tmp_path / "baseline.json"
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            r"sleep_scoring_app_v1.2.3\app_src\__init__.py",
            b'VERSION = "v1.2.3"\r\n',
        )
        package.writestr(
            r"sleep_scoring_app_v1.2.3\app_src\session.py",
            b"packaged\r\n",
        )

    manifest = LIGHTWEIGHT_MODULE.export_installed_baseline(package_zip, "v1.2.3", output)

    assert manifest["runtime_paths"] == ["app_src"]
    assert manifest["files"]["app_src/session.py"] == hashlib.sha256(b"packaged\r\n").hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_export_installed_baseline_rejects_wrong_embedded_version(tmp_path):
    package_zip = tmp_path / "full.zip"
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            r"sleep_scoring_app_v1.2.3\app_src\__init__.py",
            b'VERSION = "v1.2.2"\n',
        )

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="reports v1.2.2, expected v1.2.3",
    ):
        LIGHTWEIGHT_MODULE.export_installed_baseline(
            package_zip, "v1.2.3", tmp_path / "baseline.json"
        )


def test_validate_setup_version_only_accepts_matching_assignment():
    previous = b'setup(\n    name="sleep_scoring",\n    version="1.2.2",\n)\n'
    current = b'setup(\n    name="sleep_scoring",\n    version="1.2.3",\n)\n'

    LIGHTWEIGHT_MODULE.validate_setup_version_only(
        previous,
        current,
        previous_app_version="v1.2.2",
        current_app_version="v1.2.3",
    )


def test_validate_setup_version_only_rejects_other_setup_changes():
    previous = b'setup(\n    name="sleep_scoring",\n    version="1.2.2",\n)\n'
    current = b'setup(\n    name="renamed",\n    version="1.2.3",\n)\n'

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="changed beyond its version assignment",
    ):
        LIGHTWEIGHT_MODULE.validate_setup_version_only(
            previous,
            current,
            previous_app_version="v1.2.2",
            current_app_version="v1.2.3",
        )


def test_materialize_baseline_marks_new_runtime_files_absent(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    output = tmp_path / "materialized.json"
    source.write_text(
        json.dumps(
            {
                "version": "v1.2.2",
                "runtime_paths": ["app_src"],
                "files": {"app_src/existing.py": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        LIGHTWEIGHT_MODULE,
        "changed_runtime_paths",
        lambda repo, from_refs, to_ref: [
            "app_src/existing.py",
            "app_src/new_module.py",
        ],
    )

    materialized = LIGHTWEIGHT_MODULE.materialize_baseline(
        source, output, tmp_path, ["v1.2.2"], "HEAD"
    )

    assert materialized["files"] == {
        "app_src/existing.py": "a" * 64,
        "app_src/new_module.py": None,
    }


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
    )


def _write_citation(repo, version):
    # Today's date keeps the fixture valid whatever the workstation clock says,
    # since the validator rejects a future date-released.
    (repo / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        'title: "sleep_scoring"\n'
        f"version: {version.removeprefix('v')}\n"
        f'date-released: "{date.today().isoformat()}"\n',
        encoding="utf-8",
    )


def _write_candidate_repo(repo, version="v0.17.0"):
    (repo / "app_src").mkdir()
    (repo / "app_src" / "__init__.py").write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    (repo / "app_src" / "config.py").write_text("SETTING = 1\n", encoding="utf-8")
    (repo / "app_src" / "session.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "setup.py").write_text(
        f'setup(\n    name="sleep_scoring",\n    version="{version.removeprefix("v")}",\n)\n',
        encoding="utf-8",
    )
    _write_citation(repo, version)
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Tests")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    _git(repo, "tag", version)


def _commit_config_candidate(repo, version):
    (repo / "app_src" / "__init__.py").write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    (repo / "app_src" / "config.py").write_text("SETTING = 2\n", encoding="utf-8")
    (repo / "setup.py").write_text(
        f'setup(\n    name="sleep_scoring",\n    version="{version.removeprefix("v")}",\n)\n',
        encoding="utf-8",
    )
    _write_citation(repo, version)
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "candidate")


def test_validate_candidate_allows_schema2_config_change_from_v017(tmp_path):
    _write_candidate_repo(tmp_path)
    _commit_config_candidate(tmp_path, "v0.17.1")

    assert LIGHTWEIGHT_MODULE.validate_candidate(tmp_path, ["v0.17.0"], "HEAD") == "v0.17.1"


def test_validate_candidate_rejects_schema2_config_change_from_pre_v017(tmp_path):
    _write_candidate_repo(tmp_path, "v0.16.6")
    _commit_config_candidate(tmp_path, "v0.17.0")

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="requires schema-2-capable installed versions v0.17.0 or newer",
    ):
        LIGHTWEIGHT_MODULE.validate_candidate(tmp_path, ["v0.16.6"], "HEAD")


def _write_schema2_asset(repo, update_zip, *, editable_assignments=None, schema_version=2):
    paths = ["app_src/__init__.py", "app_src/config.py"]
    files = []
    for path in paths:
        data = (repo / path).read_bytes()
        entry = {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "previous_sha256_by_version": {"v0.17.0": "a" * 64},
        }
        if path == "app_src/config.py" and schema_version == 2:
            entry["update_strategy"] = "python-config-merge"
            entry["editable_assignments"] = (
                list(LIGHTWEIGHT_MODULE.EDITABLE_CONFIG_ASSIGNMENTS)
                if editable_assignments is None
                else editable_assignments
            )
        files.append(entry)
    manifest = {
        "schema_version": schema_version,
        "app": "sleep_scoring",
        "version": "v0.17.1",
        "from_versions": ["v0.17.0"],
        "changed_files": paths,
        "files": files,
    }
    with zipfile.ZipFile(update_zip, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        for path in paths:
            update.write(repo / path, path)
    checksum = update_zip.with_suffix(update_zip.suffix + ".sha256.txt")
    checksum.write_text(
        f"{hashlib.sha256(update_zip.read_bytes()).hexdigest()}  {update_zip.name}\n",
        encoding="utf-8",
    )
    return checksum


def test_validate_update_asset_accepts_approved_schema2_config_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_candidate_repo(repo)
    _commit_config_candidate(repo, "v0.17.1")
    update_zip = tmp_path / "update.zip"
    checksum = _write_schema2_asset(repo, update_zip)

    manifest = LIGHTWEIGHT_MODULE.validate_update_asset(
        update_zip, checksum, repo, ["v0.17.0"], "HEAD"
    )

    assert manifest["schema_version"] == 2


def test_validate_update_asset_rejects_unapproved_config_allowlist(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_candidate_repo(repo)
    _commit_config_candidate(repo, "v0.17.1")
    update_zip = tmp_path / "update.zip"
    checksum = _write_schema2_asset(repo, update_zip, editable_assignments=["SETTING"])

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="does not match the approved user-facing configuration allowlist",
    ):
        LIGHTWEIGHT_MODULE.validate_update_asset(update_zip, checksum, repo, ["v0.17.0"], "HEAD")


def test_validate_update_asset_rejects_schema1_config_replacement(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_candidate_repo(repo)
    _commit_config_candidate(repo, "v0.17.1")
    update_zip = tmp_path / "update.zip"
    checksum = _write_schema2_asset(repo, update_zip, schema_version=1)

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="must use schema 2",
    ):
        LIGHTWEIGHT_MODULE.validate_update_asset(update_zip, checksum, repo, ["v0.17.0"], "HEAD")


def test_validate_candidate_rejects_forgotten_setup_version_bump(tmp_path):
    _write_candidate_repo(tmp_path)
    (tmp_path / "app_src" / "__init__.py").write_text('VERSION = "v0.17.1"\n', encoding="utf-8")
    (tmp_path / "app_src" / "session.py").write_text("VALUE = 2\n", encoding="utf-8")
    # Bump the citation too, so setup.py is the only stale file and this test
    # still fails for the reason it names. Leaving it stale would trip the
    # citation check first, whose message also mentions v0.17.1.
    _write_citation(tmp_path, "v0.17.1")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "candidate")

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match=r"setup\.py version '0\.17\.0' does not match 'v0\.17\.1'",
    ):
        LIGHTWEIGHT_MODULE.validate_candidate(tmp_path, ["v0.17.0"], "HEAD")


def test_validate_candidate_rejects_forgotten_citation_version_bump(tmp_path):
    # Lightweight releases are still tags, and Zenodo archives every published
    # release, so a stale citation matters here as much as in a full release.
    _write_candidate_repo(tmp_path)
    (tmp_path / "app_src" / "__init__.py").write_text('VERSION = "v0.17.1"\n', encoding="utf-8")
    (tmp_path / "app_src" / "session.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text(
        'setup(\n    name="sleep_scoring",\n    version="0.17.1",\n)\n',
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "candidate")

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match=r"CITATION\.cff version '0\.17\.0' does not match 'v0\.17\.1'",
    ):
        LIGHTWEIGHT_MODULE.validate_candidate(tmp_path, ["v0.17.0"], "HEAD")


def test_validate_candidate_rejects_unparseable_citation(tmp_path):
    # Mismatched quotes make this an unterminated YAML scalar. The version and
    # date lines look right to a pattern match, but Zenodo could not read it.
    _write_candidate_repo(tmp_path)
    (tmp_path / "app_src" / "__init__.py").write_text('VERSION = "v0.17.1"\n', encoding="utf-8")
    (tmp_path / "app_src" / "session.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text(
        'setup(\n    name="sleep_scoring",\n    version="0.17.1",\n)\n',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "0.17.1\'\ndate-released: "2026-07-29"\n',
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "candidate")

    with pytest.raises(
        LIGHTWEIGHT_MODULE.LightweightReleaseError,
        match="not valid YAML",
    ):
        LIGHTWEIGHT_MODULE.validate_candidate(tmp_path, ["v0.17.0"], "HEAD")


def test_lightweight_release_gate_does_not_repeat_source_tests_in_builder():
    script = LIGHTWEIGHT_SCRIPT_PATH.with_name("release_lightweight.ps1").read_text(
        encoding="utf-8"
    )

    assert script.count('"pytest",') == 1
    assert "SkipTests = $true" in script
    assert "AllowDirty = $true" in script
    assert '"lightweight_release_" + [guid]::NewGuid()' in script
    assert '$MinimumCompatibleVersion = "v0.17.0"' in script
    assert "sleep_scoring_app_v0.17_full.zip" in script
    assert "v0.16.6-windows.json" not in script


def test_source_asset_builder_wires_the_approved_schema2_config_contract():
    script = LIGHTWEIGHT_SCRIPT_PATH.with_name("make_source_update_asset.ps1").read_text(
        encoding="utf-8"
    )

    assert '"--python-config-merge", $PythonConfigMergePath' in script
    assert '"--editable-assignment", $Assignment' in script
    assert "$PythonConfigChanged" in script
    assert "$PreserveRuntimePath" not in script
    for assignment in LIGHTWEIGHT_MODULE.EDITABLE_CONFIG_ASSIGNMENTS:
        assert f'"{assignment}"' in script
