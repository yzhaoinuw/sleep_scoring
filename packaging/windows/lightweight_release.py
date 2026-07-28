"""Validation and fixture helpers for lightweight Windows source releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


APP_NAME = "sleep_scoring"
RUNTIME_PATH = "app_src"
VERSION_FILE = "app_src/__init__.py"
PRESERVED_RUNTIME_PATHS = ("app_src/config.py",)
UPDATE_ZIP_URL_ENV = "SLEEP_SCORING_UPDATE_ZIP_URL"
SKIP_UPDATE_ENV = "SLEEP_SCORING_SKIP_UPDATE"

VERSION_RE = re.compile(r"""VERSION\s*=\s*["']([^"']+)["']""")
SETUP_VERSION_RE = re.compile(
    r"""(?m)^([ \t]*version[ \t]*=[ \t]*)(["'])([^"']+)(\2)([ \t]*,[ \t]*)$"""
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

BLOCKED_PATH_NAMES = frozenset(
    {
        "app.spec",
        "environment.yml",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "run_desktop_app.py",
        "setup.cfg",
        "unblock_app.cmd",
    }
)
BLOCKED_PATH_SUFFIXES = (".lock", ".spec")
BLOCKED_PATH_PREFIXES = (
    ".worktrees/",
    "archive/",
    "build/",
    "cache/",
    "data/",
    "dist/",
    "models/",
    "packaging/windows/release_helpers/",
)


class LightweightReleaseError(ValueError):
    """Raised when a candidate or artifact is not safe for a source release."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def git_result(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )


def git_output(repo: Path, *args: str) -> bytes:
    result = git_result(repo, *args)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise LightweightReleaseError(message or "git command failed")
    return result.stdout


def git_lines(repo: Path, *args: str) -> list[str]:
    return [line for line in git_output(repo, *args).decode(errors="replace").splitlines() if line]


def git_file(repo: Path, ref: str, path: str) -> bytes:
    return git_output(repo, "show", f"{ref}:{path}")


def read_version_bytes(data: bytes, *, source: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LightweightReleaseError(f"{source} is not UTF-8") from exc
    match = VERSION_RE.search(text)
    if match is None:
        raise LightweightReleaseError(f"could not read VERSION from {source}")
    return match.group(1)


def read_version_from_ref(repo: Path, ref: str) -> str:
    return read_version_bytes(git_file(repo, ref, VERSION_FILE), source=f"{ref}:{VERSION_FILE}")


def version_key(version: str) -> tuple[int, ...]:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", version)
    if match is None:
        raise LightweightReleaseError(f"unsupported release version: {version}")
    return tuple(int(part) for part in match.group(1).split("."))


def list_compatible_refs(repo: Path, minimum_version: str, to_ref: str) -> list[str]:
    target_key = version_key(read_version_from_ref(repo, to_ref))
    minimum_key = version_key(minimum_version)
    refs: list[tuple[tuple[int, ...], str]] = []
    for tag in git_lines(repo, "tag", "--merged", to_ref, "--list", "v*"):
        try:
            tag_version = read_version_from_ref(repo, tag)
            tag_key = version_key(tag_version)
        except LightweightReleaseError:
            continue
        if tag != tag_version:
            continue
        if minimum_key <= tag_key < target_key:
            refs.append((tag_key, tag))
    return [tag for _, tag in sorted(refs)]


def mask_setup_version(text: str, *, source: str) -> tuple[str, str]:
    matches = list(SETUP_VERSION_RE.finditer(text))
    if len(matches) != 1:
        raise LightweightReleaseError(
            f"{source} must contain exactly one literal setup() version assignment"
        )
    match = matches[0]
    masked = text[: match.start(3)] + "__VERSION__" + text[match.end(3) :]
    return match.group(3), masked


def validate_setup_version_only(
    previous_setup: bytes,
    current_setup: bytes,
    *,
    previous_app_version: str,
    current_app_version: str,
    previous_source: str = "previous setup.py",
    current_source: str = "current setup.py",
) -> None:
    try:
        previous_text = previous_setup.decode("utf-8")
        current_text = current_setup.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LightweightReleaseError("setup.py must be UTF-8") from exc

    previous_version, previous_masked = mask_setup_version(previous_text, source=previous_source)
    current_version, current_masked = mask_setup_version(current_text, source=current_source)
    if previous_masked != current_masked:
        raise LightweightReleaseError(
            f"{current_source} changed beyond its version assignment; use a full package"
        )
    if previous_version != previous_app_version.removeprefix("v"):
        raise LightweightReleaseError(
            f"{previous_source} version {previous_version!r} does not match "
            f"{previous_app_version!r}"
        )
    if current_version != current_app_version.removeprefix("v"):
        raise LightweightReleaseError(
            f"{current_source} version {current_version!r} does not match "
            f"{current_app_version!r}"
        )


def requires_full_package(path: str) -> bool:
    normalized = normalize_path(path)
    root_name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        normalized in BLOCKED_PATH_NAMES
        or root_name in BLOCKED_PATH_NAMES
        or Path(root_name).suffix.lower() in BLOCKED_PATH_SUFFIXES
        or normalized.startswith(BLOCKED_PATH_PREFIXES)
    )


def changed_paths(repo: Path, from_ref: str, to_ref: str) -> list[tuple[str, list[str]]]:
    changes = []
    for line in git_lines(repo, "diff", "--name-status", from_ref, to_ref):
        fields = line.split("\t")
        changes.append((fields[0], [normalize_path(path) for path in fields[1:]]))
    return changes


def changed_runtime_paths(repo: Path, from_refs: list[str], to_ref: str) -> list[str]:
    paths = set()
    for from_ref in from_refs:
        for status, changed in changed_paths(repo, from_ref, to_ref):
            runtime_changes = [path for path in changed if path.startswith(f"{RUNTIME_PATH}/")]
            if not runtime_changes:
                continue
            if not status.startswith(("A", "M")) or len(changed) != 1:
                raise LightweightReleaseError(
                    "runtime deletions, renames, or complex changes require a full package: "
                    f"{status}\t" + "\t".join(changed)
                )
            paths.update(runtime_changes)
    return sorted(paths)


def validate_candidate(repo: Path, from_refs: list[str], to_ref: str) -> str:
    if not from_refs:
        raise LightweightReleaseError("at least one compatible --from-ref is required")

    target_version = read_version_from_ref(repo, to_ref)
    target_key = version_key(target_version)
    versions_by_ref = {from_ref: read_version_from_ref(repo, from_ref) for from_ref in from_refs}
    all_runtime_paths = set()
    blocked_paths = set()

    for from_ref in from_refs:
        previous_version = versions_by_ref[from_ref]
        if version_key(previous_version) >= target_key:
            raise LightweightReleaseError(
                f"{from_ref} ({previous_version}) is not older than target {target_version}"
            )

        changes = changed_paths(repo, from_ref, to_ref)
        all_runtime_paths.update(
            path for _, paths in changes for path in paths if path.startswith(f"{RUNTIME_PATH}/")
        )
        blocked_paths.update(
            path
            for _, paths in changes
            for path in paths
            if path != "setup.py" and requires_full_package(path)
        )

        validate_setup_version_only(
            git_file(repo, from_ref, "setup.py"),
            git_file(repo, to_ref, "setup.py"),
            previous_app_version=previous_version,
            current_app_version=target_version,
            previous_source=f"{from_ref}:setup.py",
            current_source=f"{to_ref}:setup.py",
        )

    latest_ref = max(from_refs, key=lambda ref: version_key(versions_by_ref[ref]))
    latest_changes = changed_paths(repo, latest_ref, to_ref)
    newly_changed_preserved_paths = {
        path for _, paths in latest_changes for path in paths if path in PRESERVED_RUNTIME_PATHS
    }
    if newly_changed_preserved_paths:
        formatted = ", ".join(sorted(newly_changed_preserved_paths))
        raise LightweightReleaseError(
            "schema-1 releases cannot deliver newly changed user-owned runtime paths "
            f"({formatted}); use a full package or a schema-2-capable distribution"
        )

    changed_runtime_paths(repo, from_refs, to_ref)
    if blocked_paths:
        formatted = "\n  ".join(sorted(blocked_paths))
        raise LightweightReleaseError(
            "these changes cross the frozen-package boundary:\n  " + formatted
        )

    deliverable_paths = all_runtime_paths - set(PRESERVED_RUNTIME_PATHS)
    if not deliverable_paths:
        raise LightweightReleaseError(
            "no deliverable app_src files changed after preserving user-owned runtime paths"
        )
    return target_version


def _find_runtime_member_path(name: str, runtime_path: str) -> str | None:
    normalized = normalize_path(name).rstrip("/")
    if not normalized:
        return None
    parts = PurePosixPath(normalized).parts
    runtime_parts = PurePosixPath(runtime_path).parts
    matches = [
        index
        for index in range(len(parts) - len(runtime_parts) + 1)
        if parts[index : index + len(runtime_parts)] == runtime_parts
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise LightweightReleaseError(f"ambiguous runtime path in ZIP member {name!r}")
    return str(PurePosixPath(*parts[matches[0] :]))


def export_installed_baseline(
    package_zip: Path,
    version: str,
    output: Path,
    *,
    runtime_path: str = RUNTIME_PATH,
) -> dict[str, object]:
    files: dict[str, str] = {}
    version_bytes = None
    with zipfile.ZipFile(package_zip) as package:
        for entry in package.infolist():
            if entry.is_dir() or normalize_path(entry.filename).endswith("/"):
                continue
            path = _find_runtime_member_path(entry.filename, runtime_path)
            if path is None:
                continue
            if path in files:
                raise LightweightReleaseError(
                    f"{package_zip} contains more than one installed file for {path}"
                )
            data = package.read(entry)
            files[path] = sha256_bytes(data)
            if path == VERSION_FILE:
                version_bytes = data

    if not files:
        raise LightweightReleaseError(f"{package_zip} contains no files under {runtime_path}")
    if version_bytes is None:
        raise LightweightReleaseError(f"{package_zip} does not contain {VERSION_FILE}")
    installed_version = read_version_bytes(version_bytes, source=f"{package_zip}:{VERSION_FILE}")
    if installed_version != version:
        raise LightweightReleaseError(
            f"{package_zip} reports {installed_version}, expected {version}"
        )

    manifest: dict[str, object] = {
        "version": version,
        "runtime_paths": [runtime_path],
        "files": dict(sorted(files.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_baseline_manifest(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LightweightReleaseError(f"could not read baseline manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LightweightReleaseError(f"baseline manifest {path} must be a JSON object")
    if not isinstance(data.get("version"), str):
        raise LightweightReleaseError(f"baseline manifest {path} has no string version")
    if not isinstance(data.get("files"), dict):
        raise LightweightReleaseError(f"baseline manifest {path} has no files object")
    runtime_paths = data.get("runtime_paths")
    if not isinstance(runtime_paths, list) or not all(
        isinstance(item, str) and item for item in runtime_paths
    ):
        raise LightweightReleaseError(
            f"baseline manifest {path} must declare complete runtime_paths"
        )
    return data


def materialize_baseline(
    source: Path,
    output: Path,
    repo: Path,
    from_refs: list[str],
    to_ref: str,
) -> dict[str, object]:
    data = load_baseline_manifest(source)
    runtime_paths = tuple(normalize_path(path).rstrip("/") for path in data["runtime_paths"])
    files = {normalize_path(path): digest for path, digest in dict(data["files"]).items()}
    for path in changed_runtime_paths(repo, from_refs, to_ref):
        if path in files:
            continue
        if any(path == runtime or path.startswith(f"{runtime}/") for runtime in runtime_paths):
            files[path] = None
        else:
            raise LightweightReleaseError(
                f"baseline manifest {source} does not cover changed runtime path {path}"
            )

    materialized = {
        "version": data["version"],
        "files": dict(sorted(files.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(materialized, indent=2) + "\n", encoding="utf-8")
    return materialized


def _validate_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value.lower()) is None:
        raise LightweightReleaseError(f"{field} must be a SHA-256 string")
    return value.lower()


def validate_update_asset(
    update_zip: Path,
    checksum_file: Path,
    repo: Path,
    from_refs: list[str],
    to_ref: str,
) -> dict[str, object]:
    expected_version = read_version_from_ref(repo, to_ref)
    expected_from_versions = list(
        dict.fromkeys(read_version_from_ref(repo, ref) for ref in from_refs)
    )
    with zipfile.ZipFile(update_zip) as update:
        names = [
            normalize_path(entry.filename) for entry in update.infolist() if not entry.is_dir()
        ]
        if names.count("manifest.json") != 1:
            raise LightweightReleaseError("update ZIP must contain exactly one manifest.json")
        if len(names) != len(set(names)):
            raise LightweightReleaseError("update ZIP contains duplicate file entries")
        try:
            manifest = json.loads(update.read("manifest.json"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LightweightReleaseError(f"invalid update manifest: {exc}") from exc

        if manifest.get("schema_version") != 1:
            raise LightweightReleaseError(
                "lightweight releases must remain schema 1 until the next full redistribution"
            )
        if manifest.get("app") != APP_NAME:
            raise LightweightReleaseError("update manifest is for a different app")
        if manifest.get("version") != expected_version:
            raise LightweightReleaseError(
                f"update manifest version does not match {to_ref}:{VERSION_FILE}"
            )
        if manifest.get("from_versions") != expected_from_versions:
            raise LightweightReleaseError(
                "update manifest from_versions do not match the compatible refs"
            )

        file_entries = manifest.get("files")
        changed_files = manifest.get("changed_files")
        if not isinstance(file_entries, list) or not file_entries:
            raise LightweightReleaseError("update manifest contains no files")
        if not isinstance(changed_files, list):
            raise LightweightReleaseError("update manifest has no changed_files list")
        paths = [entry.get("path") for entry in file_entries if isinstance(entry, dict)]
        if len(paths) != len(file_entries) or len(set(paths)) != len(paths):
            raise LightweightReleaseError("update manifest contains invalid or duplicate paths")
        if changed_files != paths:
            raise LightweightReleaseError(
                "changed_files must exactly match the ordered manifest file paths"
            )
        if set(names) != {"manifest.json", *paths}:
            raise LightweightReleaseError(
                "update ZIP payload does not exactly match the manifest file list"
            )

        for entry in file_entries:
            path = entry["path"]
            if not isinstance(path, str) or not path.startswith(f"{RUNTIME_PATH}/"):
                raise LightweightReleaseError(f"unsafe update payload path: {path!r}")
            if path in PRESERVED_RUNTIME_PATHS:
                raise LightweightReleaseError(
                    f"user-owned runtime path must not be replaced: {path}"
                )
            expected_hash = _validate_digest(entry.get("sha256"), field=f"{path}.sha256")
            if sha256_bytes(update.read(path)) != expected_hash:
                raise LightweightReleaseError(f"payload hash mismatch for {path}")

            previous = entry.get("previous_sha256")
            previous_by_version = entry.get("previous_sha256_by_version")
            if (previous is None) == (previous_by_version is None):
                raise LightweightReleaseError(
                    f"{path} must declare exactly one previous-hash representation"
                )
            if previous is not None:
                if not isinstance(previous, list) or not previous:
                    raise LightweightReleaseError(
                        f"{path}.previous_sha256 must be a non-empty list"
                    )
                for index, digest in enumerate(previous):
                    _validate_digest(digest, field=f"{path}.previous_sha256[{index}]")
            else:
                if (
                    not isinstance(previous_by_version, dict)
                    or list(previous_by_version) != expected_from_versions
                ):
                    raise LightweightReleaseError(
                        f"{path}.previous_sha256_by_version does not match from_versions"
                    )
                for version, digest in previous_by_version.items():
                    if digest is not None:
                        _validate_digest(
                            digest, field=f"{path}.previous_sha256_by_version[{version}]"
                        )

    expected_checksum = sha256_file(update_zip)
    try:
        checksum_text = checksum_file.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise LightweightReleaseError(
            f"could not read checksum file {checksum_file}: {exc}"
        ) from exc
    declared_checksum = checksum_text.split(maxsplit=1)[0].lower()
    if declared_checksum != expected_checksum:
        raise LightweightReleaseError("update ZIP checksum sidecar does not match")
    return manifest


def _safe_extract_zip(source_zip: Path, destination: Path) -> None:
    with zipfile.ZipFile(source_zip) as source:
        for entry in source.infolist():
            normalized = normalize_path(entry.filename)
            pure_path = PurePosixPath(normalized)
            if (
                pure_path.is_absolute()
                or not pure_path.parts
                or any(part in {"", ".", ".."} for part in pure_path.parts)
            ):
                raise LightweightReleaseError(
                    f"unsafe ZIP member in {source_zip}: {entry.filename!r}"
                )
            output = destination.joinpath(*pure_path.parts)
            if entry.is_dir() or normalized.endswith("/"):
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as input_file, output.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


def _run_frozen_app(app_root: Path, args: list[str], env: dict[str, str]) -> None:
    executable = app_root / "run_desktop_app.exe"
    result = subprocess.run(
        [str(executable), *args],
        cwd=app_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part)
        raise LightweightReleaseError(
            f"{executable.name} {' '.join(args)} failed with exit code "
            f"{result.returncode}\n{output}"
        )


def _asset_version(update_zip: Path) -> str:
    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise LightweightReleaseError(f"{update_zip} has no manifest version")
    return version


def test_installed_update(
    package_zip: Path,
    update_zip: Path,
    prerequisite_updates: list[Path],
    *,
    expected_initial_version: str | None = None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="sleep-scoring-update-fixture-") as temp_dir:
        fixture_root = Path(temp_dir)
        _safe_extract_zip(package_zip, fixture_root)
        executables = list(fixture_root.rglob("run_desktop_app.exe"))
        if len(executables) != 1:
            raise LightweightReleaseError(
                f"expected one run_desktop_app.exe in {package_zip}, found {len(executables)}"
            )
        app_root = executables[0].parent
        version_file = app_root / VERSION_FILE
        initial_version = read_version_bytes(version_file.read_bytes(), source=str(version_file))
        if expected_initial_version and initial_version != expected_initial_version:
            raise LightweightReleaseError(
                f"{package_zip} reports {initial_version}, expected {expected_initial_version}"
            )
        config_path = app_root / PRESERVED_RUNTIME_PATHS[0]
        config_hash = sha256_file(config_path)

        for asset in [*prerequisite_updates, update_zip]:
            env = os.environ.copy()
            env.pop(SKIP_UPDATE_ENV, None)
            env[UPDATE_ZIP_URL_ENV] = str(asset.resolve())
            _run_frozen_app(app_root, ["--check-update"], env)
            installed_version = read_version_bytes(
                version_file.read_bytes(), source=str(version_file)
            )
            expected_version = _asset_version(asset)
            if installed_version != expected_version:
                raise LightweightReleaseError(
                    f"{asset} reported success but installed version is "
                    f"{installed_version}, expected {expected_version}"
                )
            if sha256_file(config_path) != config_hash:
                raise LightweightReleaseError(
                    f"{asset} changed preserved {PRESERVED_RUNTIME_PATHS[0]}"
                )

        smoke_env = os.environ.copy()
        smoke_env.pop(UPDATE_ZIP_URL_ENV, None)
        smoke_env[SKIP_UPDATE_ENV] = "1"
        _run_frozen_app(app_root, ["--smoke"], smoke_env)
        return read_version_bytes(version_file.read_bytes(), source=str(version_file))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    refs = subparsers.add_parser("compatible-refs")
    refs.add_argument("--repo", type=Path, required=True)
    refs.add_argument("--minimum-version", required=True)
    refs.add_argument("--to-ref", default="HEAD")

    version = subparsers.add_parser("version")
    version.add_argument("--repo", type=Path, required=True)
    version.add_argument("--ref", default="HEAD")

    candidate = subparsers.add_parser("validate-candidate")
    candidate.add_argument("--repo", type=Path, required=True)
    candidate.add_argument("--from-ref", action="append", required=True)
    candidate.add_argument("--to-ref", default="HEAD")

    export = subparsers.add_parser("export-baseline")
    export.add_argument("--package-zip", type=Path, required=True)
    export.add_argument("--version", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--runtime-path", default=RUNTIME_PATH)

    materialize = subparsers.add_parser("materialize-baseline")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--repo", type=Path, required=True)
    materialize.add_argument("--from-ref", action="append", required=True)
    materialize.add_argument("--to-ref", default="HEAD")

    asset = subparsers.add_parser("validate-asset")
    asset.add_argument("--update-zip", type=Path, required=True)
    asset.add_argument("--checksum-file", type=Path, required=True)
    asset.add_argument("--repo", type=Path, required=True)
    asset.add_argument("--from-ref", action="append", required=True)
    asset.add_argument("--to-ref", default="HEAD")

    fixture = subparsers.add_parser("test-installed-update")
    fixture.add_argument("--package-zip", type=Path, required=True)
    fixture.add_argument("--update-zip", type=Path, required=True)
    fixture.add_argument("--prerequisite-update", type=Path, action="append", default=[])
    fixture.add_argument("--expected-initial-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compatible-refs":
            for ref in list_compatible_refs(args.repo, args.minimum_version, args.to_ref):
                print(ref)
        elif args.command == "version":
            print(read_version_from_ref(args.repo, args.ref))
        elif args.command == "validate-candidate":
            version = validate_candidate(args.repo, args.from_ref, args.to_ref)
            print(f"Lightweight release candidate is valid: {version}")
        elif args.command == "export-baseline":
            manifest = export_installed_baseline(
                args.package_zip,
                args.version,
                args.output,
                runtime_path=args.runtime_path,
            )
            print(f"Wrote {len(manifest['files'])} installed hashes to {args.output}")
        elif args.command == "materialize-baseline":
            materialize_baseline(args.source, args.output, args.repo, args.from_ref, args.to_ref)
            print(args.output)
        elif args.command == "validate-asset":
            manifest = validate_update_asset(
                args.update_zip,
                args.checksum_file,
                args.repo,
                args.from_ref,
                args.to_ref,
            )
            print(
                f"Validated schema-1 update {manifest['version']} with "
                f"{len(manifest['files'])} payload files"
            )
        elif args.command == "test-installed-update":
            version = test_installed_update(
                args.package_zip,
                args.update_zip,
                args.prerequisite_update,
                expected_initial_version=args.expected_initial_version,
            )
            print(f"Frozen installed-app update and smoke test passed: {version}")
    except LightweightReleaseError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
