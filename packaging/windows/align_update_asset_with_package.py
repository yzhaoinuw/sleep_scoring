"""Align source-update baseline hashes with bytes from released full packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_package_spec(value: str) -> tuple[str, Path]:
    version, separator, package_path = value.partition("=")
    if not separator or not version or not package_path:
        raise argparse.ArgumentTypeError(
            "package baseline must use VERSION=PATH, for example "
            "v0.16.5=release_artifacts/app.zip"
        )
    return version, Path(package_path)


def read_packaged_file(package_zip: Path, runtime_path: str) -> bytes:
    wanted_parts = PurePosixPath(runtime_path).parts
    with zipfile.ZipFile(package_zip) as package:
        matches = [
            name
            for name in package.namelist()
            if PurePosixPath(name).parts[-len(wanted_parts) :] == wanted_parts
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {runtime_path!r} in {package_zip}, " f"found {len(matches)}"
            )
        return package.read(matches[0])


def align_update_asset(
    update_zip: Path,
    package_specs: list[tuple[str, Path]],
    preserved_paths: tuple[str, ...] = (),
) -> None:
    preserved_paths = {str(PurePosixPath(path)) for path in preserved_paths}
    with zipfile.ZipFile(update_zip) as source:
        manifest = json.loads(source.read("manifest.json"))
        entries = [(entry, source.read(entry.filename)) for entry in source.infolist()]

    from_versions = set(manifest.get("from_versions", []))
    provided_versions = {version for version, _ in package_specs}
    unknown_versions = provided_versions - from_versions
    if unknown_versions:
        raise ValueError(
            "package baseline version is not declared by the update asset: "
            + ", ".join(sorted(unknown_versions))
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("update manifest has no files list")
    files = [entry for entry in files if entry["path"] not in preserved_paths]
    manifest["files"] = files
    manifest["changed_files"] = [
        path for path in manifest.get("changed_files", []) if path not in preserved_paths
    ]

    package_hashes_by_path: dict[str, dict[str, set[str]]] = {}
    for version, package_zip in package_specs:
        for file_entry in files:
            runtime_path = file_entry["path"]
            previous_hashes = file_entry["previous_sha256_by_version"]
            if version not in previous_hashes:
                raise ValueError(f"{runtime_path!r} has no baseline entry for {version!r}")
            package_hashes_by_path.setdefault(runtime_path, {}).setdefault(version, set()).add(
                sha256(read_packaged_file(package_zip, runtime_path))
            )

    for file_entry in files:
        runtime_path = file_entry["path"]
        versioned_hashes = {
            version: ({digest} if digest is not None else {None})
            for version, digest in file_entry["previous_sha256_by_version"].items()
        }
        for version, hashes in package_hashes_by_path.get(runtime_path, {}).items():
            versioned_hashes[version].update(hashes)

        if any(len(hashes) > 1 for hashes in versioned_hashes.values()):
            if any(None in hashes for hashes in versioned_hashes.values()):
                raise ValueError(
                    f"cannot combine multiple byte lineages with a missing-file baseline for "
                    f"{runtime_path!r}"
                )
            file_entry["previous_sha256"] = sorted(
                {digest for hashes in versioned_hashes.values() for digest in hashes}
            )
            file_entry.pop("previous_sha256_by_version")
        else:
            file_entry["previous_sha256_by_version"] = {
                version: next(iter(hashes)) for version, hashes in versioned_hashes.items()
            }

    manifest_bytes = json.dumps(manifest, indent=2).encode()
    update_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=update_zip.parent, suffix=".zip", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with zipfile.ZipFile(temp_path, "w") as destination:
            for entry, data in entries:
                if entry.filename == "manifest.json":
                    data = manifest_bytes
                elif str(PurePosixPath(entry.filename)) in preserved_paths:
                    continue
                destination.writestr(entry, data)
        os.replace(temp_path, update_zip)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace source-update baseline hashes with the exact bytes shipped "
            "in released full-package ZIPs."
        )
    )
    parser.add_argument("--update-zip", type=Path, required=True)
    parser.add_argument(
        "--from-package-zip",
        action="append",
        type=parse_package_spec,
        default=[],
        help="Released baseline in VERSION=PATH form. Repeat as needed.",
    )
    parser.add_argument(
        "--preserve-path",
        action="append",
        default=[],
        help="User-owned runtime path to omit from the automatic update.",
    )
    args = parser.parse_args(argv)
    align_update_asset(args.update_zip, args.from_package_zip, tuple(args.preserve_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
