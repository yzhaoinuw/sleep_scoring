"""Prepare release documentation and render user-facing GitHub release notes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from package_naming import release_line


RELEASES = "https://github.com/yzhaoinuw/sleep_scoring/releases"
METADATA = "packaging/windows/release_metadata.json"
DOWNLOAD_LINK = re.compile(r"\[Download the Windows app\]\(([^)]+)\)")
VERSION = re.compile(r'VERSION\s*=\s*["\'](v\d+\.\d+\.\d+)["\']')


class ReleaseDocsError(ValueError):
    """Documentation is missing or inconsistent with the release candidate."""


def version_key(version: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ReleaseDocsError(f"Unsupported release version: {version!r}")
    return tuple(int(part) for part in version[1:].split("."))


def read_text(repo: Path, path: str, ref: str | None = None) -> str:
    if ref is None:
        return (repo / path).read_text(encoding="utf-8")
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ReleaseDocsError(f"Cannot read {path} from {ref}: {result.stderr.decode().strip()}")
    return result.stdout.decode("utf-8")


def app_version(repo: Path, ref: str | None = None) -> str:
    versions = VERSION.findall(read_text(repo, "app_src/__init__.py", ref))
    if len(versions) != 1:
        raise ReleaseDocsError("Expected one vMAJOR.MINOR.PATCH app version")
    return versions[0]


def download_url(version: str) -> str:
    version_key(version)
    return f"{RELEASES}/download/{version}/sleep_scoring_app_{release_line(version)}_full.zip"


def changelog_entry(changelog: str, version: str) -> str:
    headings = list(re.finditer(r"(?m)^## ([^\r\n]+)\r?$", changelog))
    matches = [index for index, heading in enumerate(headings) if heading[1].strip() == version]
    if len(matches) != 1:
        raise ReleaseDocsError(f"CHANGELOG.md must contain exactly one {version} heading")
    index = matches[0]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
    entry = changelog[headings[index].end() : end].strip()
    if not entry or not re.search(r"(?m)^- \S", entry):
        raise ReleaseDocsError(f"CHANGELOG.md has no user-facing changes for {version}")
    return entry


def validate_docs(
    repo: Path,
    kind: str,
    *,
    ref: str | None = None,
    from_versions: tuple[str, ...] = (),
) -> tuple[str, str, str]:
    version = app_version(repo, ref)
    metadata = json.loads(read_text(repo, METADATA, ref))
    full_version = metadata["full_package_version"]
    full_key = version_key(full_version)
    if kind == "full" and full_version != version:
        raise ReleaseDocsError(
            f"Full-package download targets {full_version}; this full release requires {version}. "
            "Run release_docs.py prepare-full, review, and commit before running the gate."
        )
    if kind == "update":
        if full_key >= version_key(version):
            raise ReleaseDocsError("An update must be newer than its full-package base")
        if full_version not in from_versions:
            raise ReleaseDocsError(
                f"Compatible versions must include full-package base {full_version}"
            )
        if any(not full_key <= version_key(old) < version_key(version) for old in from_versions):
            raise ReleaseDocsError(
                "Compatible versions must lie between the full base and candidate"
            )
    readme = read_text(repo, "README.md", ref)
    expected = download_url(full_version)
    if DOWNLOAD_LINK.findall(readme) != [expected]:
        raise ReleaseDocsError(
            f"README.md must have one 'Download the Windows app' link to {expected}"
        )
    full_links = re.findall(
        r"https://github\.com/yzhaoinuw/sleep_scoring/releases/download/[^\s)]+_full\.zip", readme
    )
    if full_links != [expected]:
        raise ReleaseDocsError("README.md must contain only one full-package download URL")
    entry = changelog_entry(read_text(repo, "CHANGELOG.md", ref), version)
    return version, full_version, entry


def prepare_full(repo: Path) -> None:
    version = app_version(repo)
    changelog_entry(read_text(repo, "CHANGELOG.md"), version)
    readme = read_text(repo, "README.md")
    if len(DOWNLOAD_LINK.findall(readme)) != 1:
        raise ReleaseDocsError("README.md must contain one 'Download the Windows app' link")
    updated = DOWNLOAD_LINK.sub(f"[Download the Windows app]({download_url(version)})", readme)
    metadata = json.loads(read_text(repo, METADATA))
    metadata["full_package_version"] = version
    (repo / "README.md").write_text(updated, encoding="utf-8", newline="\n")
    (repo / METADATA).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def render_notes(
    version: str,
    full_version: str,
    entry: str,
    kind: str,
    from_versions: tuple[str, ...],
    verification: str,
) -> str:
    if kind == "update":
        versions = sorted(set(from_versions), key=version_key)
        keys = [version_key(old) for old in versions]
        contiguous = len(keys) > 1 and all(
            left[:2] == right[:2] and left[2] + 1 == right[2] for left, right in zip(keys, keys[1:])
        )
        supported = f"{versions[0]}–{versions[-1]}" if contiguous else ", ".join(versions)
        installation = (
            f"- Existing {supported} installations update automatically at startup.\n"
            f"- New or older installations: [download the Windows app]({download_url(full_version)}), "
            "extract, and run `unblock_app.cmd`. The app checks for compatible updates when opened."
        )
    else:
        installation = (
            f"[Download the Windows app]({download_url(full_version)}), extract, and run "
            "`unblock_app.cmd`. Existing users also need this full package."
        )
    verification_text = {
        "complete": (
            "Release gate passed, including a fresh packaged-install update and smoke test "
            "with custom settings preserved."
            if kind == "update"
            else "Release gate passed, including packaged-app smoke, updater, and checksum checks."
        ),
        "preflight": "Source preflight passed; packaging and packaged-app checks have not run.",
        "partial": "Development checks only; the complete release gate has not passed.",
        "not-run": "Draft: release verification has not run.",
    }[verification]
    return (
        f"Sleep Scoring App {version}\n\n{entry}\n\n## Installation\n\n{installation}"
        f"\n\n## Verification\n\n{verification_text}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-full", "check", "notes"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--kind", choices=("full", "update"))
    parser.add_argument("--ref", help="Read the exact candidate from Git instead of working files")
    parser.add_argument("--from-ref", action="append", default=[])
    parser.add_argument(
        "--verification", choices=("complete", "preflight", "partial", "not-run"), default="not-run"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        if args.command == "prepare-full":
            if args.ref:
                raise ReleaseDocsError("prepare-full edits working files; omit --ref")
            prepare_full(repo)
            print(
                "Updated README download and full-package metadata. Review and commit before the gate."
            )
            return 0
        if args.kind is None:
            raise ReleaseDocsError("check and notes require --kind full or --kind update")
        from_versions = tuple(app_version(repo, ref) for ref in args.from_ref)
        version, full_version, entry = validate_docs(
            repo, args.kind, ref=args.ref, from_versions=from_versions
        )
        if args.command == "notes":
            output = args.output or repo / "release_artifacts" / f"release_notes_{version}.md"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                render_notes(
                    version, full_version, entry, args.kind, from_versions, args.verification
                ),
                encoding="utf-8",
            )
            print(output)
        else:
            print(f"Release documentation validated: {version} ({args.kind})")
    except (ReleaseDocsError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Release documentation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
