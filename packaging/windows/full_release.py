"""Preflight validation for a full Windows Sleep Scoring App release."""

from __future__ import annotations

import argparse
import ast
import re
from datetime import date
from pathlib import Path


APP_VERSION_RE = re.compile(r"""VERSION\s*=\s*["']([^"']+)["']""")
SETUP_VERSION_RE = re.compile(r"""(?m)^[ \t]*version[ \t]*=[ \t]*["']([^"']+)["'][ \t]*,[ \t]*$""")
# Top-level CFF keys only. Nested keys under `references:` are indented, so
# anchoring to the start of the line keeps this from matching a cited work.
CITATION_VERSION_RE = re.compile(r"""(?m)^version:[ \t]*["']?([^"'\s]+)["']?[ \t]*$""")
CITATION_DATE_RE = re.compile(r"""(?m)^date-released:[ \t]*["']?(\d{4}-\d{2}-\d{2})["']?[ \t]*$""")
PINNED_UPDATER_RE = re.compile(
    r"(?m)^desktop-app-source-updater\s+@\s+"
    r"git\+https://github\.com/yzhaoinuw/desktop_app_source_updater\.git@"
    r"([0-9a-f]{40})$"
)


class FullReleaseError(ValueError):
    """Raised when the checkout is not ready for a full-package build."""


def _single_match(pattern: re.Pattern[str], text: str, source: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise FullReleaseError(f"{source} must contain exactly one matching value")
    return matches[0]


def _has_assignment(source: str, assignment: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FullReleaseError(f"app_src/config.py is not valid Python: {exc}") from exc
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in tree.body
    )


def validate_citation_metadata(
    citation: str, app_version: str, source: str = "CITATION.cff"
) -> None:
    """Check the archive metadata a release permanently bakes in.

    Zenodo scrapes CITATION.cff when a release is published, so a stale version
    or date becomes the citation of record for that archived version and cannot
    be corrected in place afterwards. Nothing else in the release path reads
    this file, which is how it once drifted five releases behind the shipped
    app before anyone noticed.
    """
    citation_version = _single_match(CITATION_VERSION_RE, citation, source)
    if citation_version != app_version.removeprefix("v"):
        raise FullReleaseError(
            f"{source} version {citation_version!r} does not match {app_version!r}"
        )

    released = _single_match(CITATION_DATE_RE, citation, source)
    try:
        released_date = date.fromisoformat(released)
    except ValueError as exc:
        raise FullReleaseError(f"{source} date-released {released!r} is not a valid date") from exc
    if released_date > date.today():
        raise FullReleaseError(f"{source} date-released {released} is in the future")


def validate_full_candidate(repo: Path) -> tuple[str, str]:
    app_version = _single_match(
        APP_VERSION_RE,
        (repo / "app_src" / "__init__.py").read_text(encoding="utf-8"),
        "app_src/__init__.py",
    )
    setup_version = _single_match(
        SETUP_VERSION_RE,
        (repo / "setup.py").read_text(encoding="utf-8"),
        "setup.py",
    )
    if setup_version != app_version.removeprefix("v"):
        raise FullReleaseError(f"setup.py version {setup_version!r} does not match {app_version!r}")

    validate_citation_metadata(
        (repo / "CITATION.cff").read_text(encoding="utf-8"),
        app_version,
    )

    changelog = (repo / "change_log.txt").read_text(encoding="utf-8")
    if not re.search(rf"(?m)^#### {re.escape(app_version)}[ \t]*$", changelog):
        raise FullReleaseError(f"change_log.txt has no {app_version} release heading")

    config_source = (repo / "app_src" / "config.py").read_text(encoding="utf-8")
    if not _has_assignment(config_source, "STAGE_COLORS"):
        raise FullReleaseError("app_src/config.py has no STAGE_COLORS assignment")

    requirements = (repo / "requirements.txt").read_text(encoding="utf-8")
    updater_commit = _single_match(
        PINNED_UPDATER_RE,
        requirements,
        "requirements.txt updater dependency",
    )
    return app_version, updater_commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    version, updater_commit = validate_full_candidate(args.repo.resolve())
    print(f"{version}\t{updater_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
