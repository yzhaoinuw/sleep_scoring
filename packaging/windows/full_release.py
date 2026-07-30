"""Preflight validation for a full Windows Sleep Scoring App release."""

from __future__ import annotations

import argparse
import ast
import re
from datetime import date, datetime
from pathlib import Path

import yaml


APP_VERSION_RE = re.compile(r"""VERSION\s*=\s*["']([^"']+)["']""")
SETUP_VERSION_RE = re.compile(r"""(?m)^[ \t]*version[ \t]*=[ \t]*["']([^"']+)["'][ \t]*,[ \t]*$""")
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


def _citation_release_date(value: object, source: str) -> date:
    # PyYAML resolves an unquoted ISO date to a date object and a timestamp to
    # a datetime; a quoted one stays a string. Accept all three spellings.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise FullReleaseError(f"{source} date-released {value!r} is not a valid date") from exc
    raise FullReleaseError(f"{source} has no usable date-released")


def validate_citation_metadata(
    citation: str, app_version: str, source: str = "CITATION.cff"
) -> None:
    """Check the citation metadata a release publishes.

    Zenodo scrapes CITATION.cff when a release is published. A published
    record's metadata can be edited afterwards without changing its DOI, so a
    stale version here is recoverable, but only by hand on Zenodo, per record,
    and fixing the repository file later does not propagate to records already
    published. Catching it before the tag is far cheaper than correcting it
    after.

    Nothing else in the release path reads this file, which is how it once
    drifted five releases behind the shipped app before anyone noticed.

    The document is parsed rather than pattern-matched so that a CITATION.cff
    Zenodo cannot read fails here instead of at publication.
    """
    try:
        document = yaml.safe_load(citation)
    except yaml.YAMLError as exc:
        raise FullReleaseError(f"{source} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise FullReleaseError(f"{source} must be a YAML mapping")

    citation_version = document.get("version")
    if not isinstance(citation_version, (str, int, float)):
        raise FullReleaseError(f"{source} has no usable version")
    if str(citation_version) != app_version.removeprefix("v"):
        raise FullReleaseError(
            f"{source} version {str(citation_version)!r} does not match {app_version!r}"
        )

    released_date = _citation_release_date(document.get("date-released"), source)
    if released_date > date.today():
        raise FullReleaseError(
            f"{source} date-released {released_date.isoformat()} is in the future"
        )


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
