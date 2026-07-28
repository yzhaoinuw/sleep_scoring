"""Naming helpers shared by Windows release packaging scripts."""

from __future__ import annotations

import argparse
import re


VERSION_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?$")


def release_line(version: str) -> str:
    """Return the stable major/minor label used for full-package folders."""
    match = VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported app version: {version}")
    return f"v{match.group('major')}.{match.group('minor')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args(argv)
    print(release_line(args.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
