# -*- coding: utf-8 -*-
"""Lightweight local `.env` loading for desktop-app configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _strip_optional_quotes(value: str) -> str:
    """Remove one matching pair of wrapping single or double quotes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _candidate_search_dirs() -> list[Path]:
    """Return likely locations for a project- or app-local `.env` file."""
    candidates = []

    cwd = Path.cwd()
    candidates.append(cwd)

    project_root = Path(__file__).resolve().parents[1]
    candidates.append(project_root)

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        unique_candidates.append(resolved_candidate)

    return unique_candidates


def load_local_env(
    filename: str = ".env",
    override: bool = False,
    search_dirs: Iterable[str | Path] | None = None,
) -> Path | None:
    """
    Load environment variables from the first matching local `.env` file.

    The loader is intentionally simple:
    - blank lines and `#` comments are ignored
    - `export KEY=value` is supported
    - only the first `=` splits a line
    - surrounding single or double quotes are stripped
    - existing environment values are preserved unless `override=True`
    """
    if search_dirs is None:
        candidate_dirs = _candidate_search_dirs()
    else:
        candidate_dirs = [Path(directory).resolve() for directory in search_dirs]

    for directory in candidate_dirs:
        env_path = directory / filename
        if not env_path.is_file():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[7:].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = _strip_optional_quotes(value.strip())
            if not override and key in os.environ:
                continue

            os.environ[key] = value

        return env_path

    return None
