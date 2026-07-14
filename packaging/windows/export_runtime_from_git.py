"""Export tracked runtime files as their exact Git-blob bytes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath


def git_output(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or "git command failed")
    return result.stdout


def tracked_paths(repo: Path, ref: str, runtime_path: str) -> list[str]:
    output = git_output(repo, "ls-tree", "-r", "--name-only", "-z", ref, "--", runtime_path)
    return [path.decode() for path in output.split(b"\0") if path]


def export_runtime(repo: Path, ref: str, runtime_path: str, destination: Path) -> list[str]:
    paths = tracked_paths(repo, ref, runtime_path)
    if not paths:
        raise ValueError(f"no tracked files found at {ref}:{runtime_path}")

    for path in paths:
        relative_path = PurePosixPath(path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe tracked path: {path}")
        output_path = destination.joinpath(*relative_path.parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(git_output(repo, "show", f"{ref}:{path}"))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export tracked runtime files without checkout transformations."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    export_runtime(args.repo, args.ref, args.runtime_path, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
