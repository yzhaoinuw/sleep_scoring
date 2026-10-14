"""Copy a completed export without writing over the destination in place."""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)


def remove_temporary_export(path):
    """Cleanup must not turn a successful save into a reported failure."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary export %s", path, exc_info=True)


def _file_signature(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.digest()


def copy_verified_export(source, destination):
    """Stage, flush and verify beside destination, then replace it in one step.

    Never fall back to copying over the destination if replacement fails.
    The caller owns the local source export and retains it on failure.
    Replacement guarantees ultimately depend on the destination filesystem.
    """
    source = Path(source)
    destination = Path(destination)
    staging_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as staging:
            staging_path = Path(staging.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, staging, length=1024 * 1024)
            staging.flush()
            os.fsync(staging.fileno())

        if _file_signature(source) != _file_signature(staging_path):
            raise OSError("The copied file failed verification; the destination was not replaced.")

        os.replace(staging_path, destination)
    finally:
        if staging_path is not None:
            remove_temporary_export(staging_path)
