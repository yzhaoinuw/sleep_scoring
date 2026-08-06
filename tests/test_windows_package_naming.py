import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "package_naming.py"
SPEC = importlib.util.spec_from_file_location("package_naming", MODULE_PATH)
PACKAGE_NAMING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PACKAGE_NAMING)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("v0.17.0", "v0.17"),
        ("v0.17.4", "v0.17"),
        ("v1.2", "v1.2"),
        ("v2.3.0-rc.1", "v2.3"),
    ],
)
def test_release_line_uses_only_major_and_minor(version, expected):
    assert PACKAGE_NAMING.release_line(version) == expected


@pytest.mark.parametrize("version", ["0.17.0", "v0", "v0.17.x", "release-v0.17.0"])
def test_release_line_rejects_unsupported_versions(version):
    with pytest.raises(ValueError, match="unsupported app version"):
        PACKAGE_NAMING.release_line(version)
