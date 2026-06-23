"""Tests for app_src/mat_utils.py"""

from app_src.mat_utils import get_ne_frequency


class TestGetNeFrequency:
    """Tests for the ne_frequency / fp_frequency alias resolution."""

    def test_prefers_ne_frequency(self):
        """The canonical ne_frequency field is returned when present."""
        assert get_ne_frequency({"ne_frequency": 10}) == 10

    def test_falls_back_to_fp_frequency(self):
        """fp_frequency is used when ne_frequency is absent."""
        assert get_ne_frequency({"fp_frequency": 20}) == 20

    def test_ne_frequency_takes_precedence(self):
        """ne_frequency wins when both fields are present."""
        assert get_ne_frequency({"ne_frequency": 10, "fp_frequency": 20}) == 10

    def test_falls_back_when_ne_frequency_is_none(self):
        """A present-but-None ne_frequency still falls back to fp_frequency."""
        assert get_ne_frequency({"ne_frequency": None, "fp_frequency": 20}) == 20

    def test_returns_none_when_neither_present(self):
        """None is returned when neither field exists."""
        assert get_ne_frequency({"eeg_frequency": 512}) is None

    def test_does_not_mutate_input(self):
        """The helper is read-only and must not write an alias back."""
        mat = {"ne": [1, 2, 3], "fp_frequency": 20}
        get_ne_frequency(mat)
        assert "ne_frequency" not in mat
