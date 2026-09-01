"""Regression guard for the restored, tracking-free home screen."""

from app_src import components


def test_home_screen_has_no_usage_tracking_controls():
    home_layout = str(components.home_div)

    assert "Click here to select a mat file" in home_layout
    assert "usage" not in home_layout.lower()
    assert "impact summary" not in home_layout.lower()
