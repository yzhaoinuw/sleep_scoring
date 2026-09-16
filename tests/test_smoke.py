"""Smoke tests to verify basic imports and module loading."""

import json
from types import SimpleNamespace

import pytest


class TestImports:
    """Test that all modules can be imported without errors."""

    def test_import_preprocessing(self):
        """Test preprocessing module imports."""
        from app_src import preprocessing

        assert hasattr(preprocessing, "trim_missing_labels")
        assert hasattr(preprocessing, "reshape_sleep_data")
        assert hasattr(preprocessing, "reshape_sleep_data_ne")

    def test_import_postprocessing(self):
        """Test postprocessing module imports."""
        from app_src import postprocessing

        assert hasattr(postprocessing, "get_sleep_segments")
        assert hasattr(postprocessing, "merge_consecutive_sleep_scores")
        assert hasattr(postprocessing, "edit_sleep_scores")
        assert hasattr(postprocessing, "get_pred_label_stats")

    def test_import_get_fft_plots(self):
        """Test FFT plots module imports."""
        from app_src import get_fft_plots

        assert hasattr(get_fft_plots, "get_fft_plots")

    def test_import_config(self):
        """Test config module imports."""
        from app_src import config

        assert hasattr(config, "INSTANCE_SLOT")
        assert hasattr(config, "PEER_PORTS")
        assert len(config.STAGE_COLORS) == 6

    def test_stage_colors_support_updated_and_preserved_configs(self):
        """New configs customize colors while pre-v0.16.7 configs use defaults."""
        from app_src import make_figure

        custom_colors = ["red", "blue", "green", "yellow"]

        assert make_figure.get_stage_colors(SimpleNamespace()) == (make_figure.DEFAULT_STAGE_COLORS)
        assert make_figure.get_stage_colors(SimpleNamespace(STAGE_COLORS=custom_colors)) == (
            custom_colors + make_figure.DEFAULT_STAGE_COLORS[4:]
        )
        assert make_figure.STAGE_COLORS == make_figure.get_stage_colors()
        assert [color for _, color in make_figure.COLORSCALE[6]] == (make_figure.STAGE_COLORS)

    def test_import_version(self):
        """Test version is accessible."""
        from app_src import VERSION

        assert isinstance(VERSION, str)
        assert len(VERSION) > 0


class TestAppImport:
    """Test that the Dash app can be imported."""

    def test_import_components(self):
        """Test components module imports."""
        from app_src.components import Components

        # Should be able to instantiate without inference
        components = Components(pred_disabled=True)
        assert components is not None
        message_ids = {
            child.id
            for child in components.visualization_div.children
            if getattr(child, "id", None) is not None
        }
        assert {"annotation-message", "prediction-message"} <= message_ids

    def test_import_make_figure(self):
        """Test make_figure module imports."""
        from app_src import make_figure

        assert hasattr(make_figure, "make_figure")
        assert hasattr(make_figure, "get_padded_sleep_scores")

    @pytest.mark.parametrize("fixture_name", ["mock_mat_data", "mock_mat_data_with_ne"])
    def test_figure_serializes_score_roles_for_clientside_callbacks(self, request, fixture_name):
        """Real figures expose all three score overlays even with optional NE."""
        from app_src.make_figure import make_figure

        mat = request.getfixturevalue(fixture_name)
        figure = json.loads(make_figure(mat).to_json())
        overlays = [
            trace for trace in figure["data"] if trace.get("meta", {}).get("role") == "sleep_scores"
        ]
        assert len(overlays) == 3
        assert {trace["yaxis"] for trace in overlays} == {"y3", "y4", "y5"}
        for trace in overlays:
            assert trace["type"] == "heatmap"
            assert trace["z"] == [mat["sleep_scores"].tolist()]
