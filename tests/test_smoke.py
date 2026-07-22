"""Smoke tests to verify basic imports and module loading."""

from types import SimpleNamespace


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
        assert len(config.STAGE_COLORS) == 4

    def test_stage_colors_support_updated_and_preserved_configs(self):
        """New configs customize colors while pre-v0.16.7 configs use defaults."""
        from app_src import make_figure

        custom_colors = ["red", "blue", "green", "yellow"]

        assert make_figure.get_stage_colors(SimpleNamespace()) == (make_figure.DEFAULT_STAGE_COLORS)
        assert (
            make_figure.get_stage_colors(SimpleNamespace(STAGE_COLORS=custom_colors))
            is custom_colors
        )
        assert make_figure.STAGE_COLORS == make_figure.get_stage_colors()
        assert [color for _, color in make_figure.COLORSCALE[4]] == (make_figure.STAGE_COLORS)

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

    def test_import_make_figure(self):
        """Test make_figure module imports."""
        from app_src import make_figure

        assert hasattr(make_figure, "make_figure")
        assert hasattr(make_figure, "get_padded_sleep_scores")
