"""Smoke tests to verify basic imports and module loading."""


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

        assert hasattr(config, "PORT")

    def test_import_version(self):
        """Test version is accessible."""
        from app_src import VERSION

        assert isinstance(VERSION, str)
        assert len(VERSION) > 0


class TestAppImport:
    """Test that the Dash app can be imported."""

    def test_import_components(self):
        """Test components module imports."""
        from app_src.components_dev import Components

        # Should be able to instantiate without inference
        components = Components(pred_disabled=True)
        assert components is not None

    def test_import_make_figure(self):
        """Test make_figure module imports."""
        from app_src import make_figure_dev

        assert hasattr(make_figure_dev, "make_figure")
        assert hasattr(make_figure_dev, "get_padded_sleep_scores")
