from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def test_overlay_user_sleep_scores_preserves_only_finite_manual_labels():
    from app_src.sleep_score_layers import overlay_user_sleep_scores

    model_scores = np.array([0, 1, 2, 1], dtype=int)
    combined = overlay_user_sleep_scores(model_scores, [np.nan, 2, None])

    np.testing.assert_array_equal(combined, [0, 2, 2, 1])
    np.testing.assert_array_equal(model_scores, [0, 1, 2, 1])


def test_calibration_uses_one_user_label_before_any_overlay():
    from app_src.run_inference_stats_model import (
        StatsModelFeatures,
        calibrate_stats_model_config,
    )

    features = StatsModelFeatures(
        start_time=0.0,
        end_time=1.0,
        column_times=np.array([0.5]),
        low_band_means=np.array([0.5]),
        normalization_range=(0.0, 1.0),
        ne_for_rem=None,
        time_ne=None,
    )

    def prediction_for_threshold(_features, config):
        # A lower threshold is the only way to match the supplied Wake label.
        stage = 0 if config.wake_threshold <= 0.4 else 1
        return SimpleNamespace(sleep_scores=np.array([stage], dtype=int))

    with (
        patch("app_src.run_inference_stats_model.eeg_time_range", return_value=(0.0, 1.0)),
        patch(
            "app_src.run_inference_stats_model.compute_stats_model_features",
            return_value=features,
        ),
        patch(
            "app_src.run_inference_stats_model.predict_stats_model_from_features",
            side_effect=prediction_for_threshold,
        ),
    ):
        config, label_count = calibrate_stats_model_config({}, [0])

    assert label_count == 1
    assert config.wake_threshold <= 0.4


def test_calibration_with_no_user_labels_keeps_defaults_without_feature_work():
    from app_src.run_inference_stats_model import (
        StatsModelConfig,
        calibrate_stats_model_config,
    )

    with patch("app_src.run_inference_stats_model.compute_stats_model_features") as features:
        config, label_count = calibrate_stats_model_config(
            {"eeg": np.zeros(2), "eeg_frequency": 1}, [np.nan]
        )

    assert config == StatsModelConfig()
    assert label_count == 0
    features.assert_not_called()
