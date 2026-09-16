from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.parametrize("missing_layer", [None, []])
def test_absent_annotation_layer_uses_saved_scores(missing_layer):
    from app_src.sleep_score_layers import saved_user_sleep_scores

    mat = {"sleep_scores": [0, 1, 2, 3, 4, 5, -1], "user_sleep_scores": missing_layer}
    np.testing.assert_array_equal(
        saved_user_sleep_scores(mat, 8), [0, 1, 2, 3, 4, 5, np.nan, np.nan]
    )


def test_explicit_empty_annotations_do_not_adopt_predictions_on_reload():
    from app_src.sleep_score_layers import saved_user_sleep_scores

    mat = {"sleep_scores": [4, 5, 1], "user_sleep_scores": [np.nan] * 3}
    assert np.isnan(saved_user_sleep_scores(mat, 3)).all()


def test_overlay_user_sleep_scores_preserves_only_finite_manual_labels():
    from app_src.sleep_score_layers import overlay_user_sleep_scores

    model_scores = np.array([0, 1, 2, 1], dtype=int)
    combined = overlay_user_sleep_scores(model_scores, [np.nan, 2, None])

    np.testing.assert_array_equal(combined, [0, 2, 2, 1])
    np.testing.assert_array_equal(model_scores, [0, 1, 2, 1])


@pytest.mark.parametrize("label", [0, 4, 5])
def test_calibration_uses_one_user_label_before_any_overlay(label):
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
        config, label_count = calibrate_stats_model_config({}, [label])

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


def test_stats_model_config_uses_all_user_facing_defaults():
    from app_src.config import (
        STATS_MODEL_MIN_REM_DURATION,
        STATS_MODEL_MIN_WAKE_DURATION,
        STATS_MODEL_REM_THRESHOLD_COMPARISON_PERCENTILE,
        STATS_MODEL_REM_THRESHOLD_PERCENTILE,
        STATS_MODEL_WAKE_THRESHOLD,
    )
    from app_src.run_inference_stats_model import StatsModelConfig

    config = StatsModelConfig()

    assert config.wake_threshold == STATS_MODEL_WAKE_THRESHOLD
    assert config.min_wake_duration == STATS_MODEL_MIN_WAKE_DURATION
    assert config.min_rem_duration == STATS_MODEL_MIN_REM_DURATION
    assert config.rem_threshold_percentile == STATS_MODEL_REM_THRESHOLD_PERCENTILE
    assert (
        config.rem_threshold_comparison_percentile
        == STATS_MODEL_REM_THRESHOLD_COMPARISON_PERCENTILE
    )
