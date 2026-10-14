from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


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


def test_fully_labelled_recording_can_change_every_default_prediction():
    from app_src import run_inference_stats_model as model

    features = model.StatsModelFeatures(
        start_time=0.0,
        end_time=100.0,
        column_times=np.arange(100) + 0.5,
        low_band_means=np.full(100, 0.92),
        normalization_range=(0.0, 1.0),
        ne_for_rem=None,
        time_ne=None,
    )
    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, 100.0)),
        patch.object(model, "compute_stats_model_features", return_value=features),
    ):
        config, label_count = model.calibrate_stats_model_config({}, [0] * 100)

    assert label_count == 100
    prediction = model.predict_stats_model_from_features(features, config).sleep_scores
    np.testing.assert_array_equal(prediction, 0)


def test_sparse_wake_example_does_not_collapse_unlabelled_nrem():
    from app_src import run_inference_stats_model as model
    from app_src.sleep_score_layers import overlay_user_sleep_scores

    # A 27-second Wake candidate grows into a 33-second REM candidate, then
    # becomes Wake again after NE recovery in a recording-wide candidate.
    # This reproduces the nonmonotonic local fit behind the cached report.
    n = 1033
    features = model.StatsModelFeatures(
        start_time=0.0,
        end_time=float(n),
        column_times=np.arange(n) + 0.5,
        low_band_means=np.r_[np.full(1000, 0.90), np.full(6, 0.78), np.full(27, 0.74)],
        normalization_range=(0.0, 1.0),
        ne_for_rem=np.r_[np.zeros(40), np.ones(960), np.zeros(33)],
        time_ne=np.arange(n) + 0.5,
    )
    labels = np.r_[np.full(1000, np.nan), np.zeros(33)]
    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, float(n))),
        patch.object(model, "compute_stats_model_features", return_value=features),
    ):
        config, label_count = model.calibrate_stats_model_config({}, labels)

    prediction = model.predict_stats_model_from_features(features, config).sleep_scores
    assert label_count == 33
    assert np.count_nonzero(prediction[:1000] == 1) >= 900
    assert np.count_nonzero(prediction[1000:] == 0) >= 27
    # Raw calibration can compromise on fit; explicit annotations still win.
    displayed = overlay_user_sleep_scores(prediction, labels)
    np.testing.assert_array_equal(displayed[1000:], 0)


def test_labelled_stages_are_balanced_when_their_counts_differ():
    from app_src import run_inference_stats_model as model

    features = model.StatsModelFeatures(
        0.0, 101.0, np.array([0.5]), np.array([0.5]), (0.0, 1.0), None, None
    )
    labels = np.r_[np.ones(100), np.zeros(1)]

    def predict(_features, config):
        scores = np.ones(101, dtype=int)
        if config.wake_threshold >= 0.8:
            scores[:10] = 0
            scores[-1] = 0
        return SimpleNamespace(sleep_scores=scores)

    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, 101.0)),
        patch.object(model, "compute_stats_model_features", return_value=features),
        patch.object(model, "predict_stats_model_from_features", side_effect=predict),
    ):
        config, _ = model.calibrate_stats_model_config({}, labels)

    # Learning the Wake example outweighs a 10% error rate on NREM, despite
    # making more total mistakes than ignoring the single Wake second.
    assert predict(features, config).sleep_scores[-1] == 0


@pytest.mark.parametrize("other_labels", [np.nan, -1, 3], ids=["missing", "sentinel", "ma"])
def test_change_penalty_applies_only_to_unlabelled_seconds(other_labels):
    from app_src import run_inference_stats_model as model

    n = 133
    features = model.StatsModelFeatures(
        0.0, float(n), np.array([0.5]), np.array([0.5]), (0.0, 1.0), None, None
    )
    labels = np.r_[np.full(100, other_labels), np.zeros(33)]

    def predict(_features, config):
        scores = np.ones(n, dtype=int)
        if config.wake_threshold >= 0.94:
            scores[:] = 0
        elif config.wake_threshold >= 0.74:
            scores[100:127] = 0
        return SimpleNamespace(sleep_scores=scores)

    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, float(n))),
        patch.object(model, "compute_stats_model_features", return_value=features),
        patch.object(model, "predict_stats_model_from_features", side_effect=predict),
    ):
        config, label_count = model.calibrate_stats_model_config({}, labels)

    raw = predict(features, config).sleep_scores
    assert label_count == 33
    # MA annotations are manually specified, so there is no unlabelled
    # evidence to preserve. Missing placeholders leave the prior in play.
    assert np.count_nonzero(raw[100:] == 0) == (33 if other_labels == 3 else 27)


def test_one_real_wake_label_is_learned_without_drift_or_label_mutation():
    from app_src import run_inference_stats_model as model

    n = 110
    features = model.StatsModelFeatures(
        0.0,
        float(n),
        np.arange(n) + 0.5,
        np.r_[np.ones(100), np.full(10, 0.8)],
        (0.0, 1.0),
        None,
        None,
    )
    labels = np.full(n, np.nan)
    labels[105] = 0
    original = labels.copy()
    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, float(n))),
        patch.object(model, "compute_stats_model_features", return_value=features),
    ):
        first, count = model.calibrate_stats_model_config({}, labels)
        second, _ = model.calibrate_stats_model_config({}, labels.tolist())

    assert count == 1
    assert first == second
    assert model.predict_stats_model_from_features(features, first).sleep_scores[105] == 0
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize(
    ("majority_count", "minority_stage", "minority_count", "with_ne"),
    [(100, 0, 10, False), (400, 2, 35, True)],
    ids=["rare-wake", "rare-rem"],
)
def test_calibration_learns_minority_stage_examples(
    majority_count, minority_stage, minority_count, with_ne
):
    from app_src import run_inference_stats_model as model

    # Correct NREM labels dominate; the minority-stage example is only matched
    # by raising the Wake threshold above its signal level.
    n = majority_count + minority_count
    features = model.StatsModelFeatures(
        start_time=0.0,
        end_time=float(n),
        column_times=np.arange(n) + 0.5,
        low_band_means=np.r_[np.ones(majority_count), np.full(minority_count, 0.8)],
        normalization_range=(0.0, 1.0),
        ne_for_rem=np.r_[np.ones(majority_count), np.zeros(minority_count)] if with_ne else None,
        time_ne=np.arange(n) + 0.5 if with_ne else None,
    )
    labels = np.r_[np.ones(majority_count), np.full(minority_count, float(minority_stage))]

    with (
        patch.object(model, "eeg_time_range", return_value=(0.0, float(n))),
        patch.object(model, "compute_stats_model_features", return_value=features),
    ):
        config, _ = model.calibrate_stats_model_config({}, labels)

    prediction = model.predict_stats_model_from_features(features, config).sleep_scores
    np.testing.assert_array_equal(prediction[majority_count:], minority_stage)
    np.testing.assert_array_equal(prediction[:majority_count], 1)


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
