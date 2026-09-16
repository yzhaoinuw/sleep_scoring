"""Exercise real calibration, subdivision, and MAT persistence together."""

from collections import deque
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest
from plotly.utils import PlotlyJSONEncoder
from scipy.io import loadmat, savemat

from app_src.sleep_score_layers import coarse_sleep_scores, saved_user_sleep_scores
from app_src.wake_activity import subdivide_wake


@pytest.fixture
def recording():
    fs = 512
    time = np.arange(fs * 40) / fs
    rng = np.random.default_rng(90)
    emg = 0.03 * rng.normal(size=time.size) + 0.02 * np.sin(2 * np.pi * 80 * time)
    emg += ((time >= 15) & (time < 30)) * 2 * np.sin(2 * np.pi * 80 * time)
    return {
        "eeg": rng.normal(size=time.size),
        "emg": emg,
        "eeg_frequency": fs,
        "ne": 1 + np.sin(np.arange(400) / 30),
        "ne_frequency": 10,
        "sleep_scores": np.r_[np.zeros(30), np.ones(5), np.full(5, 2)],
    }


def test_sleep_calibration_and_prediction_are_repeatable_and_ignore_subtypes(recording):
    from app_src.run_inference_stats_model import calibrate_stats_model_config, infer

    labels = np.full(40, np.nan)
    labels[[2, 20, 32, 37]] = [5, 4, 1, 2]
    original = deepcopy(recording)
    settings, count = calibrate_stats_model_config(recording, labels)
    first, _ = infer(recording, config=settings)
    # Even a previous automatic result stored in the MAT must not become input.
    recording["sleep_scores"] = np.full(40, 5)
    repeated_settings, repeated_count = calibrate_stats_model_config(recording, labels)
    second, _ = infer(recording, config=repeated_settings)
    coarse_settings, coarse_count = calibrate_stats_model_config(
        recording, coarse_sleep_scores(labels)
    )
    assert settings == repeated_settings == coarse_settings
    assert count == repeated_count == coarse_count == 4
    np.testing.assert_array_equal(first, second)
    for signal in ("eeg", "emg", "ne"):
        np.testing.assert_array_equal(recording[signal], original[signal])


@pytest.mark.parametrize("fine_labels", [False, True])
def test_wake_subdivision_is_deterministic_and_idempotent(recording, fine_labels):
    users = recording["sleep_scores"].copy()
    if fine_labels:
        users[2:6], users[20:26] = 5, 4
    original_users = users.copy()
    first = subdivide_wake(recording["emg"], 512, recording["sleep_scores"], users)
    repeat = subdivide_wake(recording["emg"], 512, recording["sleep_scores"], users)
    reapplied = subdivide_wake(recording["emg"], 512, first.sleep_scores, users)
    for result in (repeat, reapplied):
        np.testing.assert_array_equal(first.sleep_scores, result.sleep_scores)
        np.testing.assert_array_equal(first.envelope, result.envelope)
        assert first.metadata == result.metadata
    assert 0 not in first.sleep_scores
    assert {4, 5} <= set(first.sleep_scores)
    np.testing.assert_array_equal(users, original_users)


@pytest.mark.parametrize("fine_labels", [False, True])
def test_existing_mat_prediction_save_reload_repeats_without_feedback(
    recording, fine_labels, tmp_path
):
    from app_src.callbacks.loading import create_visualization
    from app_src.callbacks.prediction import generate_prediction, read_mat_pred
    from app_src.callbacks.saving import save_annotations

    if fine_labels:
        recording["sleep_scores"][2:6] = 5
        recording["sleep_scores"][20:26] = 4
    input_path, output_path = tmp_path / "input.mat", tmp_path / "output.mat"
    savemat(input_path, recording)
    values = {
        "filepath": str(input_path),
        "filename": "input",
        "sleep_scores_history": deque(maxlen=2),
        "user_sleep_scores_history": deque(maxlen=2),
    }
    with (
        patch("app_src.config.STATS_MODEL_DETECT_WAKE_ACTIVITY", True),
        patch("app_src.callbacks.prediction.SLEEP_SCORING_MODEL", "stats_model"),
        patch("app_src.inference.SLEEP_SCORING_MODEL", "stats_model"),
        patch("app_src.server.cache.get", side_effect=values.get),
        patch(
            "app_src.server.cache.set", side_effect=lambda key, value: values.update({key: value})
        ),
        patch("app_src.callbacks.loading.create_fig"),
        patch("app_src.callbacks.loading.components"),
        patch("app_src.callbacks.saving.TEMP_PATH", tmp_path),
        patch("app_src.callbacks.saving.save_file_dialog", side_effect=[str(output_path), None]),
        patch("app_src.callbacks.saving.record_scored_recording"),
    ):
        _, _, users = create_visualization("vis")
        np.testing.assert_array_equal(users, recording["sleep_scores"])
        request = read_mat_pred(1, True, users)[2]
        first = generate_prediction(request)
        repeat = generate_prediction(request)
        assert first == repeat
        assert 0 not in first[1]
        np.testing.assert_array_equal(first[1][30:], recording["sleep_scores"][30:])
        assert f"{10 if fine_labels else 0} fine-labelled" in first[0]
        # Saving the displayed output must keep original calibration annotations.
        values["sleep_scores_history"].append(np.asarray(first[1]))
        save_annotations(1)
        loaded = loadmat(output_path, squeeze_me=True)
        np.testing.assert_array_equal(saved_user_sleep_scores(loaded, 40), users)
        np.testing.assert_array_equal(loaded["sleep_scores"], first[1])
        values["filepath"] = str(output_path)
        after_reload = generate_prediction({"user_sleep_scores": None})
        assert after_reload == first
        # Config alone disables subdivision; saved coarse Wake stays coarse.
        with patch("app_src.config.STATS_MODEL_DETECT_WAKE_ACTIVITY", False):
            disabled = generate_prediction(request)
        np.testing.assert_array_equal(disabled[1], users)


def test_layout_and_prediction_dependencies_have_no_pilot_controls():
    from app_src.app import app
    from app_src.components import Components

    layout = json.dumps(Components(pred_disabled=False).visualization_div, cls=PlotlyJSONEncoder)
    assert "wake-activity" not in layout
    assert "wake-envelope" not in layout
    assert "pred-button" in layout
    for callback in app.callback_map.values():
        for dependency in callback["inputs"] + callback["state"]:
            assert not dependency["id"].startswith(("wake-activity", "wake-envelope"))
