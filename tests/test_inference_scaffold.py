"""Tests for the lightweight ChatGPT inference scaffold."""

import numpy as np


def test_chatgpt_backend_preserves_existing_scores(mock_mat_data):
    """ChatGPT scaffold should keep current labels until API wiring is added."""
    from app_src.inference import run_inference

    original_scores = mock_mat_data["sleep_scores"].copy()
    updated_mat, _ = run_inference(mock_mat_data, backend="chatgpt", postprocess=False)

    assert np.array_equal(updated_mat["sleep_scores"], original_scores)
    assert updated_mat["confidence"].shape == original_scores.shape
    assert np.isnan(updated_mat["confidence"]).all()
