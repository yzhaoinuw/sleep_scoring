"""Tests for the ChatGPT inference backend."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class MockResponsesClient:
    """Minimal test double for the OpenAI Responses API."""

    def __init__(self, payloads):
        if isinstance(payloads, list):
            self.payloads = list(payloads)
        else:
            self.payloads = [payloads]
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload_index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return SimpleNamespace(output_text=json.dumps(self.payloads[payload_index]))


def _fake_write_snapshot(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake-png")
    return output_path


def test_chatgpt_backend_applies_confident_coarse_bouts_without_refinement(
    mock_mat_data,
    monkeypatch,
):
    """A single confident bout should write back directly from the overview pass."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    client = MockResponsesClient(
        {
            "summary": "Clear wake at the start of the session.",
            "bouts": [
                {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
            ],
            "uncertain_intervals": [],
        }
    )

    monkeypatch.setattr(chatgpt_inference, "make_figure", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )

    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_coarse_only"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
        confidence_threshold=0.7,
    )

    assert len(client.calls) == 1
    assert np.all(predictions[:20] == 0)
    assert np.isnan(predictions[20:]).all()
    assert np.all(confidence[:20] == 0.95)
    assert np.isnan(confidence[20:]).all()


def test_chatgpt_backend_refines_uncertain_interval_with_zoom_snapshot_and_features(
    mock_mat_data,
    monkeypatch,
):
    """The backend should run a bounded second pass for uncertain local intervals."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    client = MockResponsesClient(
        [
            {
                "summary": "Wake first, REM later, unclear middle interval.",
                "bouts": [
                    {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
                    {"start_s": 40, "end_s": 60, "state": "REM", "confidence": 0.9},
                ],
                "uncertain_intervals": [
                    {"start_s": 20, "end_s": 40, "reason": "boundary-heavy middle segment"},
                ],
            },
            {
                "summary": "The first half of the local interval is likely NREM.",
                "bouts": [
                    {"start_s": 20, "end_s": 30, "state": "NREM", "confidence": 0.88},
                ],
                "uncertain_intervals": [
                    {"start_s": 30, "end_s": 40, "reason": "remaining local ambiguity"},
                ],
            },
        ]
    )

    monkeypatch.setattr(chatgpt_inference, "make_figure", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_zoom_snapshot",
        lambda _fig, _start_s, _end_s, output_path: _fake_write_snapshot(output_path),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "get_interval_features",
        lambda *args, **kwargs: {
            "start_s": 20,
            "end_s": 40,
            "duration_s": 20,
            "theta_delta_ratio_mean_db": 1.2,
            "emg_rms": 0.3,
            "current_score_dominant_state": None,
        },
    )

    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_refinement"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
        confidence_threshold=0.7,
    )

    assert len(client.calls) == 2
    assert np.all(predictions[:20] == 0)
    assert np.all(predictions[20:30] == 1)
    assert np.isnan(predictions[30:40]).all()
    assert np.all(predictions[40:60] == 2)
    assert np.isnan(predictions[60:]).all()

    assert np.all(confidence[:20] == 0.95)
    assert np.all(confidence[20:30] == 0.88)
    assert np.isnan(confidence[30:40]).all()
    assert np.all(confidence[40:60] == 0.9)

    coarse_request = client.calls[0]
    refinement_request = client.calls[1]
    assert coarse_request["input"][0]["content"].startswith(
        "# ChatGPT Sleep Scoring Guidance Draft"
    )
    assert "Refine only this local interval" in refinement_request["input"][1]["content"][0]["text"]
    assert "interval_features=" in refinement_request["input"][1]["content"][0]["text"]
    assert "current_scores=" in refinement_request["input"][1]["content"][0]["text"]
    assert any(
        part["type"] == "input_image" and part["image_url"].startswith("data:image/png;base64,")
        for part in refinement_request["input"][1]["content"]
    )


def test_chatgpt_backend_preserves_existing_scores_when_client_is_unavailable(
    mock_mat_data,
    monkeypatch,
):
    """Missing SDK or API key should not break the current app flow."""
    from app_src import chatgpt_inference

    monkeypatch.setattr(chatgpt_inference, "_build_openai_client", lambda client=None: None)

    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_fallback"

    predictions, confidence = chatgpt_inference.infer(mock_mat_data, snapshot_dir=snapshot_dir)

    assert np.array_equal(predictions, mock_mat_data["sleep_scores"])
    assert np.isnan(confidence).all()


def test_chatgpt_backend_ready_status_reports_missing_api_key(monkeypatch):
    """The UI should get a clear message when the API key is not configured."""
    from app_src import chatgpt_inference

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    is_ready, message = chatgpt_inference.get_backend_ready_status()

    assert not is_ready
    assert "OPENAI_API_KEY" in message


def test_chatgpt_backend_ready_status_accepts_installed_sdk(monkeypatch):
    """A present SDK plus API key should mark the backend as ready."""
    from app_src import chatgpt_inference

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())

    is_ready, message = chatgpt_inference.get_backend_ready_status()

    assert is_ready
    assert "ready" in message.lower()


def test_chatgpt_backend_falls_back_when_structured_output_is_invalid(
    mock_mat_data,
    monkeypatch,
):
    """Malformed model output should revert to the safe baseline."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    client = MockResponsesClient(
        {
            "summary": "Invalid overlapping bout output.",
            "bouts": [
                {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
                {"start_s": 10, "end_s": 30, "state": "REM", "confidence": 0.85},
            ],
            "uncertain_intervals": [],
        }
    )

    monkeypatch.setattr(chatgpt_inference, "make_figure", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )

    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_invalid"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
    )

    assert np.isnan(predictions).all()
    assert np.isnan(confidence).all()
