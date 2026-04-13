"""Tests for the ChatGPT inference backend."""

import json
import sys
import uuid
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


class FakeFigure:
    """Minimal figure stub that records title updates."""

    def __init__(self):
        self.title_updates = []

    def update_layout(self, title=None, **_kwargs):
        if isinstance(title, dict):
            self.title_updates.append(title.get("text"))


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
    mat["_source_filename"] = "trace_demo"
    client = MockResponsesClient(
        {
            "summary": "Clear wake at the start of the session.",
            "bouts": [
                {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
            ],
            "uncertain_intervals": [],
        }
    )

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: object(),
    )
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
        refinement_mode="none",
    )

    assert len(client.calls) == 1
    assert np.all(predictions[:20] == 0)
    assert np.isnan(predictions[20:]).all()
    assert np.all(confidence[:20] == 0.95)
    assert np.isnan(confidence[20:]).all()


def test_chatgpt_backend_uses_focused_figure_mode_by_default(
    mock_mat_data,
    monkeypatch,
):
    """The current default should keep using the focused ChatGPT export figure."""
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

    focused_calls = []

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: focused_calls.append(kwargs.get("plot_name")) or object(),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "make_figure",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full figure should not be used")
        ),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )

    chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_focused_mode",
        confidence_threshold=0.7,
        refinement_mode="none",
    )

    assert len(focused_calls) == 1


def test_chatgpt_backend_can_use_full_figure_mode_for_backend_comparison(
    mock_mat_data,
    monkeypatch,
):
    """A backend-only full mode should reuse the original 4-panel export builder."""
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

    full_calls = []

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("focused figure should not be used in full mode")
        ),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "make_figure",
        lambda *args, **kwargs: full_calls.append(kwargs.get("plot_name")) or object(),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )

    chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_full_mode",
        confidence_threshold=0.7,
        refinement_mode="none",
        vision_figure_mode="full",
    )

    assert len(full_calls) == 1


def test_chatgpt_backend_refines_uncertain_interval_with_zoom_snapshot_and_features(
    mock_mat_data,
    monkeypatch,
):
    """The backend should run a bounded second pass for uncertain local intervals."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    mat["_source_filename"] = "trace_demo"
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

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: object(),
    )
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
    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_refinement"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
        confidence_threshold=0.7,
        refinement_mode="adaptive",
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
    assert coarse_request["input"][0]["content"].startswith("# ChatGPT Sleep Scoring Guidance")
    assert "Refine only this local interval" in refinement_request["input"][1]["content"][0]["text"]
    assert (
        "Base the decision only on the image"
        in refinement_request["input"][1]["content"][0]["text"]
    )
    assert "interval_features=" not in refinement_request["input"][1]["content"][0]["text"]
    assert "current_scores=" not in refinement_request["input"][1]["content"][0]["text"]
    assert any(
        part["type"] == "input_image" and part["image_url"].startswith("data:image/png;base64,")
        for part in refinement_request["input"][1]["content"]
    )


def test_chatgpt_backend_writes_trace_file_when_show_thoughts_is_enabled(
    mock_mat_data,
    monkeypatch,
):
    """Opt-in trace logging should write a readable txt file next to snapshots."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    mat["_source_filename"] = "trace_demo"
    client = MockResponsesClient(
        {
            "summary": "Clear wake at the start of the session.",
            "bouts": [
                {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
            ],
            "uncertain_intervals": [],
        }
    )

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )

    snapshot_dir = Path("test-artifacts") / f"chatgpt_inference_trace_{uuid.uuid4().hex}"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
        confidence_threshold=0.7,
        show_thoughts=True,
        refinement_mode="none",
    )

    trace_files = sorted(snapshot_dir.glob("*_thoughts.txt"))

    assert len(trace_files) == 1
    assert trace_files[0].name == "trace_demo_thoughts.txt"
    trace_text = trace_files[0].read_text(encoding="utf-8")
    assert "## Trace Info" in trace_text
    assert "## Coarse Pass Summary" in trace_text
    assert "## Coarse Pass Proposed Bouts" in trace_text
    assert "Clear wake at the start of the session." in trace_text
    assert "## Coarse Pass Applied Bouts" in trace_text
    assert "## Guidance Prompt" not in trace_text
    assert "## Coarse Pass Input" not in trace_text
    assert "data:image/png;base64" not in trace_text
    assert np.all(predictions[:20] == 0)
    assert np.all(confidence[:20] == 0.95)


def test_chatgpt_backend_skips_refinement_in_none_mode_even_with_uncertain_intervals(
    mock_mat_data,
    monkeypatch,
):
    """The overview-only mode should skip second-pass zoom refinement."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    client = MockResponsesClient(
        {
            "summary": "Wake first, REM later, unclear middle interval.",
            "bouts": [
                {"start_s": 0, "end_s": 20, "state": "Wake", "confidence": 0.95},
                {"start_s": 40, "end_s": 60, "state": "REM", "confidence": 0.9},
            ],
            "uncertain_intervals": [
                {"start_s": 20, "end_s": 40, "reason": "boundary-heavy middle segment"},
            ],
        }
    )

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: object(),
    )
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

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_overview_only",
        confidence_threshold=0.7,
        refinement_mode="none",
    )

    assert len(client.calls) == 1
    assert np.all(predictions[:20] == 0)
    assert np.isnan(predictions[20:40]).all()
    assert np.all(predictions[40:60] == 2)
    assert np.isnan(confidence[20:40]).all()


def test_chatgpt_backend_can_refine_using_fixed_broad_sections(
    mock_mat_data,
    monkeypatch,
):
    """Fixed broad sections should produce a bounded second pass over coarse intervals."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    mat["_source_filename"] = "fixed_sections_demo"
    client = MockResponsesClient(
        [
            {
                "summary": "Coarse pass leaves the session uncertain.",
                "bouts": [],
                "uncertain_intervals": [],
            },
            {
                "summary": "Broad section keeps the interval uncertain.",
                "bouts": [],
                "uncertain_intervals": [],
            },
        ]
    )

    fake_figure = FakeFigure()
    zoom_titles = []

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: fake_figure,
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_zoom_snapshot",
        lambda fig, _start_s, _end_s, output_path: zoom_titles.append(fig.title_updates[-1])
        or _fake_write_snapshot(output_path),
    )
    chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_fixed_sections",
        confidence_threshold=0.7,
        refinement_mode="fixed_sections",
        fixed_refinement_section_count=4,
    )

    assert len(client.calls) == 5
    assert zoom_titles == [
        "fixed_sections_demo | 0s-25s",
        "fixed_sections_demo | 25s-50s",
        "fixed_sections_demo | 50s-75s",
        "fixed_sections_demo | 75s-100s",
    ]


def test_chatgpt_backend_uses_recording_name_and_interval_in_snapshot_titles(
    mock_mat_data,
    monkeypatch,
):
    """Overview and zoom snapshots should use the source mat name plus interval bounds."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    mat["_source_filename"] = "115_gs"
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
                "uncertain_intervals": [],
            },
        ]
    )

    fake_figure = FakeFigure()
    observed_plot_names = []
    zoom_titles = []

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda _mat, plot_name="", **_kwargs: observed_plot_names.append(plot_name) or fake_figure,
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_overview_snapshot",
        lambda _fig, output_path: _fake_write_snapshot(output_path),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_zoom_snapshot",
        lambda fig, _start_s, _end_s, output_path: zoom_titles.append(fig.title_updates[-1])
        or _fake_write_snapshot(output_path),
    )
    chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_titles",
        confidence_threshold=0.7,
        refinement_mode="adaptive",
    )

    assert observed_plot_names == ["115_gs | 0s-100s"]
    assert zoom_titles == ["115_gs | 5s-55s"]


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

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: object(),
    )
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
