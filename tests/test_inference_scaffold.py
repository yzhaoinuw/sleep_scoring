"""Tests for the ChatGPT inference backend."""

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class MockResponsesClient:
    """Minimal test double for the OpenAI Responses API."""

    def __init__(self, payloads, usages=None):
        if isinstance(payloads, list):
            self.payloads = list(payloads)
        else:
            self.payloads = [payloads]
        if usages is None:
            self.usages = [None] * len(self.payloads)
        elif isinstance(usages, list):
            self.usages = list(usages)
        else:
            self.usages = [usages]
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload_index = min(len(self.calls) - 1, len(self.payloads) - 1)
        usage_index = min(len(self.calls) - 1, len(self.usages) - 1)
        return SimpleNamespace(
            output_text=json.dumps(self.payloads[payload_index]),
            usage=self.usages[usage_index],
        )


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


def _fake_write_prediction_snapshot(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"prediction-png")
    return output_path


def _segment(start_s, end_s, state, confidence=0.9, reason="model visual cue"):
    return {
        "start_s": start_s,
        "end_s": end_s,
        "state": state,
        "reason": reason,
        "confidence": confidence,
    }


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
            "segments": [
                _segment(
                    0,
                    20,
                    "Wake",
                    confidence=0.95,
                    reason="clear wake at the start of the session",
                ),
            ],
        },
        usages={
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 180},
            "total_tokens": 1500,
        },
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
        use_overview_pass=True,
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
            "segments": [
                _segment(
                    0,
                    20,
                    "Wake",
                    confidence=0.95,
                    reason="clear wake at the start of the session",
                ),
            ],
        },
        usages={
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 180},
            "total_tokens": 1500,
        },
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
        use_overview_pass=True,
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
            "segments": [
                _segment(
                    0,
                    20,
                    "Wake",
                    confidence=0.95,
                    reason="clear wake at the start of the session",
                ),
            ],
        },
        usages={
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 180},
            "total_tokens": 1500,
        },
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
        use_overview_pass=True,
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
                "segments": [
                    _segment(0, 20, "Wake", confidence=0.95, reason="wake first"),
                    _segment(40, 60, "REM", confidence=0.9, reason="rem later"),
                ],
            },
            {
                "segments": [
                    _segment(20, 30, "Wake", confidence=0.88, reason="local wake strip"),
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
        use_overview_pass=True,
    )

    assert len(client.calls) == 2
    assert np.all(predictions[:20] == 0)
    assert np.all(predictions[20:30] == 0)
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
    assert (
        "sleep score the figure based on the provided guidance to the best of your judgment"
        in refinement_request["input"][1]["content"][0]["text"]
    )
    assert (
        "If there truly are some parts that you cannot resolve"
        in refinement_request["input"][1]["content"][0]["text"]
    )
    assert (
        "Refine only this local interval"
        not in refinement_request["input"][1]["content"][0]["text"]
    )
    assert (
        "Base the decision only on the image"
        not in refinement_request["input"][1]["content"][0]["text"]
    )
    assert "interval_features=" not in refinement_request["input"][1]["content"][0]["text"]
    assert "current_scores=" not in refinement_request["input"][1]["content"][0]["text"]
    assert any(
        part["type"] == "input_image" and part["image_url"].startswith("data:image/png;base64,")
        for part in refinement_request["input"][1]["content"]
    )


def test_chatgpt_backend_attaches_reference_examples_only_to_coarse_pass_and_forwards_reasoning_effort(
    mock_mat_data,
    monkeypatch,
):
    """The coarse pass should include the reference pack once and keep refinements lighter."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    client = MockResponsesClient(
        [
            {
                "segments": [
                    _segment(0, 20, "Wake", confidence=0.95, reason="wake first"),
                    _segment(40, 60, "REM", confidence=0.9, reason="rem later"),
                ],
            },
            {
                "segments": [
                    _segment(20, 30, "Wake", confidence=0.88, reason="local wake strip"),
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
    monkeypatch.setattr(
        chatgpt_inference,
        "_build_reference_examples_message",
        lambda _reference_examples_dir=None: {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "REFERENCE PACK"},
                {"type": "input_image", "image_url": "data:image/png;base64,cmVm"},
            ],
        },
    )

    chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_reference_examples",
        confidence_threshold=0.7,
        refinement_mode="adaptive",
        use_reference_examples=True,
        use_overview_pass=True,
        reasoning_effort="high",
    )

    coarse_request = client.calls[0]
    refinement_request = client.calls[1]

    assert coarse_request["reasoning"] == {"effort": "high"}
    assert refinement_request["reasoning"] == {"effort": "high"}
    assert len(coarse_request["input"]) == 3
    assert coarse_request["input"][1]["content"][0]["text"] == "REFERENCE PACK"
    assert len(refinement_request["input"]) == 2


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
            "segments": [
                _segment(
                    0,
                    20,
                    "Wake",
                    confidence=0.95,
                    reason="clear wake at the start of the session",
                ),
            ],
        },
        usages={
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 180},
            "total_tokens": 1500,
        },
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
        use_overview_pass=True,
    )

    trace_files = sorted(snapshot_dir.glob("*_thoughts.txt"))

    assert len(trace_files) == 1
    assert trace_files[0].name == "trace_demo_thoughts.txt"
    trace_text = trace_files[0].read_text(encoding="utf-8")
    assert "## Trace Info" in trace_text
    assert "## Coarse Pass API Usage" in trace_text
    assert "- reasoning_effort: medium" in trace_text
    assert "- overview_pass: True" in trace_text
    assert "- input_tokens: 1200" in trace_text
    assert "- cached_input_tokens: 200" in trace_text
    assert "- output_tokens: 300" in trace_text
    assert "- reasoning_tokens: 180" in trace_text
    assert "- total_tokens: 1500" in trace_text
    assert "- estimated_cost_usd: $0.007050" in trace_text
    assert "## Coarse Pass Proposed Segments" in trace_text
    assert "- 0s-20s | Wake | 0.95 | clear wake at the start of the session" in trace_text
    assert "## Coarse Pass Applied Segments" not in trace_text
    assert "Applied Segments" not in trace_text
    assert "## Coarse Pass Summary" not in trace_text
    assert "## Coarse Pass Proposed Bouts" not in trace_text
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
            "segments": [
                _segment(0, 20, "Wake", confidence=0.95, reason="wake first"),
                _segment(40, 60, "REM", confidence=0.9, reason="rem later"),
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
        use_overview_pass=True,
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
            {"segments": []},
            {"segments": []},
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
        use_overview_pass=True,
    )

    assert len(client.calls) == 5
    assert zoom_titles == [
        "fixed_sections_demo | 0s-25s",
        "fixed_sections_demo | 25s-50s",
        "fixed_sections_demo | 50s-75s",
        "fixed_sections_demo | 75s-100s",
    ]


def test_chatgpt_backend_defaults_to_zoom_sections_without_overview_or_examples(
    mock_mat_data,
    monkeypatch,
):
    """The current experiment should score fixed zoomed sections without a coarse overview."""
    from app_src import chatgpt_inference

    mat = {key: value for key, value in mock_mat_data.items() if key != "sleep_scores"}
    mat["_source_filename"] = "zoom_only_demo"
    client = MockResponsesClient(
        [
            {"segments": []},
            {
                "segments": [
                    _segment(
                        25,
                        35,
                        "Wake",
                        confidence=0.2,
                        reason="discontinuity in the yellow band",
                    ),
                ],
            },
            {
                "segments": [
                    _segment(
                        50,
                        60,
                        "REM",
                        confidence=0.88,
                        reason="spectrogram break with NE valley",
                    ),
                ],
            },
            {"segments": []},
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
        lambda _fig, _output_path: (_ for _ in ()).throw(
            AssertionError("overview snapshot should not be captured")
        ),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "_build_reference_examples_message",
        lambda _reference_examples_dir=None: (_ for _ in ()).throw(
            AssertionError("reference examples should not be attached by default")
        ),
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_zoom_snapshot",
        lambda fig, _start_s, _end_s, output_path: zoom_titles.append(fig.title_updates[-1])
        or _fake_write_snapshot(output_path),
    )

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=Path("test-artifacts") / "chatgpt_inference_zoom_only_default",
    )

    assert len(client.calls) == 4
    assert zoom_titles == [
        "zoom_only_demo | 0s-25s",
        "zoom_only_demo | 25s-50s",
        "zoom_only_demo | 50s-75s",
        "zoom_only_demo | 75s-100s",
    ]
    first_request_text = client.calls[0]["input"][1]["content"][0]["text"]
    assert (
        "sleep score the figure based on the provided guidance to the best of your judgment"
        in first_request_text
    )
    assert "If there truly are some parts that you cannot resolve" in first_request_text
    assert "Score only this zoomed section" not in first_request_text
    assert "No full-recording overview image is provided" not in first_request_text
    assert "Do not label any segment as uncertain" not in first_request_text
    assert len(client.calls[0]["input"]) == 2
    assert np.all(predictions[:25] == 1)
    assert np.all(confidence[:25] == 1.0)
    assert np.all(predictions[25:35] == 0)
    assert np.all(confidence[25:35] == 0.2)
    assert np.all(predictions[35:50] == 1)
    assert np.all(confidence[35:50] == 1.0)
    assert np.all(predictions[50:60] == 2)
    assert np.all(predictions[60:75] == 1)
    assert np.all(predictions[75:] == 1)


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
                "segments": [
                    _segment(0, 20, "Wake", confidence=0.95, reason="wake first"),
                    _segment(40, 60, "REM", confidence=0.9, reason="rem later"),
                ],
            },
            {
                "segments": [
                    _segment(20, 30, "Wake", confidence=0.88, reason="local wake strip"),
                ],
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
        use_overview_pass=True,
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
            "segments": [
                _segment(0, 20, "Wake", confidence=0.95, reason="wake first"),
                _segment(10, 30, "REM", confidence=0.85, reason="overlapping rem"),
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

    snapshot_dir = Path("test-artifacts") / "chatgpt_inference_invalid"

    predictions, confidence = chatgpt_inference.infer(
        mat,
        client=client,
        snapshot_dir=snapshot_dir,
        use_overview_pass=True,
    )

    assert np.isnan(predictions).all()
    assert np.isnan(confidence).all()


def test_chatgpt_preview_writes_dry_run_artifacts_without_modifying_mat_file(
    mock_mat_data,
    monkeypatch,
):
    """The preview pipeline should write model artifacts but leave the source .mat untouched."""
    from scipy.io import loadmat, savemat

    from app_src import chatgpt_inference, chatgpt_preview

    test_root = Path("test-artifacts") / f"chatgpt_preview_{uuid.uuid4().hex}"
    test_root.mkdir(parents=True, exist_ok=True)
    mat_path = test_root / "preview_demo.mat"
    savemat(mat_path, mock_mat_data)
    output_dir = test_root / "preview_output"
    client = MockResponsesClient(
        [
            {
                "segments": [
                    _segment(
                        0,
                        5,
                        "Wake",
                        confidence=0.9,
                        reason="clear wake discontinuity",
                    )
                ],
            },
            {
                "segments": [
                    _segment(
                        50,
                        55,
                        "REM",
                        confidence=0.8,
                        reason="brief REM-like spectrogram break",
                    )
                ],
            },
        ]
    )
    fake_model_figure = FakeFigure()

    monkeypatch.setattr(
        chatgpt_inference,
        "make_chatgpt_vision_figure",
        lambda *args, **kwargs: fake_model_figure,
    )
    monkeypatch.setattr(
        chatgpt_inference,
        "capture_zoom_snapshot",
        lambda _fig, _start_s, _end_s, output_path: _fake_write_snapshot(output_path),
    )
    monkeypatch.setattr(
        chatgpt_preview,
        "capture_zoom_snapshot",
        lambda _fig, _start_s, _end_s, output_path: _fake_write_prediction_snapshot(output_path),
    )

    result = chatgpt_preview.run_chatgpt_preview(
        mat_path=mat_path,
        output_dir=output_dir,
        client=client,
        fixed_refinement_section_count=2,
    )

    model_output = json.loads(result["model_output_json_path"].read_text(encoding="utf-8"))
    reloaded_mat = loadmat(mat_path, squeeze_me=True)

    assert result["output_dir"] == output_dir.resolve()
    assert result["model_output_json_path"] == output_dir.resolve() / "model_output.json"
    assert result["model_output_json_path"].exists()
    assert result["thoughts_path"].exists()
    assert len(result["input_image_paths"]) == 2
    assert all(path.exists() for path in result["input_image_paths"])
    assert len(result["prediction_image_paths"]) == 2
    assert all(path.read_bytes() == b"prediction-png" for path in result["prediction_image_paths"])
    assert len(model_output["input_images"]) == 2
    assert len(model_output["prediction_images"]) == 2
    assert len(model_output["model_calls"]) == 2
    assert model_output["model_calls"][0]["payload"]["segments"][0]["state"] == "Wake"
    assert model_output["model_calls"][1]["payload"]["segments"][0]["state"] == "REM"
    assert "visualization_path" not in model_output
    assert (
        model_output["model_calls"][0]["prediction_image_path"]
        == model_output["prediction_images"][0]["path"]
    )
    assert np.array_equal(reloaded_mat["sleep_scores"], mock_mat_data["sleep_scores"])
