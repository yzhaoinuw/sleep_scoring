# -*- coding: utf-8 -*-
"""
ChatGPT sleep-scoring backend.

This module implements a two-stage scoring flow:

1. Render a deterministic overview snapshot for a coarse first pass.
2. Send the guidance prompt plus the overview image to the Responses API.
3. Parse structured contiguous sleep-state bouts from the model output.
4. Write back only bouts above a configurable confidence threshold.
5. Run targeted local refinement for uncertain, low-confidence, or
   transition-heavy intervals using zoom snapshots only.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app_src.chatgpt_tools import (
    SLEEP_STAGE_TO_SCORE,
    capture_overview_snapshot,
    capture_zoom_snapshot,
)
from app_src.config import CHATGPT_MODEL, CHATGPT_SHOW_THOUGHTS
from app_src.config import CHATGPT_FIXED_REFINEMENT_SECTION_COUNT, CHATGPT_REFINEMENT_MODE
from app_src.make_figure_chatgpt import make_chatgpt_vision_figure
from app_src.make_figure_dev import get_padded_sleep_scores, make_figure

DEFAULT_CHATGPT_MODEL = CHATGPT_MODEL
DEFAULT_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "sleep_scoring_app_data" / "chatgpt_snapshots"
DEFAULT_GUIDANCE_PROMPT_PATH = Path(__file__).with_name("chatgpt_scoring_guidance.md")
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SHOW_THOUGHTS = CHATGPT_SHOW_THOUGHTS
DEFAULT_REFINEMENT_MODE = CHATGPT_REFINEMENT_MODE
DEFAULT_FIXED_REFINEMENT_SECTION_COUNT = CHATGPT_FIXED_REFINEMENT_SECTION_COUNT
DEFAULT_VISION_FIGURE_MODE = "focused"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_REFINEMENT_MARGIN_S = 15
DEFAULT_MAX_REFINEMENT_INTERVALS = 6

RESPONSE_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "sleep_scoring_bouts",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "bouts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_s": {"type": "integer"},
                        "end_s": {"type": "integer"},
                        "state": {
                            "type": "string",
                            "enum": sorted(SLEEP_STAGE_TO_SCORE),
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["start_s", "end_s", "state", "confidence"],
                },
            },
            "uncertain_intervals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_s": {"type": "integer"},
                        "end_s": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["start_s", "end_s", "reason"],
                },
            },
        },
        "required": ["summary", "bouts", "uncertain_intervals"],
    },
}


def _scalar_value(value: Any, default: float = 0.0) -> float:
    """Return a float for MATLAB-loaded scalar values or a default."""
    if value is None:
        return float(default)

    array = np.asarray(value)
    if array.size == 0:
        return float(default)

    return float(array.reshape(-1)[0])


def _get_recording_window(mat: dict[str, Any]) -> tuple[float, int, float]:
    """Return the absolute recording start, integer duration, and end time."""
    start_s = _scalar_value(mat.get("start_time"), default=0.0)
    duration_s = int(get_padded_sleep_scores(mat).size)
    end_s = start_s + duration_s
    return start_s, duration_s, end_s


def _get_recording_label(mat: dict[str, Any]) -> str:
    """Return a human-readable recording label for exported figure titles."""
    for key in (
        "_source_filename",
        "filename",
        "file_name",
        "mat_name",
        "recording_name",
    ):
        value = mat.get(key)
        if value is None:
            continue

        label = str(np.asarray(value).reshape(-1)[0]).strip()
        if label:
            return Path(label).stem

    return "recording"


def _format_title_second(value: float) -> str:
    """Format a second value compactly for figure titles."""
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))

    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _build_snapshot_title(recording_label: str, start_s: float, end_s: float) -> str:
    """Build a concise figure title with the recording label and interval."""
    return (
        f"{recording_label} | " f"{_format_title_second(start_s)}s-{_format_title_second(end_s)}s"
    )


def _set_figure_title(figure: Any, title_text: str) -> None:
    """Best-effort figure title update for exported snapshots."""
    update_layout = getattr(figure, "update_layout", None)
    if callable(update_layout):
        update_layout(title=dict(text=title_text))


def _normalize_confidence_threshold(confidence_threshold: float | None) -> float:
    """Return a bounded confidence threshold in [0, 1]."""
    if confidence_threshold is None:
        raw_threshold = os.getenv(
            "SLEEP_SCORING_CHATGPT_CONFIDENCE_THRESHOLD",
            str(DEFAULT_CONFIDENCE_THRESHOLD),
        )
    else:
        raw_threshold = confidence_threshold

    threshold = float(raw_threshold)
    if not np.isfinite(threshold):
        raise ValueError("confidence_threshold must be finite.")

    return float(min(1.0, max(0.0, threshold)))


def _load_guidance_prompt(prompt_path: str | Path = DEFAULT_GUIDANCE_PROMPT_PATH) -> str:
    """Load the current sleep-scoring guidance prompt."""
    return Path(prompt_path).read_text(encoding="utf-8").strip()


def _normalize_show_thoughts(show_thoughts: bool | None) -> bool:
    """Return the explicit trace-logging choice for this inference run."""
    if show_thoughts is None:
        return bool(DEFAULT_SHOW_THOUGHTS)

    return bool(show_thoughts)


def _normalize_refinement_mode(refinement_mode: str | None) -> str:
    """Return the requested refinement mode for this inference run."""
    raw_mode = DEFAULT_REFINEMENT_MODE if refinement_mode is None else refinement_mode
    normalized_mode = str(raw_mode).strip().lower()
    aliases = {
        "disabled": "none",
        "off": "none",
        "overview_only": "none",
        "fixed": "fixed_sections",
    }
    normalized_mode = aliases.get(normalized_mode, normalized_mode)

    if normalized_mode not in {"none", "adaptive", "fixed_sections"}:
        raise ValueError("refinement_mode must be one of 'none', 'adaptive', or 'fixed_sections'.")

    return normalized_mode


def _normalize_fixed_refinement_section_count(section_count: int | None) -> int:
    """Return the number of fixed broad refinement sections to use."""
    raw_count = DEFAULT_FIXED_REFINEMENT_SECTION_COUNT if section_count is None else section_count
    normalized_count = int(raw_count)
    if normalized_count < 1:
        raise ValueError("fixed_refinement_section_count must be at least 1.")

    return normalized_count


def _normalize_vision_figure_mode(vision_figure_mode: str | None) -> str:
    """Return the requested model-facing figure layout mode."""
    raw_mode = DEFAULT_VISION_FIGURE_MODE if vision_figure_mode is None else vision_figure_mode
    normalized_mode = str(raw_mode).strip().lower()
    aliases = {
        "chatgpt": "focused",
        "compact": "focused",
        "spectrogram_ne": "focused",
        "focused_2panel": "focused",
        "ui": "full",
        "full_4panel": "full",
        "overview_full": "full",
    }
    normalized_mode = aliases.get(normalized_mode, normalized_mode)

    if normalized_mode not in {"focused", "full"}:
        raise ValueError("vision_figure_mode must be one of 'focused' or 'full'.")

    return normalized_mode


def _build_model_figure(
    mat: dict[str, Any],
    plot_name: str,
    vision_figure_mode: str,
) -> Any:
    """Build the model-facing figure for the requested layout mode."""
    if vision_figure_mode == "full":
        return make_figure(mat, plot_name=plot_name)

    return make_chatgpt_vision_figure(mat, plot_name=plot_name)


def _sanitize_trace_value(value: Any) -> Any:
    """Redact large binary payloads so trace logs stay readable."""
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, _, encoded = value.partition(";base64,")
            return f"{prefix};base64,<omitted {len(encoded)} chars>"
        return value

    if isinstance(value, list):
        return [_sanitize_trace_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _sanitize_trace_value(item) for key, item in value.items()}

    return value


class _TraceLogger:
    """Write an opt-in plain-text debug trace for one ChatGPT inference run."""

    def __init__(self, enabled: bool, path: Path | None):
        self.enabled = enabled
        self.path = path if enabled else None
        self._lines: list[str] = []

    def add(self, title: str, value: Any | None = None) -> None:
        """Append one trace section when logging is enabled."""
        if not self.enabled:
            return

        self._lines.append(f"## {title}")
        if value is not None:
            self._lines.append(self._format_value(value))
        self._lines.append("")

    def save(self) -> None:
        """Flush the accumulated trace to disk."""
        if not self.enabled or self.path is None:
            return

        self.path.write_text("\n".join(self._lines).rstrip() + "\n", encoding="utf-8")

    def add_text_block(self, title: str, lines: list[str]) -> None:
        """Append a preformatted plain-text block."""
        if not self.enabled:
            return

        self._lines.append(f"## {title}")
        self._lines.extend(lines or ["(none)"])
        self._lines.append("")

    def _format_value(self, value: Any) -> str:
        sanitized = _sanitize_trace_value(value)
        if isinstance(sanitized, str):
            return sanitized

        return json.dumps(sanitized, indent=2, sort_keys=True, default=str)


def _format_confidence(confidence: Any) -> str:
    """Format a confidence value for trace output."""
    if confidence is None:
        return "n/a"

    numeric = float(confidence)
    return f"{numeric:.2f}"


def _trace_interval_bounds(item: dict[str, Any]) -> tuple[Any, Any]:
    """Return whichever interval keys are available for trace formatting."""
    start_value = item.get("start_idx", item.get("start_s"))
    end_value = item.get("end_idx", item.get("end_s"))
    return start_value, end_value


def _format_trace_bouts(bouts: list[dict[str, Any]]) -> list[str]:
    """Format scored bouts for compact trace output."""
    if not bouts:
        return ["(none)"]

    lines = []
    for bout in bouts:
        start_value, end_value = _trace_interval_bounds(bout)
        lines.append(
            "- "
            f"{start_value}s-{end_value}s | "
            f"{bout['state']} | confidence {_format_confidence(bout.get('confidence'))}"
        )
    return lines


def _format_trace_uncertain_intervals(intervals: list[dict[str, Any]]) -> list[str]:
    """Format uncertain intervals for compact trace output."""
    if not intervals:
        return ["(none)"]

    lines = []
    for interval in intervals:
        start_value, end_value = _trace_interval_bounds(interval)
        reason = interval.get("reason") or "uncertain"
        lines.append(f"- {start_value}s-{end_value}s | {reason}")
    return lines


def _format_trace_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Convert a structured model payload into compact visible-reasoning blocks."""
    bouts = payload.get("bouts", [])
    uncertain_intervals = payload.get("uncertain_intervals", [])
    summary = str(payload.get("summary", "")).strip() or "(no summary)"
    return {
        "summary": [summary],
        "bouts": _format_trace_bouts(bouts),
        "uncertain_intervals": _format_trace_uncertain_intervals(uncertain_intervals),
    }


def _format_refinement_window(
    candidate: dict[str, Any],
    recording_start_s: float,
) -> list[str]:
    """Format the requested refinement window as a small trace block."""
    return [
        "- "
        f"{recording_start_s + candidate['start_idx']:.0f}s-"
        f"{recording_start_s + candidate['end_idx']:.0f}s | "
        f"{candidate.get('reason') or 'refinement'}"
    ]


def _sanitize_filename_part(value: str) -> str:
    """Convert a label into a filesystem-safe filename fragment."""
    sanitized = []
    for char in str(value).strip():
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")

    result = "".join(sanitized).strip("._")
    return result or "recording"


def _build_snapshot_basename(recording_label: str, start_s: float, end_s: float) -> str:
    """Return a readable basename for overview/zoom snapshot files."""
    safe_label = _sanitize_filename_part(recording_label)
    return f"{safe_label}_{_format_title_second(start_s)}s_{_format_title_second(end_s)}s"


def _build_snapshot_path(
    snapshot_dir: Path,
    recording_label: str,
    start_s: float,
    end_s: float,
    suffix: str = ".png",
) -> Path:
    """Return a deterministic snapshot path for the given recording interval."""
    basename = _build_snapshot_basename(recording_label, start_s, end_s)
    return snapshot_dir / f"{basename}{suffix}"


def _build_trace_path(snapshot_dir: Path, recording_label: str) -> Path:
    """Return a deterministic trace path for one recording."""
    safe_label = _sanitize_filename_part(recording_label)
    return snapshot_dir / f"{safe_label}_thoughts.txt"


def _build_openai_client(client: Any = None) -> Any:
    """Return the provided client or create an OpenAI client when available."""
    if client is not None:
        return client

    api_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    return OpenAI(api_key=api_key)


def get_backend_ready_status() -> tuple[bool, str]:
    """Return whether the ChatGPT backend can make live API requests."""
    api_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        return (
            False,
            f"Set `{OPENAI_API_KEY_ENV_VAR}` in a local `.env` file or environment variable "
            "to enable ChatGPT scoring.",
        )

    try:
        import openai  # noqa: F401
    except ImportError:
        return False, "Install the `openai` package to enable ChatGPT scoring."

    return True, "ChatGPT scoring is ready."


def _image_path_to_data_url(image_path: str | Path) -> str:
    """Encode an image file as a data URL for the Responses API."""
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    encoded_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


def _build_coarse_request_input(
    guidance_prompt: str,
    image_data_url: str,
    recording_start_s: float,
    recording_end_s: float,
) -> list[dict[str, Any]]:
    """Build a deterministic prompt for the full-session overview pass."""
    metadata_prompt = (
        "Score this full recording from the overview plot.\n"
        f"Recording start time: {recording_start_s:.3f} seconds.\n"
        f"Recording end time: {recording_end_s:.3f} seconds.\n"
        "Use absolute seconds that match the x-axis in the image.\n"
        "Return only contiguous non-overlapping high-confidence bouts in `bouts`.\n"
        "Put ambiguous regions in `uncertain_intervals` instead of forcing a label.\n"
        "Do not invent extra sleep states.\n"
        "Keep the summary concise."
    )

    return [
        {
            "role": "system",
            "content": guidance_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": metadata_prompt,
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        },
    ]


def _build_refinement_request_input(
    guidance_prompt: str,
    image_data_url: str,
    interval_start_s: float,
    interval_end_s: float,
    refinement_reason: str,
) -> list[dict[str, Any]]:
    """Build a bounded prompt for a local refinement pass."""
    metadata_prompt = (
        "Refine only this local interval from the zoomed plot.\n"
        f"Target interval start time: {interval_start_s:.3f} seconds.\n"
        f"Target interval end time: {interval_end_s:.3f} seconds.\n"
        f"Refinement reason: {refinement_reason}\n"
        "Base the decision only on the image for this interval.\n"
        "Only return bouts that stay entirely inside the target interval.\n"
        "If any sub-interval remains ambiguous, include it in `uncertain_intervals`.\n"
        "Do not rely on prior labels or non-image helper summaries."
    )

    return [
        {
            "role": "system",
            "content": guidance_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": metadata_prompt,
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        },
    ]


def _extract_response_payload(response: Any) -> dict[str, Any]:
    """Extract a JSON payload from a Responses API result or test double."""
    payload = None
    if isinstance(response, dict):
        payload = response.get("output_parsed", response.get("output_text"))
    else:
        payload = getattr(response, "output_parsed", None)
        if payload is None:
            payload = getattr(response, "output_text", None)

    if payload is None:
        raise ValueError("ChatGPT response did not include any structured output.")

    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise ValueError("ChatGPT response payload must decode to a JSON object.")

    return payload


def _normalize_interval_indices(
    start_s: Any,
    end_s: Any,
    recording_start_s: float,
    duration_s: int,
) -> tuple[int, int]:
    """Convert absolute-second bounds into clamped array indices."""
    start_idx = max(0, int(math.floor(float(start_s) - recording_start_s)))
    end_idx = min(duration_s, int(math.ceil(float(end_s) - recording_start_s)))

    if end_idx <= start_idx:
        raise ValueError("Interval must span at least one second within the recording.")

    return start_idx, end_idx


def _normalize_bouts(
    payload: dict[str, Any],
    recording_start_s: float,
    duration_s: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and normalize structured bout output."""
    raw_bouts = payload.get("bouts", [])
    raw_uncertain_intervals = payload.get("uncertain_intervals", [])

    if not isinstance(raw_bouts, list) or not isinstance(raw_uncertain_intervals, list):
        raise ValueError(
            "Structured output must contain list-valued bouts and uncertain_intervals."
        )

    normalized_bouts = []
    for raw_bout in raw_bouts:
        if not isinstance(raw_bout, dict):
            raise ValueError("Each bout must be an object.")

        state = raw_bout.get("state")
        if state not in SLEEP_STAGE_TO_SCORE:
            raise ValueError(f"Unsupported sleep state from ChatGPT: {state!r}.")

        confidence = float(raw_bout["confidence"])
        if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
            raise ValueError("Bout confidence must be a finite value in [0, 1].")

        start_idx, end_idx = _normalize_interval_indices(
            raw_bout["start_s"],
            raw_bout["end_s"],
            recording_start_s=recording_start_s,
            duration_s=duration_s,
        )
        normalized_bouts.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "state": state,
                "confidence": confidence,
            }
        )

    normalized_bouts.sort(key=lambda bout: (bout["start_idx"], bout["end_idx"]))
    previous_end_idx = 0
    for bout in normalized_bouts:
        if bout["start_idx"] < previous_end_idx:
            raise ValueError("ChatGPT returned overlapping scored bouts.")
        previous_end_idx = bout["end_idx"]

    normalized_uncertain_intervals = []
    for raw_interval in raw_uncertain_intervals:
        if not isinstance(raw_interval, dict):
            raise ValueError("Each uncertain interval must be an object.")

        start_idx, end_idx = _normalize_interval_indices(
            raw_interval["start_s"],
            raw_interval["end_s"],
            recording_start_s=recording_start_s,
            duration_s=duration_s,
        )
        normalized_uncertain_intervals.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "reason": str(raw_interval.get("reason", "")).strip(),
            }
        )

    return normalized_bouts, normalized_uncertain_intervals


def _overlay_structured_scoring(
    current_predictions: np.ndarray,
    current_confidence: np.ndarray,
    baseline_scores: np.ndarray,
    bouts: list[dict[str, Any]],
    uncertain_intervals: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay confident bouts while restoring uncertain regions to the baseline."""
    predictions = current_predictions.copy()
    confidence = current_confidence.copy()

    for bout in bouts:
        if bout["confidence"] < confidence_threshold:
            continue

        predictions[bout["start_idx"] : bout["end_idx"]] = SLEEP_STAGE_TO_SCORE[bout["state"]]
        confidence[bout["start_idx"] : bout["end_idx"]] = bout["confidence"]

    for interval in uncertain_intervals:
        predictions[interval["start_idx"] : interval["end_idx"]] = baseline_scores[
            interval["start_idx"] : interval["end_idx"]
        ]
        confidence[interval["start_idx"] : interval["end_idx"]] = np.nan

    return predictions, confidence


def _merge_refinement_candidates(
    candidates: list[dict[str, Any]],
    max_intervals: int,
) -> list[dict[str, Any]]:
    """Merge overlapping refinement windows so the second pass stays bounded."""
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (candidate["start_idx"], candidate["end_idx"]),
    )

    merged_candidates = []
    for candidate in sorted_candidates:
        if not merged_candidates or candidate["start_idx"] > merged_candidates[-1]["end_idx"]:
            merged_candidates.append(
                {
                    "start_idx": candidate["start_idx"],
                    "end_idx": candidate["end_idx"],
                    "reasons": [candidate["reason"]],
                }
            )
            continue

        merged_candidates[-1]["end_idx"] = max(
            merged_candidates[-1]["end_idx"],
            candidate["end_idx"],
        )
        merged_candidates[-1]["reasons"].append(candidate["reason"])

    normalized_candidates = []
    for candidate in merged_candidates[:max_intervals]:
        unique_reasons = list(dict.fromkeys(reason for reason in candidate["reasons"] if reason))
        normalized_candidates.append(
            {
                "start_idx": candidate["start_idx"],
                "end_idx": candidate["end_idx"],
                "reason": "; ".join(unique_reasons) or "local ambiguity",
            }
        )

    return normalized_candidates


def _build_refinement_candidates(
    bouts: list[dict[str, Any]],
    uncertain_intervals: list[dict[str, Any]],
    duration_s: int,
    confidence_threshold: float,
    transition_margin_s: int = DEFAULT_REFINEMENT_MARGIN_S,
    max_intervals: int = DEFAULT_MAX_REFINEMENT_INTERVALS,
) -> list[dict[str, Any]]:
    """Select bounded local intervals for zoomed follow-up refinement."""
    candidates = []

    for interval in uncertain_intervals:
        candidates.append(
            {
                "start_idx": interval["start_idx"],
                "end_idx": interval["end_idx"],
                "reason": interval["reason"] or "coarse-pass uncertainty",
            }
        )

    for bout in bouts:
        if bout["confidence"] >= confidence_threshold:
            continue

        candidates.append(
            {
                "start_idx": bout["start_idx"],
                "end_idx": bout["end_idx"],
                "reason": (
                    f"low-confidence {bout['state']} bout from coarse pass "
                    f"({bout['confidence']:.2f})"
                ),
            }
        )

    for previous_bout, next_bout in zip(bouts, bouts[1:]):
        if previous_bout["state"] == next_bout["state"]:
            continue

        start_idx = max(0, previous_bout["end_idx"] - transition_margin_s)
        end_idx = min(duration_s, next_bout["start_idx"] + transition_margin_s)
        if end_idx <= start_idx:
            continue

        candidates.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "reason": (
                    "transition-heavy boundary between "
                    f"{previous_bout['state']} and {next_bout['state']}"
                ),
            }
        )

    return _merge_refinement_candidates(candidates, max_intervals=max_intervals)


def _build_fixed_section_refinement_candidates(
    duration_s: int,
    section_count: int,
) -> list[dict[str, Any]]:
    """Split the recording into a small number of broad fixed refinement windows."""
    candidates = []
    for section_index in range(section_count):
        start_idx = (section_index * duration_s) // section_count
        end_idx = ((section_index + 1) * duration_s) // section_count
        if end_idx <= start_idx:
            continue

        candidates.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "reason": f"fixed broad section {section_index + 1} of {section_count}",
            }
        )

    return candidates


def _validate_intervals_within_candidate(
    bouts: list[dict[str, Any]],
    uncertain_intervals: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    """Reject refinement output that tries to rewrite outside the requested window."""
    for item in [*bouts, *uncertain_intervals]:
        if item["start_idx"] < candidate["start_idx"] or item["end_idx"] > candidate["end_idx"]:
            raise ValueError("Refinement output must stay inside the requested interval.")


def _request_structured_scoring(
    client: Any,
    model_name: str,
    request_input: list[dict[str, Any]],
    trace_logger: _TraceLogger | None = None,
    trace_label: str = "ChatGPT Request",
) -> dict[str, Any]:
    """Send one structured scoring request and return the decoded payload."""
    response = client.responses.create(
        model=model_name,
        input=request_input,
        text={"format": RESPONSE_TEXT_FORMAT},
    )
    payload = _extract_response_payload(response)

    if trace_logger is not None:
        formatted_payload = _format_trace_payload(payload)
        trace_logger.add_text_block(f"{trace_label} Summary", formatted_payload["summary"])
        trace_logger.add_text_block(f"{trace_label} Proposed Bouts", formatted_payload["bouts"])
        trace_logger.add_text_block(
            f"{trace_label} Proposed Uncertain Intervals",
            formatted_payload["uncertain_intervals"],
        )

    return payload


def _run_refinement_pass(
    *,
    mat: dict[str, Any],
    figure: Any,
    snapshot_dir: Path,
    client: Any,
    model_name: str,
    guidance_prompt: str,
    recording_label: str,
    recording_start_s: float,
    duration_s: int,
    current_predictions: np.ndarray,
    current_confidence: np.ndarray,
    coarse_bouts: list[dict[str, Any]],
    coarse_uncertain_intervals: list[dict[str, Any]],
    confidence_threshold: float,
    refinement_mode: str,
    fixed_refinement_section_count: int,
    trace_logger: _TraceLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run zoomed local follow-up requests for ambiguous or transition-heavy regions."""
    if refinement_mode == "none":
        refinement_candidates = []
    elif refinement_mode == "fixed_sections":
        refinement_candidates = _build_fixed_section_refinement_candidates(
            duration_s=duration_s,
            section_count=fixed_refinement_section_count,
        )
    else:
        refinement_candidates = _build_refinement_candidates(
            bouts=coarse_bouts,
            uncertain_intervals=coarse_uncertain_intervals,
            duration_s=duration_s,
            confidence_threshold=confidence_threshold,
        )
    predictions = current_predictions.copy()
    confidence = current_confidence.copy()

    if trace_logger is not None:
        trace_logger.add_text_block(
            "Refinement Mode",
            [f"- {refinement_mode}"],
        )
        trace_logger.add_text_block(
            "Refinement Windows",
            [
                line
                for candidate in refinement_candidates
                for line in _format_refinement_window(candidate, recording_start_s)
            ]
            or ["(none)"],
        )

    if not refinement_candidates:
        return predictions, confidence

    for candidate_index, candidate in enumerate(refinement_candidates):
        interval_start_s = recording_start_s + candidate["start_idx"]
        interval_end_s = recording_start_s + candidate["end_idx"]
        baseline_scores = predictions.copy()

        try:
            _set_figure_title(
                figure,
                _build_snapshot_title(recording_label, interval_start_s, interval_end_s),
            )
            snapshot_path = _build_snapshot_path(
                snapshot_dir,
                recording_label,
                interval_start_s,
                interval_end_s,
            )
            snapshot_path = capture_zoom_snapshot(
                figure,
                interval_start_s,
                interval_end_s,
                snapshot_path,
            )
            image_data_url = _image_path_to_data_url(snapshot_path)

            if trace_logger is not None:
                trace_logger.add_text_block(
                    f"Refinement {candidate_index} Window",
                    _format_refinement_window(candidate, recording_start_s),
                )

            payload = _request_structured_scoring(
                client=client,
                model_name=model_name,
                request_input=_build_refinement_request_input(
                    guidance_prompt=guidance_prompt,
                    image_data_url=image_data_url,
                    interval_start_s=interval_start_s,
                    interval_end_s=interval_end_s,
                    refinement_reason=candidate["reason"],
                ),
                trace_logger=trace_logger,
                trace_label=f"Refinement {candidate_index}",
            )
            refined_bouts, refined_uncertain_intervals = _normalize_bouts(
                payload,
                recording_start_s=recording_start_s,
                duration_s=duration_s,
            )
            _validate_intervals_within_candidate(
                refined_bouts,
                refined_uncertain_intervals,
                candidate,
            )
        except Exception as exc:
            if trace_logger is not None:
                trace_logger.add(
                    f"Refinement {candidate_index} Error",
                    {
                        "candidate": candidate,
                        "error": repr(exc),
                    },
                )
            continue

        predictions, confidence = _overlay_structured_scoring(
            current_predictions=predictions,
            current_confidence=confidence,
            baseline_scores=baseline_scores,
            bouts=refined_bouts,
            uncertain_intervals=refined_uncertain_intervals,
            confidence_threshold=confidence_threshold,
        )
        if trace_logger is not None:
            trace_logger.add_text_block(
                f"Refinement {candidate_index} Applied Bouts",
                _format_trace_bouts(refined_bouts),
            )
            trace_logger.add_text_block(
                f"Refinement {candidate_index} Applied Uncertain Intervals",
                _format_trace_uncertain_intervals(refined_uncertain_intervals),
            )

    return predictions, confidence


def infer(
    mat: dict[str, Any],
    model_name: str = DEFAULT_CHATGPT_MODEL,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    client: Any = None,
    confidence_threshold: float | None = None,
    show_thoughts: bool | None = None,
    refinement_mode: str | None = None,
    fixed_refinement_section_count: int | None = None,
    vision_figure_mode: str | None = None,
    guidance_prompt_path: str | Path = DEFAULT_GUIDANCE_PROMPT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ChatGPT-generated sleep scores and confidence values.

    If the OpenAI client, API key, snapshot export, or structured response
    parsing is unavailable, the function falls back to the current in-memory
    scores so the app remains usable while the beta path is still being
    hardened.
    """
    base_scores = get_padded_sleep_scores(mat).astype(float)
    fallback_confidence = np.full(base_scores.shape, np.nan, dtype=float)
    threshold = _normalize_confidence_threshold(confidence_threshold)
    normalized_refinement_mode = _normalize_refinement_mode(refinement_mode)
    normalized_fixed_refinement_section_count = _normalize_fixed_refinement_section_count(
        fixed_refinement_section_count
    )
    normalized_vision_figure_mode = _normalize_vision_figure_mode(vision_figure_mode)
    recording_label = _get_recording_label(mat)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = _TraceLogger(
        enabled=_normalize_show_thoughts(show_thoughts),
        path=_build_trace_path(snapshot_dir, recording_label),
    )
    trace_logger.add_text_block(
        "Trace Info",
        [
            f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"- model: {model_name}",
            f"- confidence_threshold: {threshold:.2f}",
            f"- refinement_mode: {normalized_refinement_mode}",
            f"- vision_figure_mode: {normalized_vision_figure_mode}",
        ],
    )

    client = _build_openai_client(client=client)
    if client is None:
        trace_logger.add_text_block(
            "Fallback",
            ["- OpenAI client unavailable. Returning the current in-memory scores."],
        )
        trace_logger.save()
        return base_scores, fallback_confidence

    try:
        guidance_prompt = _load_guidance_prompt(guidance_prompt_path)
        recording_start_s, duration_s, recording_end_s = _get_recording_window(mat)
        figure = _build_model_figure(
            mat=mat,
            plot_name=_build_snapshot_title(
                recording_label,
                recording_start_s,
                recording_end_s,
            ),
            vision_figure_mode=normalized_vision_figure_mode,
        )
        snapshot_path = _build_snapshot_path(
            snapshot_dir,
            recording_label,
            recording_start_s,
            recording_end_s,
        )
        snapshot_path = capture_overview_snapshot(figure, snapshot_path)
        image_data_url = _image_path_to_data_url(snapshot_path)

        payload = _request_structured_scoring(
            client=client,
            model_name=model_name,
            request_input=_build_coarse_request_input(
                guidance_prompt=guidance_prompt,
                image_data_url=image_data_url,
                recording_start_s=recording_start_s,
                recording_end_s=recording_end_s,
            ),
            trace_logger=trace_logger,
            trace_label="Coarse Pass",
        )

        coarse_bouts, coarse_uncertain_intervals = _normalize_bouts(
            payload,
            recording_start_s=recording_start_s,
            duration_s=duration_s,
        )
        predictions, confidence = _overlay_structured_scoring(
            current_predictions=base_scores,
            current_confidence=fallback_confidence,
            baseline_scores=base_scores,
            bouts=coarse_bouts,
            uncertain_intervals=coarse_uncertain_intervals,
            confidence_threshold=threshold,
        )
        trace_logger.add_text_block(
            "Coarse Pass Applied Bouts",
            _format_trace_bouts(coarse_bouts),
        )
        trace_logger.add_text_block(
            "Coarse Pass Applied Uncertain Intervals",
            _format_trace_uncertain_intervals(coarse_uncertain_intervals),
        )
        predictions, confidence = _run_refinement_pass(
            mat=mat,
            figure=figure,
            snapshot_dir=snapshot_dir,
            client=client,
            model_name=model_name,
            guidance_prompt=guidance_prompt,
            recording_label=recording_label,
            recording_start_s=recording_start_s,
            duration_s=duration_s,
            current_predictions=predictions,
            current_confidence=confidence,
            coarse_bouts=coarse_bouts,
            coarse_uncertain_intervals=coarse_uncertain_intervals,
            confidence_threshold=threshold,
            refinement_mode=normalized_refinement_mode,
            fixed_refinement_section_count=normalized_fixed_refinement_section_count,
            trace_logger=trace_logger,
        )
        trace_logger.add_text_block(
            "Final Writeback",
            [
                f"- predictions_written: {int(np.count_nonzero(~np.isnan(predictions)))}",
                f"- confidence_written: {int(np.count_nonzero(~np.isnan(confidence)))}",
            ],
        )
        return predictions, confidence
    except Exception as exc:
        trace_logger.add_text_block(
            "Run Error",
            [
                f"- {repr(exc)}",
                "- Returning the current in-memory scores.",
            ],
        )
        return base_scores, fallback_confidence
    finally:
        try:
            trace_logger.save()
        except Exception:
            pass
