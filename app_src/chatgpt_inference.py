# -*- coding: utf-8 -*-
"""
ChatGPT sleep-scoring backend.

This module implements the ChatGPT scoring flow:

1. Render deterministic model-facing snapshots.
2. Send the guidance prompt plus snapshot images to the Responses API.
3. Parse structured Wake/REM segments from the model output.
4. Write back model-returned segments.
5. Optionally skip the overview pass and score fixed zoomed sections only.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
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
from app_src.config import (
    CHATGPT_FIXED_REFINEMENT_SECTION_COUNT,
    CHATGPT_MODEL,
    CHATGPT_REASONING_EFFORT,
    CHATGPT_REFINEMENT_MODE,
    CHATGPT_SHOW_THOUGHTS,
    CHATGPT_USE_OVERVIEW_PASS,
    CHATGPT_USE_REFERENCE_EXAMPLES,
)
from app_src.make_figure_chatgpt import make_chatgpt_vision_figure
from app_src.make_figure_dev import get_padded_sleep_scores, make_figure

DEFAULT_CHATGPT_MODEL = CHATGPT_MODEL
DEFAULT_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "sleep_scoring_app_data" / "chatgpt_snapshots"
DEFAULT_GUIDANCE_PROMPT_PATH = Path(__file__).with_name("chatgpt_scoring_guidance.md")
DEFAULT_REFERENCE_EXAMPLES_DIR = (
    Path(__file__).resolve().parent / "assets" / "chatgpt_reference_examples"
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.0
DEFAULT_REASONING_EFFORT = CHATGPT_REASONING_EFFORT
DEFAULT_SHOW_THOUGHTS = CHATGPT_SHOW_THOUGHTS
DEFAULT_REFINEMENT_MODE = CHATGPT_REFINEMENT_MODE
DEFAULT_FIXED_REFINEMENT_SECTION_COUNT = CHATGPT_FIXED_REFINEMENT_SECTION_COUNT
DEFAULT_USE_OVERVIEW_PASS = CHATGPT_USE_OVERVIEW_PASS
DEFAULT_USE_REFERENCE_EXAMPLES = CHATGPT_USE_REFERENCE_EXAMPLES
DEFAULT_VISION_FIGURE_MODE = "focused"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_REFINEMENT_MARGIN_S = 15
DEFAULT_MAX_REFINEMENT_INTERVALS = 6
DEFAULT_NREM_CONFIDENCE = 1.0
MODEL_PRICING_USD_PER_1M_TOKENS = {
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.50,
    },
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.02,
        "output": 1.25,
    },
}
REFERENCE_EXAMPLE_TEXT_FILENAME = "groundtruth_reasons_model_friendly.txt"
REFERENCE_EXAMPLE_IMAGE_SPECS = [
    (
        "35_app13_groundtruth_overview_0s_10300s.png",
        "overview",
        "35_app13 reference overview",
    ),
    (
        "35_app13_groundtruth_refinement_1_0s_2575s.png",
        "0-2575 s",
        "35_app13 reference zoom 0-2575 s",
    ),
    (
        "35_app13_groundtruth_refinement_2_2575s_5150s.png",
        "2575-5150 s",
        "35_app13 reference zoom 2575-5150 s",
    ),
    (
        "35_app13_groundtruth_refinement_3_5150s_7725s.png",
        "5150-7725 s",
        "35_app13 reference zoom 5150-7725 s",
    ),
    (
        "35_app13_groundtruth_refinement_4_7725s_10300s.png",
        "7725-10300 s",
        "35_app13 reference zoom 7725-10300 s",
    ),
]

RESPONSE_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "sleep_scoring_segments",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_s": {"type": "integer"},
                        "end_s": {"type": "integer"},
                        "state": {
                            "type": "string",
                            "enum": ["REM", "Wake"],
                        },
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["start_s", "end_s", "state", "reason", "confidence"],
                },
            },
        },
        "required": ["segments"],
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


def _normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    """Return the requested Responses API reasoning effort."""
    raw_effort = DEFAULT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    normalized_effort = str(raw_effort).strip().lower()

    if normalized_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError(
            "reasoning_effort must be one of 'none', 'minimal', 'low', "
            "'medium', 'high', or 'xhigh'."
        )

    return normalized_effort


def _normalize_use_reference_examples(use_reference_examples: bool | None) -> bool:
    """Return whether the reference example pack should be attached to requests."""
    if use_reference_examples is None:
        return bool(DEFAULT_USE_REFERENCE_EXAMPLES)

    return bool(use_reference_examples)


def _normalize_use_overview_pass(use_overview_pass: bool | None) -> bool:
    """Return whether inference should begin with the full-recording overview pass."""
    if use_overview_pass is None:
        return bool(DEFAULT_USE_OVERVIEW_PASS)

    return bool(use_overview_pass)


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


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Return a shallow dict view for SDK objects or plain mappings."""
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key, None))
    }


def _extract_response_usage(response: Any) -> dict[str, Any] | None:
    """Return normalized usage fields for one Responses API call when available."""
    raw_usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if raw_usage is None:
        return None

    usage = _coerce_mapping(raw_usage)
    input_details = _coerce_mapping(usage.get("input_tokens_details"))
    output_details = _coerce_mapping(usage.get("output_tokens_details"))

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_input_tokens = int(input_details.get("cached_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_pricing_model_key(model_name: str) -> str | None:
    """Map a model name or snapshot name to a pricing table key."""
    normalized_model_name = str(model_name).strip().lower()
    for model_key in sorted(MODEL_PRICING_USD_PER_1M_TOKENS, key=len, reverse=True):
        if normalized_model_name == model_key or normalized_model_name.startswith(f"{model_key}-"):
            return model_key

    return None


def _estimate_response_cost_usd(model_name: str, usage: dict[str, Any] | None) -> float | None:
    """Estimate the cost of one model call when usage and pricing are available."""
    if not usage:
        return None

    pricing_key = _normalize_pricing_model_key(model_name)
    if pricing_key is None:
        return None

    pricing = MODEL_PRICING_USD_PER_1M_TOKENS[pricing_key]
    uncached_input_cost = usage["uncached_input_tokens"] * pricing["input"] / 1_000_000
    cached_input_cost = usage["cached_input_tokens"] * pricing["cached_input"] / 1_000_000
    output_cost = usage["output_tokens"] * pricing["output"] / 1_000_000
    return float(uncached_input_cost + cached_input_cost + output_cost)


def _format_cost_usd(cost_usd: float | None) -> str:
    """Format a USD cost estimate for trace output."""
    if cost_usd is None:
        return "unavailable"

    return f"${cost_usd:.6f}"


def _trace_interval_bounds(item: dict[str, Any]) -> tuple[Any, Any]:
    """Return whichever interval keys are available for trace formatting."""
    start_value = item.get("start_idx", item.get("start_s"))
    end_value = item.get("end_idx", item.get("end_s"))
    return start_value, end_value


def _format_trace_segments(segments: list[dict[str, Any]]) -> list[str]:
    """Format scored model segments for compact trace output."""
    if not segments:
        return ["(none)"]

    lines = []
    for segment in segments:
        start_value, end_value = _trace_interval_bounds(segment)
        reason = str(segment.get("reason", "")).strip() or "no reason provided"
        lines.append(
            "- "
            f"{start_value}s-{end_value}s | "
            f"{segment['state']} | "
            f"{_format_confidence(segment.get('confidence'))} | "
            f"{reason}"
        )
    return lines


def _format_trace_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Convert a structured model payload into compact visible-reasoning blocks."""
    return {
        "segments": _format_trace_segments(payload.get("segments", [])),
    }


def _segments_to_absolute_seconds(
    segments: list[dict[str, Any]],
    recording_start_s: float,
) -> list[dict[str, Any]]:
    """Return normalized segments with absolute-second bounds for JSON artifacts."""
    absolute_segments = []
    for segment in segments:
        absolute_segments.append(
            {
                "start_s": int(round(recording_start_s + segment["start_idx"])),
                "end_s": int(round(recording_start_s + segment["end_idx"])),
                "state": segment["state"],
                "reason": segment["reason"],
                "confidence": segment["confidence"],
            }
        )

    return absolute_segments


def _record_model_call_artifact(
    artifact_log: list[dict[str, Any]] | None,
    *,
    label: str,
    kind: str,
    snapshot_path: str | Path,
    start_s: float,
    end_s: float,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Append a JSON-serializable record for one model-facing request."""
    if artifact_log is None:
        return None

    record = {
        "label": label,
        "kind": kind,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "image_path": str(Path(snapshot_path).resolve()),
        "payload": payload,
        "normalized_segments": [],
    }
    artifact_log.append(record)
    return record


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


def _split_reference_examples_text(reference_text: str) -> tuple[str, dict[str, str]]:
    """Split the model-friendly reference text into conventions and per-window notes."""
    section_headers = {label for _filename, label, _description in REFERENCE_EXAMPLE_IMAGE_SPECS}
    section_headers.discard("overview")
    heading_pattern = re.compile(r"^\d+-\d+ s$")

    conventions_lines: list[str] = []
    section_lines: dict[str, list[str]] = {}
    current_header: str | None = None

    for raw_line in reference_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped in section_headers or heading_pattern.match(stripped):
            current_header = stripped
            section_lines.setdefault(current_header, [])
            continue

        if current_header is None:
            conventions_lines.append(line)
            continue

        section_lines[current_header].append(line)

    return (
        "\n".join(line for line in conventions_lines if line.strip()).strip(),
        {
            header: "\n".join(line for line in lines if line.strip()).strip()
            for header, lines in section_lines.items()
        },
    )


def _build_reference_examples_message(
    reference_examples_dir: str | Path = DEFAULT_REFERENCE_EXAMPLES_DIR,
) -> dict[str, Any] | None:
    """Return a single user message containing the ground-truth reference pack."""
    reference_examples_dir = Path(reference_examples_dir)
    reference_text_path = reference_examples_dir / REFERENCE_EXAMPLE_TEXT_FILENAME
    if not reference_text_path.exists():
        return None

    reference_text = reference_text_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        return None

    conventions_text, section_text = _split_reference_examples_text(reference_text)
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Reference example pack:\n"
                "Study these labeled ground-truth examples before scoring the target recording. "
                "They use the same model-facing figure style. Treat them as calibration examples "
                "for identifying obvious non-NREM bouts, not as a template to copy.\n\n"
                f"{conventions_text}"
            ).strip(),
        }
    ]

    for filename, label, description in REFERENCE_EXAMPLE_IMAGE_SPECS:
        image_path = reference_examples_dir / filename
        if not image_path.exists():
            continue

        if label == "overview":
            text = (
                f"{description}.\n"
                "Use this image to calibrate rough bout placement and overall cadence. "
                "Exact second-level boundaries are less important here than correct detection "
                "of obvious non-NREM intervals."
            )
        else:
            section_notes = section_text.get(label, "")
            text = f"{description}.\nMatching labeled bouts and reasons:\n{section_notes}".strip()

        content.extend(
            [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": _image_path_to_data_url(image_path)},
            ]
        )

    if len(content) == 1:
        return None

    return {
        "role": "user",
        "content": content,
    }


def _build_coarse_request_input(
    guidance_prompt: str,
    image_data_url: str,
    recording_start_s: float,
    recording_end_s: float,
    reference_examples_message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic prompt for the full-session overview pass."""
    metadata_prompt = (
        "sleep score the figure based on the provided guidance to the best of your judgment.\n"
        "If there truly are some parts that you cannot resolve, tell me the reasons "
        "that prevent you from making the call."
    )

    request_input = [
        {
            "role": "system",
            "content": guidance_prompt,
        },
    ]
    if reference_examples_message is not None:
        request_input.append(reference_examples_message)

    request_input.append(
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
        }
    )

    return request_input


def _build_refinement_request_input(
    guidance_prompt: str,
    image_data_url: str,
    interval_start_s: float,
    interval_end_s: float,
    refinement_reason: str,
) -> list[dict[str, Any]]:
    """Build a bounded prompt for a local refinement pass."""
    metadata_prompt = (
        "sleep score the figure based on the provided guidance to the best of your judgment.\n"
        "If there truly are some parts that you cannot resolve, tell me the reasons "
        "that prevent you from making the call."
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


def _build_zoom_section_request_input(
    guidance_prompt: str,
    image_data_url: str,
    interval_start_s: float,
    interval_end_s: float,
    section_reason: str,
) -> list[dict[str, Any]]:
    """Build a prompt for scoring one zoomed section without an overview pass."""
    metadata_prompt = (
        "sleep score the figure based on the provided guidance to the best of your judgment.\n"
        "If there truly are some parts that you cannot resolve, tell me the reasons "
        "that prevent you from making the call."
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


def _normalize_segments(
    payload: dict[str, Any],
    recording_start_s: float,
    duration_s: int,
) -> list[dict[str, Any]]:
    """Validate and normalize structured Wake/REM segment output."""
    raw_segments = payload.get("segments", [])

    if not isinstance(raw_segments, list):
        raise ValueError("Structured output must contain list-valued segments.")

    normalized_segments = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise ValueError("Each segment must be an object.")

        state = raw_segment.get("state")
        if state not in {"Wake", "REM"}:
            raise ValueError(f"Unsupported sleep state from ChatGPT: {state!r}.")

        confidence = float(raw_segment["confidence"])
        if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
            raise ValueError("Segment confidence must be a finite value in [0, 1].")

        reason = str(raw_segment.get("reason", "")).strip()
        if not reason:
            raise ValueError("Each segment must include a non-empty reason.")

        start_idx, end_idx = _normalize_interval_indices(
            raw_segment["start_s"],
            raw_segment["end_s"],
            recording_start_s=recording_start_s,
            duration_s=duration_s,
        )
        normalized_segments.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "state": state,
                "reason": reason,
                "confidence": confidence,
            }
        )

    normalized_segments.sort(key=lambda segment: (segment["start_idx"], segment["end_idx"]))
    previous_end_idx = 0
    for segment in normalized_segments:
        if segment["start_idx"] < previous_end_idx:
            raise ValueError("ChatGPT returned overlapping scored segments.")
        previous_end_idx = segment["end_idx"]

    return normalized_segments


def _overlay_structured_scoring(
    current_predictions: np.ndarray,
    current_confidence: np.ndarray,
    segments: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay model-returned Wake/REM segments onto the current scores."""
    predictions = current_predictions.copy()
    confidence = current_confidence.copy()

    for segment in segments:
        if segment["confidence"] < confidence_threshold:
            continue

        predictions[segment["start_idx"] : segment["end_idx"]] = SLEEP_STAGE_TO_SCORE[
            segment["state"]
        ]
        confidence[segment["start_idx"] : segment["end_idx"]] = segment["confidence"]

    return predictions, confidence


def _fill_interval_with_stage(
    predictions: np.ndarray,
    confidence: np.ndarray,
    start_idx: int,
    end_idx: int,
    state: str,
    confidence_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copies with one interval filled as a baseline stage."""
    filled_predictions = predictions.copy()
    filled_confidence = confidence.copy()
    filled_predictions[start_idx:end_idx] = SLEEP_STAGE_TO_SCORE[state]
    filled_confidence[start_idx:end_idx] = confidence_value
    return filled_predictions, filled_confidence


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
    segments: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    """Reject refinement output that tries to rewrite outside the requested window."""
    for item in segments:
        if item["start_idx"] < candidate["start_idx"] or item["end_idx"] > candidate["end_idx"]:
            raise ValueError("Refinement output must stay inside the requested interval.")


def _request_structured_scoring(
    client: Any,
    model_name: str,
    request_input: list[dict[str, Any]],
    reasoning_effort: str,
    trace_logger: _TraceLogger | None = None,
    trace_label: str = "ChatGPT Request",
) -> dict[str, Any]:
    """Send one structured scoring request and return the decoded payload."""
    response = client.responses.create(
        model=model_name,
        input=request_input,
        reasoning={"effort": reasoning_effort},
        text={"format": RESPONSE_TEXT_FORMAT},
    )
    payload = _extract_response_payload(response)
    usage = _extract_response_usage(response)
    cost_estimate_usd = _estimate_response_cost_usd(model_name, usage)

    if trace_logger is not None:
        usage_lines = [f"- reasoning_effort: {reasoning_effort}"]
        if usage is None:
            usage_lines.extend(
                [
                    "- input_tokens: unavailable",
                    "- output_tokens: unavailable",
                    "- reasoning_tokens: unavailable",
                    "- total_tokens: unavailable",
                    f"- estimated_cost_usd: {_format_cost_usd(cost_estimate_usd)}",
                ]
            )
        else:
            usage_lines.extend(
                [
                    f"- input_tokens: {usage['input_tokens']}",
                    f"- cached_input_tokens: {usage['cached_input_tokens']}",
                    f"- uncached_input_tokens: {usage['uncached_input_tokens']}",
                    f"- output_tokens: {usage['output_tokens']}",
                    f"- reasoning_tokens: {usage['reasoning_tokens']}",
                    f"- total_tokens: {usage['total_tokens']}",
                    f"- estimated_cost_usd: {_format_cost_usd(cost_estimate_usd)}",
                ]
            )
        trace_logger.add_text_block(f"{trace_label} API Usage", usage_lines)
        formatted_payload = _format_trace_payload(payload)
        trace_logger.add_text_block(
            f"{trace_label} Proposed Segments",
            formatted_payload["segments"],
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
    reasoning_effort: str,
    zoom_section_only: bool = False,
    trace_logger: _TraceLogger | None = None,
    artifact_log: list[dict[str, Any]] | None = None,
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
    trace_prefix = "Zoom Section" if zoom_section_only else "Refinement"

    if trace_logger is not None:
        trace_logger.add_text_block(
            f"{trace_prefix} Mode",
            [f"- {refinement_mode}"],
        )
        trace_logger.add_text_block(
            f"{trace_prefix} Windows",
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
        if zoom_section_only:
            predictions, confidence = _fill_interval_with_stage(
                predictions=predictions,
                confidence=confidence,
                start_idx=candidate["start_idx"],
                end_idx=candidate["end_idx"],
                state="NREM",
                confidence_value=DEFAULT_NREM_CONFIDENCE,
            )
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
                    f"{trace_prefix} {candidate_index} Window",
                    _format_refinement_window(candidate, recording_start_s),
                )

            request_input = (
                _build_zoom_section_request_input(
                    guidance_prompt=guidance_prompt,
                    image_data_url=image_data_url,
                    interval_start_s=interval_start_s,
                    interval_end_s=interval_end_s,
                    section_reason=candidate["reason"],
                )
                if zoom_section_only
                else _build_refinement_request_input(
                    guidance_prompt=guidance_prompt,
                    image_data_url=image_data_url,
                    interval_start_s=interval_start_s,
                    interval_end_s=interval_end_s,
                    refinement_reason=candidate["reason"],
                )
            )
            payload = _request_structured_scoring(
                client=client,
                model_name=model_name,
                request_input=request_input,
                reasoning_effort=reasoning_effort,
                trace_logger=trace_logger,
                trace_label=f"{trace_prefix} {candidate_index}",
            )
            artifact_record = _record_model_call_artifact(
                artifact_log,
                label=f"{trace_prefix} {candidate_index}",
                kind="zoom_section" if zoom_section_only else "refinement",
                snapshot_path=snapshot_path,
                start_s=interval_start_s,
                end_s=interval_end_s,
                payload=payload,
            )
            refined_segments = _normalize_segments(
                payload,
                recording_start_s=recording_start_s,
                duration_s=duration_s,
            )
            _validate_intervals_within_candidate(
                refined_segments,
                candidate,
            )
            if artifact_record is not None:
                artifact_record["normalized_segments"] = _segments_to_absolute_seconds(
                    refined_segments,
                    recording_start_s=recording_start_s,
                )
        except Exception as exc:
            if trace_logger is not None:
                trace_logger.add(
                    f"{trace_prefix} {candidate_index} Error",
                    {
                        "candidate": candidate,
                        "error": repr(exc),
                    },
                )
            continue

        predictions, confidence = _overlay_structured_scoring(
            current_predictions=predictions,
            current_confidence=confidence,
            segments=refined_segments,
            confidence_threshold=confidence_threshold,
        )
        if trace_logger is not None:
            trace_logger.add_text_block(
                f"{trace_prefix} {candidate_index} Applied Segments",
                _format_trace_segments(refined_segments),
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
    reasoning_effort: str | None = None,
    use_overview_pass: bool | None = None,
    use_reference_examples: bool | None = None,
    reference_examples_dir: str | Path = DEFAULT_REFERENCE_EXAMPLES_DIR,
    guidance_prompt_path: str | Path = DEFAULT_GUIDANCE_PROMPT_PATH,
    artifact_log: list[dict[str, Any]] | None = None,
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
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    normalized_use_overview_pass = _normalize_use_overview_pass(use_overview_pass)
    normalized_use_reference_examples = _normalize_use_reference_examples(use_reference_examples)
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
            f"- reasoning_effort: {normalized_reasoning_effort}",
            f"- overview_pass: {normalized_use_overview_pass}",
            f"- reference_examples: {normalized_use_reference_examples}",
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
        reference_examples_message = None
        if normalized_use_overview_pass and normalized_use_reference_examples:
            reference_examples_message = _build_reference_examples_message(reference_examples_dir)
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
        if normalized_use_overview_pass:
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
                    reference_examples_message=reference_examples_message,
                ),
                reasoning_effort=normalized_reasoning_effort,
                trace_logger=trace_logger,
                trace_label="Coarse Pass",
            )
            artifact_record = _record_model_call_artifact(
                artifact_log,
                label="Coarse Pass",
                kind="overview",
                snapshot_path=snapshot_path,
                start_s=recording_start_s,
                end_s=recording_end_s,
                payload=payload,
            )

            coarse_segments = _normalize_segments(
                payload,
                recording_start_s=recording_start_s,
                duration_s=duration_s,
            )
            if artifact_record is not None:
                artifact_record["normalized_segments"] = _segments_to_absolute_seconds(
                    coarse_segments,
                    recording_start_s=recording_start_s,
                )
            coarse_uncertain_intervals = []
            predictions, confidence = _overlay_structured_scoring(
                current_predictions=base_scores,
                current_confidence=fallback_confidence,
                segments=coarse_segments,
                confidence_threshold=threshold,
            )
            trace_logger.add_text_block(
                "Coarse Pass Applied Segments",
                _format_trace_segments(coarse_segments),
            )
        else:
            coarse_segments = []
            coarse_uncertain_intervals = []
            predictions = base_scores.copy()
            confidence = fallback_confidence.copy()
            trace_logger.add_text_block(
                "Coarse Pass",
                ["- skipped; scoring fixed zoomed sections only."],
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
            coarse_bouts=coarse_segments,
            coarse_uncertain_intervals=coarse_uncertain_intervals,
            confidence_threshold=threshold,
            refinement_mode=normalized_refinement_mode,
            fixed_refinement_section_count=normalized_fixed_refinement_section_count,
            reasoning_effort=normalized_reasoning_effort,
            zoom_section_only=not normalized_use_overview_pass,
            trace_logger=trace_logger,
            artifact_log=artifact_log,
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


def infer_with_artifacts(
    mat: dict[str, Any],
    model_name: str = DEFAULT_CHATGPT_MODEL,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    client: Any = None,
    confidence_threshold: float | None = None,
    show_thoughts: bool | None = None,
    refinement_mode: str | None = None,
    fixed_refinement_section_count: int | None = None,
    vision_figure_mode: str | None = None,
    reasoning_effort: str | None = None,
    use_overview_pass: bool | None = None,
    use_reference_examples: bool | None = None,
    reference_examples_dir: str | Path = DEFAULT_REFERENCE_EXAMPLES_DIR,
    guidance_prompt_path: str | Path = DEFAULT_GUIDANCE_PROMPT_PATH,
) -> dict[str, Any]:
    """Run ChatGPT inference and return predictions plus dry-run artifacts."""
    artifact_log: list[dict[str, Any]] = []
    snapshot_dir = Path(snapshot_dir)
    recording_label = _get_recording_label(mat)
    trace_path = _build_trace_path(snapshot_dir, recording_label)

    predictions, confidence = infer(
        mat=mat,
        model_name=model_name,
        snapshot_dir=snapshot_dir,
        client=client,
        confidence_threshold=confidence_threshold,
        show_thoughts=show_thoughts,
        refinement_mode=refinement_mode,
        fixed_refinement_section_count=fixed_refinement_section_count,
        vision_figure_mode=vision_figure_mode,
        reasoning_effort=reasoning_effort,
        use_overview_pass=use_overview_pass,
        use_reference_examples=use_reference_examples,
        reference_examples_dir=reference_examples_dir,
        guidance_prompt_path=guidance_prompt_path,
        artifact_log=artifact_log,
    )

    return {
        "predictions": predictions,
        "confidence": confidence,
        "model_calls": artifact_log,
        "thoughts_path": trace_path if trace_path.exists() else None,
    }
