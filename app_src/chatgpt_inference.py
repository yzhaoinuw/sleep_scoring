# -*- coding: utf-8 -*-
"""
ChatGPT sleep-scoring backend.

This module implements a two-stage scoring flow:

1. Render a deterministic overview snapshot for a coarse first pass.
2. Send the guidance prompt plus the overview image to the Responses API.
3. Parse structured contiguous sleep-state bouts from the model output.
4. Write back only bouts above a configurable confidence threshold.
5. Run targeted local refinement for uncertain, low-confidence, or
   transition-heavy intervals using zoom snapshots plus numeric helpers.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app_src.chatgpt_tools import (
    SLEEP_STAGE_TO_SCORE,
    capture_overview_snapshot,
    capture_zoom_snapshot,
    get_current_scores,
    get_interval_features,
)
from app_src.config import CHATGPT_MODEL
from app_src.make_figure_dev import get_padded_sleep_scores, make_figure

DEFAULT_CHATGPT_MODEL = CHATGPT_MODEL
DEFAULT_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "sleep_scoring_app_data" / "chatgpt_snapshots"
DEFAULT_GUIDANCE_PROMPT_PATH = Path(__file__).with_name("chatgpt_scoring_guidance.md")
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
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
    interval_features: dict[str, Any],
    current_scores: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a bounded prompt for a local refinement pass."""
    metadata_prompt = (
        "Refine only this local interval from the zoomed plot.\n"
        f"Target interval start time: {interval_start_s:.3f} seconds.\n"
        f"Target interval end time: {interval_end_s:.3f} seconds.\n"
        f"Refinement reason: {refinement_reason}\n"
        "Use the image and helper outputs together.\n"
        "Only return bouts that stay entirely inside the target interval.\n"
        "If any sub-interval remains ambiguous, include it in `uncertain_intervals`.\n"
        "The current in-memory interval labels are provided in `current_scores`.\n"
        f"interval_features={json.dumps(interval_features, sort_keys=True)}\n"
        f"current_scores={json.dumps(current_scores, sort_keys=True)}"
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


def _shift_current_scores_to_absolute(
    current_scores: dict[str, Any],
    recording_start_s: float,
) -> dict[str, Any]:
    """Offset per-second score summaries into absolute recording seconds."""
    absolute_scores = []
    for item in current_scores["scores"]:
        absolute_scores.append(
            {
                **item,
                "second": recording_start_s + item["second"],
            }
        )

    absolute_blocks = []
    for block in current_scores["score_blocks"]:
        absolute_blocks.append(
            {
                **block,
                "start_s": recording_start_s + block["start_s"],
                "end_s": recording_start_s + block["end_s"],
            }
        )

    return {
        **current_scores,
        "start_s": recording_start_s + current_scores["start_s"],
        "end_s": recording_start_s + current_scores["end_s"],
        "scores": absolute_scores,
        "score_blocks": absolute_blocks,
    }


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
) -> dict[str, Any]:
    """Send one structured scoring request and return the decoded payload."""
    response = client.responses.create(
        model=model_name,
        input=request_input,
        text={"format": RESPONSE_TEXT_FORMAT},
    )
    return _extract_response_payload(response)


def _run_refinement_pass(
    *,
    mat: dict[str, Any],
    figure: Any,
    snapshot_dir: Path,
    client: Any,
    model_name: str,
    guidance_prompt: str,
    recording_start_s: float,
    duration_s: int,
    current_predictions: np.ndarray,
    current_confidence: np.ndarray,
    coarse_bouts: list[dict[str, Any]],
    coarse_uncertain_intervals: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run zoomed local follow-up requests for ambiguous or transition-heavy regions."""
    refinement_candidates = _build_refinement_candidates(
        bouts=coarse_bouts,
        uncertain_intervals=coarse_uncertain_intervals,
        duration_s=duration_s,
        confidence_threshold=confidence_threshold,
    )

    predictions = current_predictions.copy()
    confidence = current_confidence.copy()

    for candidate_index, candidate in enumerate(refinement_candidates):
        interval_start_s = recording_start_s + candidate["start_idx"]
        interval_end_s = recording_start_s + candidate["end_idx"]
        baseline_scores = predictions.copy()

        try:
            snapshot_path = snapshot_dir / f"zoom_{candidate_index}_{uuid.uuid4().hex}.png"
            snapshot_path = capture_zoom_snapshot(
                figure,
                interval_start_s,
                interval_end_s,
                snapshot_path,
            )
            image_data_url = _image_path_to_data_url(snapshot_path)

            interval_features = get_interval_features(
                mat,
                start_s=interval_start_s,
                end_s=interval_end_s,
                fig=figure,
            )
            current_scores = get_current_scores(
                predictions,
                start_s=candidate["start_idx"],
                end_s=candidate["end_idx"],
            )
            current_scores = _shift_current_scores_to_absolute(
                current_scores,
                recording_start_s=recording_start_s,
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
                    interval_features=interval_features,
                    current_scores=current_scores,
                ),
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
        except Exception:
            continue

        predictions, confidence = _overlay_structured_scoring(
            current_predictions=predictions,
            current_confidence=confidence,
            baseline_scores=baseline_scores,
            bouts=refined_bouts,
            uncertain_intervals=refined_uncertain_intervals,
            confidence_threshold=confidence_threshold,
        )

    return predictions, confidence


def infer(
    mat: dict[str, Any],
    model_name: str = DEFAULT_CHATGPT_MODEL,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    client: Any = None,
    confidence_threshold: float | None = None,
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

    client = _build_openai_client(client=client)
    if client is None:
        return base_scores, fallback_confidence

    try:
        guidance_prompt = _load_guidance_prompt(guidance_prompt_path)
        recording_start_s, duration_s, recording_end_s = _get_recording_window(mat)

        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"overview_{uuid.uuid4().hex}.png"

        figure = make_figure(mat, plot_name="ChatGPT Sleep Scoring")
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
        predictions, confidence = _run_refinement_pass(
            mat=mat,
            figure=figure,
            snapshot_dir=snapshot_dir,
            client=client,
            model_name=model_name,
            guidance_prompt=guidance_prompt,
            recording_start_s=recording_start_s,
            duration_s=duration_s,
            current_predictions=predictions,
            current_confidence=confidence,
            coarse_bouts=coarse_bouts,
            coarse_uncertain_intervals=coarse_uncertain_intervals,
            confidence_threshold=threshold,
        )
        return predictions, confidence
    except Exception:
        return base_scores, fallback_confidence
