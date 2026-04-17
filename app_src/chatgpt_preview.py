# -*- coding: utf-8 -*-
"""Dry-run ChatGPT scoring pipeline for a single MATLAB recording."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy.io import loadmat

import app_src.chatgpt_inference as chatgpt_inference
from app_src.chatgpt_tools import capture_overview_snapshot, capture_zoom_snapshot

DEFAULT_MODEL_OUTPUT_FILENAME = "model_output.json"
DEFAULT_PREDICTION_IMAGE_PREFIX = "prediction_"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json_default(value: Any) -> Any:
    """Convert common path/numpy values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


def _input_images_from_model_calls(model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact manifest of the images sent to the model."""
    return [
        {
            "label": call["label"],
            "kind": call["kind"],
            "start_s": call["start_s"],
            "end_s": call["end_s"],
            "path": call["image_path"],
        }
        for call in model_calls
    ]


def _prediction_snapshot_path(output_dir: Path, model_call: dict[str, Any]) -> Path:
    """Return the prediction-overlaid model-facing PNG path for one model call."""
    input_image_name = Path(model_call["image_path"]).name
    return output_dir / f"{DEFAULT_PREDICTION_IMAGE_PREFIX}{input_image_name}"


def _build_prediction_title(recording_label: str, model_call: dict[str, Any]) -> str:
    """Build a compact title for a model-facing prediction snapshot."""
    start_s = chatgpt_inference._format_title_second(model_call["start_s"])
    end_s = chatgpt_inference._format_title_second(model_call["end_s"])
    return f"{recording_label} | ChatGPT prediction | {start_s}s-{end_s}s"


def _write_prediction_model_snapshots(
    *,
    mat: dict[str, Any],
    model_calls: list[dict[str, Any]],
    output_dir: Path,
    recording_label: str,
    vision_figure_mode: str | None,
) -> list[dict[str, Any]]:
    """Write prediction-overlaid model-facing snapshots matching each input image."""
    if not model_calls:
        return []

    normalized_vision_figure_mode = chatgpt_inference._normalize_vision_figure_mode(
        vision_figure_mode
    )
    figure = chatgpt_inference._build_model_figure(
        mat=mat,
        plot_name=f"{recording_label} | ChatGPT prediction",
        vision_figure_mode=normalized_vision_figure_mode,
    )

    prediction_images = []
    for model_call in model_calls:
        output_path = _prediction_snapshot_path(output_dir, model_call)
        chatgpt_inference._set_figure_title(
            figure,
            _build_prediction_title(recording_label, model_call),
        )
        if model_call["kind"] == "overview":
            written_path = capture_overview_snapshot(figure, output_path)
        else:
            written_path = capture_zoom_snapshot(
                figure,
                model_call["start_s"],
                model_call["end_s"],
                output_path,
            )

        model_call["prediction_image_path"] = str(written_path.resolve())
        prediction_images.append(
            {
                "label": model_call["label"],
                "kind": model_call["kind"],
                "start_s": model_call["start_s"],
                "end_s": model_call["end_s"],
                "path": str(written_path.resolve()),
                "input_image_path": model_call["image_path"],
            }
        )

    return prediction_images


def run_chatgpt_preview(
    mat_path: str | Path,
    output_dir: str | Path,
    *,
    client: Any = None,
    model_name: str = chatgpt_inference.DEFAULT_CHATGPT_MODEL,
    confidence_threshold: float | None = None,
    show_thoughts: bool = True,
    refinement_mode: str | None = None,
    fixed_refinement_section_count: int | None = None,
    vision_figure_mode: str | None = None,
    reasoning_effort: str | None = None,
    use_overview_pass: bool | None = None,
    use_reference_examples: bool | None = None,
    reference_examples_dir: str | Path = chatgpt_inference.DEFAULT_REFERENCE_EXAMPLES_DIR,
    guidance_prompt_path: str | Path = chatgpt_inference.DEFAULT_GUIDANCE_PROMPT_PATH,
) -> dict[str, Any]:
    """
    Run ChatGPT scoring on a .mat file without opening the app or saving scores to it.

    The output directory receives:
    - the PNG images sent to the model,
    - ``model_output.json`` with one entry per model call,
    - a thoughts trace when ``show_thoughts`` is enabled,
    - prediction-overlaid model-facing PNGs matching each input image window.
    """
    mat_path = Path(mat_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file does not exist: {mat_path}")

    if client is None:
        ready, message = chatgpt_inference.get_backend_ready_status()
        if not ready:
            raise RuntimeError(message)

    mat = loadmat(mat_path, squeeze_me=True)
    mat["_source_filename"] = mat_path.stem

    inference_result = chatgpt_inference.infer_with_artifacts(
        mat=mat,
        model_name=model_name,
        snapshot_dir=output_dir,
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
    )

    predictions = inference_result["predictions"]
    confidence = inference_result["confidence"]
    model_calls = inference_result["model_calls"]
    thoughts_path = inference_result["thoughts_path"]

    prediction_mat = dict(mat)
    prediction_mat["sleep_scores"] = predictions
    prediction_mat["confidence"] = confidence
    prediction_images = _write_prediction_model_snapshots(
        mat=prediction_mat,
        model_calls=model_calls,
        output_dir=output_dir,
        recording_label=mat_path.stem,
        vision_figure_mode=vision_figure_mode,
    )

    model_output_path = output_dir / DEFAULT_MODEL_OUTPUT_FILENAME
    output_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mat_path": str(mat_path),
        "output_dir": str(output_dir),
        "model": model_name,
        "reasoning_effort": reasoning_effort or chatgpt_inference.DEFAULT_REASONING_EFFORT,
        "thoughts_path": str(thoughts_path) if thoughts_path is not None else None,
        "input_images": _input_images_from_model_calls(model_calls),
        "prediction_images": prediction_images,
        "model_calls": model_calls,
    }
    model_output_path.write_text(
        json.dumps(output_payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "input_image_paths": [Path(item["path"]) for item in output_payload["input_images"]],
        "prediction_image_paths": [
            Path(item["path"]) for item in output_payload["prediction_images"]
        ],
        "model_output_json_path": model_output_path,
        "thoughts_path": thoughts_path,
        "predictions": predictions,
        "confidence": confidence,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dry ChatGPT sleep-scoring preview for one .mat file."
    )
    parser.add_argument("mat_path", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Directory where preview artifacts will be written.")
    parser.add_argument("--model", default=chatgpt_inference.DEFAULT_CHATGPT_MODEL)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--refinement-mode", default=None)
    parser.add_argument("--fixed-section-count", type=int, default=None)
    parser.add_argument("--vision-figure-mode", default=None)
    parser.add_argument("--use-overview-pass", action="store_true")
    parser.add_argument("--use-reference-examples", action="store_true")
    parser.add_argument("--no-thoughts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = run_chatgpt_preview(
        mat_path=args.mat_path,
        output_dir=args.output_dir,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        show_thoughts=not args.no_thoughts,
        refinement_mode=args.refinement_mode,
        fixed_refinement_section_count=args.fixed_section_count,
        vision_figure_mode=args.vision_figure_mode,
        reasoning_effort=args.reasoning_effort,
        use_overview_pass=True if args.use_overview_pass else None,
        use_reference_examples=True if args.use_reference_examples else None,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "model_output_json_path": str(result["model_output_json_path"]),
                "thoughts_path": (
                    str(result["thoughts_path"]) if result["thoughts_path"] is not None else None
                ),
                "input_image_paths": [str(path) for path in result["input_image_paths"]],
                "prediction_image_paths": [str(path) for path in result["prediction_image_paths"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())

    # Spyder-friendly direct-run settings. Edit these values, then press Run.
    mat_path = PROJECT_ROOT / "user_test_files" / "830.mat"
    output_dir = PROJECT_ROOT / "chatgpt_preview_outputs" / mat_path.stem
    model_name = chatgpt_inference.DEFAULT_CHATGPT_MODEL
    reasoning_effort = None
    confidence_threshold = None
    show_thoughts = True
    refinement_mode = None
    fixed_refinement_section_count = None
    vision_figure_mode = None
    use_overview_pass = None
    use_reference_examples = None

    result = run_chatgpt_preview(
        mat_path=mat_path,
        output_dir=output_dir,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        show_thoughts=show_thoughts,
        refinement_mode=refinement_mode,
        fixed_refinement_section_count=fixed_refinement_section_count,
        vision_figure_mode=vision_figure_mode,
        reasoning_effort=reasoning_effort,
        use_overview_pass=use_overview_pass,
        use_reference_examples=use_reference_examples,
    )
    print("ChatGPT preview complete.")
    print(f"Output folder: {result['output_dir']}")
    print(f"Model output JSON: {result['model_output_json_path']}")
    print(f"Thoughts file: {result['thoughts_path']}")
    print("Prediction PNGs:")
    for prediction_image_path in result["prediction_image_paths"]:
        print(f"- {prediction_image_path}")
