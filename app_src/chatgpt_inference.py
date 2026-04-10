# -*- coding: utf-8 -*-
"""
Placeholder ChatGPT sleep-scoring backend.

This module intentionally keeps the first implementation lightweight:

1. Preserve the app's current sleep scores as a no-op fallback.
2. Define the future orchestration point for overview snapshots, zoom requests,
   feature requests, and score writeback.
3. Keep imports dependency-free until the OpenAI client wiring is added.
"""

import tempfile
from pathlib import Path

import numpy as np

from app_src.make_figure_dev import get_padded_sleep_scores

DEFAULT_CHATGPT_MODEL = "gpt-4.1-mini"
DEFAULT_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "sleep_scoring_app_data" / "chatgpt_snapshots"
DEFAULT_GUIDANCE_PROMPT_PATH = Path(__file__).with_name("chatgpt_scoring_guidance.md")


def infer(
    mat,
    model_name=DEFAULT_CHATGPT_MODEL,
    snapshot_dir=DEFAULT_SNAPSHOT_DIR,
    client=None,
):
    """
    Return placeholder predictions and confidence for the ChatGPT backend.

    Future implementation notes:
    - Save a deterministic overview snapshot for the current recording.
    - Load the draft guidance prompt from `DEFAULT_GUIDANCE_PROMPT_PATH`.
    - Send the image plus the sleep-scoring guidance prompt to the OpenAI Responses API.
    - Parse coarse segments from the model response.
    - Request targeted zoom snapshots or numeric interval features for uncertain
      or transition-heavy regions.
    - Convert contiguous blocks back into per-second sleep scores.
    - Prefer model-side global transition reasoning over local post-hoc rewrites.

    Parameters
    ----------
    mat : dict
        Loaded MATLAB data structure containing EEG/EMG/(optional) NE arrays.
    model_name : str
        OpenAI model name to use once API wiring is implemented.
    snapshot_dir : pathlib.Path
        Temp directory where overview and zoom snapshots should be written.
    client : Any
        Reserved for a future OpenAI client instance.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    del model_name, snapshot_dir, client

    # Preserve any existing labels for now so the UI keeps working while the
    # API and helper functions are being filled in.
    predictions = get_padded_sleep_scores(mat).astype(float)
    confidence = np.full(predictions.shape, np.nan, dtype=float)

    return predictions, confidence
