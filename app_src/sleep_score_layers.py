"""Keep manual sleep labels separate from model-generated sleep scores."""

from __future__ import annotations

import numpy as np


def normalize_sleep_scores(values, length: int) -> np.ndarray:
    """Return a one-second score layer of exactly ``length`` values.

    ``None`` values from Dash stores become ``nan`` here.  The function is
    deliberately stage-agnostic: MA labels remain valid manual overrides even
    though the statistical model itself only predicts Wake, NREM, and REM.
    """
    normalized = np.full(length, np.nan, dtype=float)
    if values is None:
        return normalized

    source = np.asarray(values, dtype=float).reshape(-1)
    count = min(length, source.size)
    normalized[:count] = source[:count]
    return normalized


def overlay_user_sleep_scores(
    model_sleep_scores: np.ndarray,
    user_sleep_scores: np.ndarray | list[float] | None,
) -> np.ndarray:
    """Apply finite manual labels to a model score array without mutating it."""
    scores = np.asarray(model_sleep_scores, dtype=float).reshape(-1).copy()
    user_scores = normalize_sleep_scores(user_sleep_scores, scores.size)
    manual_mask = np.isfinite(user_scores)
    scores[manual_mask] = user_scores[manual_mask]
    return scores
