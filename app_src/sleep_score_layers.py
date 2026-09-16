"""Keep manual sleep labels separate from model-generated sleep scores."""

from __future__ import annotations

import numpy as np

WAKE = 0
NREM = 1
ACTIVE_WAKE = 4
QUIET_WAKE = 5
WAKE_STAGES = (WAKE, ACTIVE_WAKE, QUIET_WAKE)


def coarse_sleep_scores(values) -> np.ndarray:
    """Collapse Wake subtypes for sleep scoring without modifying the source."""
    scores = np.asarray(values, dtype=float).reshape(-1).copy()
    scores[np.isin(scores, WAKE_STAGES)] = WAKE
    return scores


def saved_user_sleep_scores(mat, length: int) -> np.ndarray:
    """New files preserve sparse annotations; legacy files treat scores as manual."""
    source = mat.get("user_sleep_scores")
    if source is None or np.asarray(source).size == 0:
        source = mat.get("sleep_scores")
    scores = normalize_sleep_scores(source, length)
    scores[scores == -1] = np.nan
    return scores


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
