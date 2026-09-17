"""Optional, EMG-only subdivision of an already scored Wake mask.

No Dash or stats-model dependencies. All times here are recording-relative.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, detrend, sosfiltfilt

from app_src.sleep_score_layers import (
    ACTIVE_WAKE,
    NREM,
    QUIET_WAKE,
    WAKE,
    coarse_sleep_scores,
    normalize_sleep_scores,
    overlay_user_sleep_scores,
)


@dataclass(frozen=True)
class WakeActivityConfig:
    method: str = "wake_rank_per_second"
    active_fraction: float = 0.8
    threshold: float | None = None
    min_duration: float = 1.0
    gap_tolerance: float = 0.5
    rms_window: float = 0.5
    envelope_rate: int = 20
    nrem_baseline_percentile: float = 75.0
    nrem_deviation_multiplier: float = 2.0

    def __post_init__(self):
        if self.method not in {"wake_rank_per_second", "nrem_envelope"}:
            raise ValueError("Wake activity method must be wake_rank_per_second or nrem_envelope.")
        if not np.isfinite(self.active_fraction) or not 0 < self.active_fraction < 1:
            raise ValueError("Active Wake fraction must be between zero and one.")
        if self.threshold is not None and (not np.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("EMG threshold must be a finite non-negative number or Auto.")
        if not np.isfinite(self.min_duration) or self.min_duration <= 0:
            raise ValueError("Minimum active duration must be greater than zero.")
        if not np.isfinite(self.gap_tolerance) or self.gap_tolerance < 0:
            raise ValueError("EMG gap tolerance must be non-negative.")
        if not np.isfinite(self.rms_window) or self.rms_window <= 0:
            raise ValueError("EMG smoothing window must be greater than zero.")
        if self.envelope_rate != 20:
            raise ValueError("The pilot uses a 20 Hz EMG envelope.")
        if (
            not np.isfinite(self.nrem_baseline_percentile)
            or not 0 <= self.nrem_baseline_percentile <= 100
        ):
            raise ValueError("NREM baseline percentile must be between 0 and 100.")
        if not np.isfinite(self.nrem_deviation_multiplier) or self.nrem_deviation_multiplier < 0:
            raise ValueError("NREM deviation multiplier must be non-negative.")


@dataclass
class WakeActivityResult:
    sleep_scores: np.ndarray
    envelope: np.ndarray
    threshold: float
    calibrated_seconds: int
    active_bouts: list[tuple[float, float]]
    metadata: dict


def runs(mask):
    """Half-open runs of True values, including recording edges."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def emg_envelope(emg, frequency, config=None):
    """Return the centered moving-RMS EMG envelope at 20 Hz.

    A value centered at ``t`` is ``sqrt(mean(filtered_emg**2))`` over the
    approximately 0.5-second window from ``t - 0.25`` to ``t + 0.25``. Squaring
    prevents the positive and negative raw waveform from cancelling, and the square
    root preserves EMG units. The zero-phase filter and centered window avoid a
    directional delay, but smooth an activity boundary by roughly 0.25 seconds on
    either side.

    Envelope centers are 50 ms apart, so adjacent 0.5-second RMS windows overlap by
    about 0.45 seconds. This is a deliberately smooth intermediate measurement for
    the 0.5-second gap and one-second occupancy rules, not a 50 ms behavior label or
    a biologically validated temporal resolution. Filter finite stretches
    independently. Flatlines of at least one second and RMS windows touching invalid
    samples remain invalid, not Quiet Wake. The original EMG is not changed.
    """
    config = config or WakeActivityConfig()
    fs = float(frequency)
    if not np.isfinite(fs) or fs <= 50:
        raise ValueError("Raw EMG detection requires EEG/EMG sampling above 50 Hz.")
    raw = np.asarray(emg, dtype=float).reshape(-1)
    if raw.size < math.ceil(fs):
        raise ValueError("At least one second of raw EMG is required.")
    valid = np.isfinite(raw)
    for start, end in runs(np.r_[False, np.diff(raw) == 0]):
        if end - start + 1 >= fs:
            valid[max(0, start - 1) : end] = False
    width = max(1, round(config.rms_window * fs))
    sos = butter(4, [20, min(200, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    power = np.full(raw.size, np.nan)
    for start, end in runs(valid):
        if end - start <= max(width, 30):
            continue
        filtered = sosfiltfilt(sos, detrend(raw[start:end], type="linear"))
        rms = np.sqrt(np.maximum(uniform_filter1d(filtered**2, width, mode="nearest"), 0))
        # Do not bridge a dropout through the RMS window.
        left = width // 2 if start else 0
        right = width // 2 if end < raw.size else 0
        stop = end - right
        power[start + left : stop] = rms[left : len(rms) - right if right else None]
    count = math.ceil(raw.size / fs * config.envelope_rate)
    centres = (np.arange(count) + 0.5) / config.envelope_rate
    sample_indices = np.minimum((centres * fs).astype(int), raw.size - 1)
    return power[sample_indices]


def emg_second_rms(emg, frequency, length):
    """Return one filtered-EMG RMS value for each one-second score epoch.

    This is the default feature for the per-recording Active/Quiet rank split. Each
    value is ``sqrt(mean(filtered_emg**2))`` over that score's own one-second raw-EMG
    interval. It is not the ordinary EMG mean, whose positive and negative waveform
    values cancel, and it does not reuse the 20 Hz envelope. A final, partial score
    epoch uses its available EMG samples, matching the app's EEG-derived recording
    duration. Invalid samples and flatlines of at least one second leave their
    affected score seconds invalid.
    """
    fs = float(frequency)
    if not np.isfinite(fs) or fs <= 50:
        raise ValueError("Raw EMG detection requires EEG/EMG sampling above 50 Hz.")
    raw = np.asarray(emg, dtype=float).reshape(-1)
    if raw.size < math.ceil(fs):
        raise ValueError("At least one second of raw EMG is required.")
    valid = np.isfinite(raw)
    for start, end in runs(np.r_[False, np.diff(raw) == 0]):
        if end - start + 1 >= fs:
            valid[max(0, start - 1) : end] = False
    sos = butter(4, [20, min(200, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    filtered = np.full(raw.size, np.nan)
    for start, end in runs(valid):
        if end - start > 30:
            filtered[start:end] = sosfiltfilt(sos, detrend(raw[start:end], type="linear"))
    values = np.full(length, np.nan)
    edges = np.rint(np.arange(length + 1) * fs).astype(int)
    for second, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        if start < 0 or start >= filtered.size or end <= start:
            continue
        # EEG defines the score duration. An equally long EMG vector can therefore
        # end within the final, ceiling-rounded score second; use that finite tail
        # rather than turning an otherwise valid final Wake label into an error.
        segment = filtered[start : min(end, filtered.size)]
        if np.all(np.isfinite(segment)):
            values[second] = np.sqrt(np.mean(segment**2))
    return values


def active_mask(envelope, wake_mask, threshold, config):
    """Threshold the 20 Hz envelope, bridge short Wake dips, and retain bouts.

    Only valid Wake envelope bins may be active. Interior dips up to
    ``gap_tolerance`` are joined, then bouts shorter than ``min_duration`` are
    removed. Durations are measured on the intermediate envelope, before the result
    is collapsed to the app's one-second labels by :func:`second_activity`.
    """
    eligible = np.repeat(wake_mask, config.envelope_rate)[: len(envelope)]
    eligible &= np.isfinite(envelope)
    active = (envelope > threshold) & eligible
    max_gap = math.floor(config.gap_tolerance * config.envelope_rate + 1e-9)
    # Only interior gaps whose every bin is eligible can be joined.
    for start, end in runs(~active):
        if (
            start > 0
            and end < active.size
            and end - start <= max_gap
            and np.all(eligible[start:end])
        ):
            active[start:end] = True
    minimum = math.ceil(config.min_duration * config.envelope_rate - 1e-9)
    for start, end in runs(active):
        if end - start < minimum:
            active[start:end] = False
    return active


def second_activity(active, length, rate=20):
    """Map intermediate activity to one-second labels by time occupancy.

    A second is Active when at least half of its ``rate`` bins are active; at the
    default 20 Hz this is ten of twenty 50 ms envelope bins. This preserves one saved
    score per second while avoiding dependence on the exact second-boundary placement
    of a sustained activity bout.
    """
    bins = np.arange(len(active)) // rate
    counts = np.bincount(bins, minlength=length)[:length]
    active_counts = np.bincount(bins, weights=active, minlength=length)[:length]
    return (counts > 0) & (active_counts >= 0.5 * counts)


def nrem_baseline_threshold(envelope, nrem_mask, config):
    """Return a conservative threshold: NREM percentile + multiplier * robust SD.

    The automatic reference comes from finite NREM envelope values, using the
    configured percentile plus ``nrem_deviation_multiplier * 1.4826 * MAD`` around
    the NREM median. This prevents the Wake distribution, which may have been
    influenced by EMG during scoring, from defining its own activity baseline.
    """
    nrem_bins = np.repeat(nrem_mask, config.envelope_rate)[: len(envelope)]
    reference = envelope[nrem_bins & np.isfinite(envelope)]
    if reference.size == 0:
        raise ValueError(
            "Wake activity detection requires valid NREM EMG for its automatic baseline. "
            "Set WAKE_ACTIVITY_THRESHOLD to use an explicit RMS threshold instead."
        )
    median = float(np.median(reference))
    robust_sd = float(1.4826 * np.median(np.abs(reference - median)))
    percentile = float(np.percentile(reference, config.nrem_baseline_percentile))
    threshold = percentile + config.nrem_deviation_multiplier * robust_sd
    return threshold, percentile, robust_sd, int(reference.size)


def _subdivide_wake_nrem_envelope(
    emg, frequency, sleep_scores, user_sleep_scores=None, config=None
):
    """Anchor automatic activity to NREM EMG, then preserve explicit subtypes.

    Generic Wake is a coarse label, never an example of Quiet. Candidate errors
    are measured before manual overrides; class-balanced error gives each supplied
    subtype equal weight. Ties keep the threshold nearest its NREM-derived value.
    """
    config = config or WakeActivityConfig()
    coarse = coarse_sleep_scores(sleep_scores)
    users = normalize_sleep_scores(user_sleep_scores, len(coarse))
    coarse = overlay_user_sleep_scores(coarse, coarse_sleep_scores(users))
    wake = coarse == WAKE
    if not np.any(wake):
        return WakeActivityResult(coarse, np.array([]), 0.0, 0, [], {})
    envelope = emg_envelope(emg, frequency, config)[: len(coarse) * config.envelope_rate]
    bins = np.arange(envelope.size) // config.envelope_rate
    counts = np.bincount(bins, minlength=len(coarse))[: len(coarse)]
    bad = np.bincount(bins, weights=~np.isfinite(envelope), minlength=len(coarse))[: len(coarse)]
    if np.any(wake & ((counts == 0) | (bad > 0))):
        raise ValueError(
            "Wake contains missing, flatlined, or invalid EMG. Scores were not changed."
        )
    wake_values = envelope[np.repeat(wake, config.envelope_rate)[: envelope.size]]
    labelled = wake & np.isin(users, [ACTIVE_WAKE, QUIET_WAKE])
    if config.threshold is None:
        initial, baseline_percentile, baseline_robust_sd, baseline_samples = (
            nrem_baseline_threshold(envelope, coarse == NREM, config)
        )
        baseline_source = "NREM"
    else:
        initial = float(config.threshold)
        baseline_percentile = None
        baseline_robust_sd = None
        baseline_samples = 0
        baseline_source = "configured_threshold"

    def classify(threshold):
        activity = active_mask(envelope, wake, threshold, config)
        return activity, second_activity(activity, len(coarse), config.envelope_rate)

    threshold = initial
    if np.any(labelled):
        candidates = np.unique(
            np.r_[0.0, np.percentile(wake_values, np.linspace(0, 100, 41)), initial]
        )
        targets = users == ACTIVE_WAKE

        def error(candidate):
            _, predicted = classify(candidate)
            errors = [
                np.mean(predicted[labelled & (targets == value)] != value)
                for value in (False, True)
                if np.any(labelled & (targets == value))
            ]
            return float(np.mean(errors)), abs(candidate - initial), float(candidate)

        threshold = float(min(candidates, key=error))
    activity, seconds = classify(threshold)
    result = coarse.copy()
    result[wake] = QUIET_WAKE
    result[wake & seconds] = ACTIVE_WAKE
    # Fine labels remain authoritative; generic Wake must not erase subdivision.
    result[labelled] = users[labelled]
    bouts = [(s / config.envelope_rate, e / config.envelope_rate) for s, e in runs(activity)]
    metadata = {
        "schema_version": 2,
        "algorithm": "emg_rms_nrem_baseline_sustained_v2",
        "method": "nrem_envelope",
        "config": asdict(config),
        "threshold_used": threshold,
        "threshold_initial": initial,
        "threshold_baseline_source": baseline_source,
        "nrem_baseline_percentile_rms": baseline_percentile,
        "nrem_baseline_robust_sd_rms": baseline_robust_sd,
        "nrem_baseline_samples": baseline_samples,
        "calibrated_seconds": int(labelled.sum()),
        "sampling_rate": float(frequency),
        "bandpass_hz": [20, min(200, 0.45 * float(frequency))],
        "automatic_active_bouts_relative_seconds": bouts,
        "label_mapping": {"Wake": 0, "Active Wake": 4, "Quiet Wake": 5},
    }
    return WakeActivityResult(result, envelope, threshold, int(labelled.sum()), bouts, metadata)


def _subdivide_wake_ranked_per_second(emg, frequency, sleep_scores, user_sleep_scores, config):
    """Split each recording's Wake seconds by direct RMS rank, preserving fine labels."""
    coarse = coarse_sleep_scores(sleep_scores)
    users = normalize_sleep_scores(user_sleep_scores, len(coarse))
    coarse = overlay_user_sleep_scores(coarse, coarse_sleep_scores(users))
    wake = coarse == WAKE
    if not np.any(wake):
        return WakeActivityResult(coarse, np.array([]), 0.0, 0, [], {})
    rms = emg_second_rms(emg, frequency, len(coarse))
    if np.any(wake & ~np.isfinite(rms)):
        raise ValueError(
            "Wake contains missing, flatlined, or invalid EMG. Scores were not changed."
        )
    labelled = wake & np.isin(users, [ACTIVE_WAKE, QUIET_WAKE])
    manual_active = labelled & (users == ACTIVE_WAKE)
    unlabelled = wake & ~labelled
    wake_count = int(wake.sum())
    target_active = int(math.floor(config.active_fraction * wake_count + 0.5))
    needed = int(np.clip(target_active - manual_active.sum(), 0, unlabelled.sum()))
    candidates = np.flatnonzero(unlabelled)
    # Exact RMS ties are resolved by earlier recording second to keep reruns deterministic.
    ranked = candidates[np.lexsort((candidates, -rms[candidates]))]
    active = manual_active.copy()
    active[ranked[:needed]] = True
    result = coarse.copy()
    result[wake] = QUIET_WAKE
    result[active] = ACTIVE_WAKE
    result[labelled] = users[labelled]
    active_count = int((result == ACTIVE_WAKE).sum())
    cutoff = float(rms[ranked[needed - 1]]) if needed else float("nan")
    bouts = [(float(start), float(end)) for start, end in runs(active)]
    metadata = {
        "schema_version": 3,
        "algorithm": "filtered_emg_rms_wake_rank_v3",
        "method": "wake_rank_per_second",
        "config": asdict(config),
        "activity_feature": "one_second_filtered_emg_rms",
        "threshold_used": cutoff,
        "target_active_fraction": float(config.active_fraction),
        "target_active_seconds": target_active,
        "wake_seconds": wake_count,
        "active_seconds": active_count,
        "achieved_active_fraction": active_count / wake_count,
        "manual_active_seconds": int(manual_active.sum()),
        "manual_quiet_seconds": int((labelled & (users == QUIET_WAKE)).sum()),
        "target_constrained_by_manual_labels": active_count != target_active,
        "tie_breaker": "earlier_recording_second",
        "calibrated_seconds": int(labelled.sum()),
        "sampling_rate": float(frequency),
        "bandpass_hz": [20, min(200, 0.45 * float(frequency))],
        "automatic_active_bouts_relative_seconds": bouts,
        "label_mapping": {"Wake": 0, "Active Wake": 4, "Quiet Wake": 5},
    }
    return WakeActivityResult(result, rms, cutoff, int(labelled.sum()), bouts, metadata)


def subdivide_wake(emg, frequency, sleep_scores, user_sleep_scores=None, config=None):
    """Apply the configured per-recording Wake subdivision method.

    ``wake_rank_per_second`` is the default: direct one-second filtered EMG RMS is
    ranked within the recording's Wake seconds to target the configured Active-Wake
    fraction, while explicit manual subtypes remain fixed. ``nrem_envelope`` retains
    the prior NREM-anchored 20 Hz method as a parked comparison alternative.
    """
    config = config or WakeActivityConfig()
    if config.method == "wake_rank_per_second":
        return _subdivide_wake_ranked_per_second(
            emg, frequency, sleep_scores, user_sleep_scores, config
        )
    return _subdivide_wake_nrem_envelope(emg, frequency, sleep_scores, user_sleep_scores, config)
