# Next Steps

Use this checklist alongside `codex_work_log.md`.

## Statistical Wake Model

- Keep the first-pass feature simple:
  - compute the app-style EEG spectrogram
  - normalize the displayed spectrogram values globally
  - average the normalized `1-5 Hz` rows column-wise
  - label Wake where the feature falls below a threshold
- Current best visual threshold candidate: `0.6`.
- Continue testing nearby values, especially below `0.6`, using `scripts/visualize_low_band_wake_bouts.py`.

## Immediate Next Experiment

- Add simple merge rules to reduce fragmented Wake labels:
  - merge Wake bouts separated by very short non-Wake gaps
  - optionally remove Wake bouts shorter than a minimum duration
  - compare several gap and duration values visually before choosing defaults
- Keep the merge-rule parameters editable in the script's direct-run block.
- Preserve the raw threshold-only behavior as an option so the effect of merge rules is easy to compare.

## Notes

- The current script already has `MERGE_GAP_S` and `MIN_BOUT_DURATION_S` parameters, but the next pass should tune and document useful defaults.
- Visual inspection is the intended evaluation method for now; quantitative fitting against ground truth can come after the feature and merge rules look plausible.
