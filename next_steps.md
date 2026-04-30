# Next Steps

Use this checklist alongside `codex_work_log.md`.

## Annotation Feature Status

- Current reliable baseline:
  - normal click still creates the thin selection box
  - drag-box selection still works
- Keep the nonzero-`start_time` click-selection fix in place.
- The `B`-armed full-bout selection should be treated as a previously working experiment, not as the current committed code.
- If revisiting this feature, start from the current committed baseline and reintroduce explicit bout-select behavior incrementally rather than assuming it is already present.

## Annotation Feature Next Experiment

- If we return to the faster "select existing bout" architecture later:
  - reintroduce a segment-store idea incrementally
  - keep debug mode on from the start
  - avoid overlapping click callbacks that write to the same outputs
- Do not start with:
  - double-click detection
  - `Ctrl`/`Cmd` modifier-click
  - large callback rewires before confirming event behavior in the live app
- More promising candidate than double click:
  - pilot a right-click / context-menu style gesture for full-bout selection, if the graph surface and desktop wrapper expose it cleanly
  - treat it as an explicit alternate gesture, not as inferred timing logic

## Current Statistical Model

- Wake detection:
  - compute the app-style EEG spectrogram
  - clip and normalize the displayed spectrogram values
  - average the normalized sleep-wave band column-wise
  - label Wake where the feature falls below the threshold
- Wake cleanup:
  - merge short NREM gaps relative to neighboring Wake durations
  - remove very short Wake bouts
- REM detection:
  - start from Wake bouts that pass the duration rule
  - require low NE relative to a global low-percentile threshold
  - currently skip REM shape gating to match the validated `shape_test="none"` behavior
- Post-REM Wake:
  - within each REM bout, find the NE trough
  - after the trough, split at the first cumulative NE recovery crossing above epsilon
  - relabel the recovery tail as Wake

## Immediate Goal

- Improve REM detection inside long Wake bouts without destabilizing the current working defaults.

## Next Experiment

- Instead of relabeling an entire Wake bout as REM, allow the REM detector to carve out a likely REM subsection from within a Wake bout.
- Compare two approaches:
  - identify a low-NE subsection before promoting Wake to REM
  - or split a Wake-derived REM candidate after the initial REM relabeling step
- Focus on files where merged Wake currently swallows a smaller REM region.

## App Status

- The app can now switch between:
  - `sdreamer`
  - `stats_model`
- Selection is config-only through `app_src/config.py`.
- Legacy app postprocessing should remain disabled for the stats model path.

## User-Facing Controls

- Current exposed stats-model controls in `app_src/config.py`:
  - Wake threshold
  - minimum Wake duration
  - minimum REM duration

## Developer Controls

- Keep these internal for now:
  - sleep-wave frequency range
  - spectrogram normalization range
  - Wake merge coefficient
  - REM threshold percentile
  - REM threshold comparison percentile
  - NE smoothing window
  - REM recovery epsilon

## Validation Plan

- Use side-by-side visual inspection in the app first.
- Focus especially on:
  - Wake bouts that contain a smaller likely REM subsection
  - post-REM Wake boundary placement
  - files where merged Wake looks too broad before REM relabeling
  - how stable the defaults feel across files

## Notes

- `app_src/run_inference_stats_model.py` is now the active app-side stats model.
- `scripts/visualize_low_band_wake_bouts.py` remains the broader sandbox for experiment history and visual debugging.
