# Next Steps

Use this checklist alongside `codex_work_log.md`.

## Navigation And Extended Selection Feedback

Status: implemented and manually validated.

Users raised two related interaction problems:

- In annotation mode, drag selection is limited to the currently visible x-range.
  - Implemented behavior: client-side annotation drag auto-pan keeps extending the selected range beyond the current view until mouse release.
  - Trace updates during auto-pan use direct browser fetches to `/_sleep_scoring/resample`, request a small lead window in the drag direction, merge the returned trace data into the visible graph, and replace with the exact final range on release.
- Trace updates feel too slow during zooming and panning.
  - Current keyboard panning updates `relayoutData` client-side in `pan_figure`, but the waveform update still depends on the server-side Plotly-resampler callback.
  - The slow part may be Python resampling, Dash transport/serialization, browser redraw, or a combination.

Treat these as one engineering problem before implementing auto-pan selection. If normal pan/zoom latency is already high, auto-panning during selection may amplify the same bottleneck and feel worse.

Proposed plan:

- Measurement completed so far.
  - Added env-gated resampler profiling with update timing, overlap/idle timing, payload size, x-range width, and patch breakdown.
  - Original default `x1` updates were roughly 300 KB with 22 patch operations.
  - The server usually keeps up with individual callbacks, but fast interaction can leave very small idle gaps or occasional overlap.
  - The dominant cost is shipping/redrawing EEG, EMG, and NE x/y arrays.
- Payload reductions completed so far.
  - Removed point markers from EEG, EMG, and NE while keeping signal lines black.
  - Removed theta/delta `customdata`.
  - Made theta/delta static/full-resolution so it drops out of relayout patches.
  - Capped NE relayout updates at 1024 samples while keeping EEG/EMG at normal `x1` density.
  - Added optional `x0.5` sampling level for fast-mode testing, but kept `x1` as the default.
  - Default `x1` payload dropped to roughly 180 KB with 9 patch operations.
- Next optimization candidates.
  - Do not lower the normal/default EEG/EMG display density unless users explicitly accept the visual tradeoff.
  - Visualization-only EEG/EMG source downsampling to 128 Hz was tested and reverted:
    - it did not noticeably improve interaction
    - profiling stayed around 55-80 ms construction time
    - payload increased to roughly 200 KB because the displayed point count stayed at `x1`
  - Debounce/coalescing prototype added in `app_src/assets/graphRelayoutCoalescer.js`.
    - It now routes all range relayouts through fast transient updates and schedules final updates after idle.
  - "Coarse while moving, detailed after idle/release" prototype added and manually validated:
    - Use lower-density patches only for active movement.
    - Restore normal `x1` detail after idle/release so the final view keeps user-facing detail.
    - Measured fast callbacks are roughly 11-15 ms with a 55-56 KB payload; final callbacks are roughly 20-30 ms with normal 180-185 KB detail.
  - In-memory resampler storage added:
    - `resampler_get` is now near zero instead of the previous 200-600 ms filesystem-cache retrieval cost.
- Further optimization candidates, only if navigation still needs more polish.
  - Tune `FINAL_IDLE_MS` in `app_src/assets/graphRelayoutCoalescer.js` if the final refresh feels too early or too late.
  - Suppress duplicate fast updates for unchanged or near-unchanged ranges more aggressively.
  - Explore deriving regular x arrays client-side or otherwise avoiding repeated x-array payloads.
  - Consider precomputed downsample tiers per loaded file if on-demand resampling becomes a bottleneck again.
- Drag-select auto-pan status.
  - Edge-triggered x-axis panning during annotation selection is now implemented in `app_src/assets/annotationAutoPan.js`.
  - The final selected `[start, end]` range is preserved for the existing annotation flow.
  - Browser-side merge buffers are capped so long auto-pan drags stay around 7k-8k active points instead of growing without bound.
  - Profiling from long manual drags showed direct resampler requests usually around 13-16 ms server time, browser merge applies usually around 230-305 ms, and no progressive point-count growth.

Remaining polish:

- Clamp auto-pan lead requests at the recording bounds.
  - When dragging briefly past the end of the recording, the lead request can go beyond available trace data.
  - Manual testing showed a momentary straight-line trace before the final replace refresh recovers.
  - The next small fix should clamp or skip out-of-bounds lead windows while still allowing the selection frontier to reach the true recording end.

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
