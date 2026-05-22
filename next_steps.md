# Next Steps

Use this checklist alongside `codex_work_log.md`.

## Navigation And Extended Selection Feedback

Users raised two related interaction problems:

- In annotation mode, drag selection is limited to the currently visible x-range.
  - Desired behavior: select a region that extends beyond the current view without stopping to pan manually.
  - Likely implementation shape: custom client-side auto-pan while selection is active, preserving or extending the selected range until mouse release.
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
  - Use the new browser navigation profiler to measure input-to-afterplot timing:
    - enable browser-only logs with `ENABLE_BROWSER_NAVIGATION_PERF_LOG = True` in `app_src/config.py`
    - enable server and browser logs with `ENABLE_RESAMPLER_PERF_LOG = True` in `app_src/config.py`
    - env vars `SLEEP_SCORING_BROWSER_NAV_PERF_LOG=1` and `SLEEP_SCORING_RESAMPLER_PERF_LOG=1` still work for one-off runs
    - compare `[resampler] total` against `[browser-nav] dash_apply` and `browser_total`
    - if browser timing dominates, prioritize client-side active navigation and x-array payload elimination
  - Current cadence experiment:
    - `ENABLE_FAST_NAVIGATION_TRACE_UPDATES = False`
    - active movement skips fast trace patches and only applies final trace refresh after idle/release
    - compare subjective pan/zoom feel and final settle delay against the fast/final baseline
  - Current custom drag experiment:
    - `app_src/assets/graphCustomPointerPan.js`
    - custom horizontal pointer drag bypasses native Plotly drag in pan mode
    - coalescer ignores Plotly relayout events during custom drag, and custom drag requests final-only refresh on release
    - custom drag also pans EEG/EMG y-axis ranges when dragging inside those rows
    - relayout echo suppression is active around custom drag so logs should show `source=custom-drag`
    - confirmed: EEG/EMG vertical drag works, mouse drag panning is noticeably faster, and final refresh still settles
    - measured custom-drag final browser totals are roughly 304-400 ms versus roughly 745-853 ms on the normal Plotly/coalesced final path in the same run
  - Current best status:
    - keyboard panning feels smooth
    - mouse drag panning is noticeably faster after custom pointer pan
    - active trace updates are disabled during movement, with final detail refresh after release/idle
    - remaining gap: mouse drag still feels a little slower than keyboard panning
  - Tune `FINAL_IDLE_MS` in `app_src/assets/graphRelayoutCoalescer.js` if the final refresh feels too early or too late.
  - Suppress duplicate fast updates for unchanged or near-unchanged ranges more aggressively.
  - Explore deriving regular x arrays client-side or otherwise avoiding repeated x-array payloads.
  - Consider precomputed downsample tiers per loaded file if on-demand resampling becomes a bottleneck again.
- Only after navigation feels responsive, revisit drag-select auto-pan.
  - Prototype edge-triggered x-axis panning during annotation selection.
  - Preserve the final selected `[start, end]` range for the existing annotation flow.
  - Validate behavior across all subplots before making it the default.

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
