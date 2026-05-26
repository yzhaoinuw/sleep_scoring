# Next Steps

Use this as the forward-looking checklist. Completed experiments and measured
outcomes live in `work_log.md`.

## Visualization Performance

Current status:

- Navigation now feels smooth across keyboard, mouse drag, wheel zoom, and reset.
- Server-side resampler work is no longer the main bottleneck.
- Final refresh payloads are compacted, usually around `95-110 KB`.
- Direct browser-side `Plotly.restyle` is the active final refresh path.
- Remaining cost is mostly Plotly/WebGL redraw time.
- Mac M4 baseline captured on 2026-05-25 (see
  `ui_response_time_optimization_progress.txt`).
- The final refresh now fetches `/_sleep_scoring/resample` directly and applies
  via `graphDirectRestyle`, bypassing the Dash callback and store roundtrip
  (landed 2026-05-25). Apples-to-apples measurement against the baseline
  showed no measurable change in browser_total; kept as structural cleanup
  before further items.

Active branch:

- `optimization/further_ui_speedup`, branched from `dev` at `7a867bb`.

Active experiments, in order:

1. Synthesize regular `x` arrays client-side.
   - EEG/EMG `x` is uniform (`start_time + i / eeg_freq`); the resampler
     picks indices that are recoverable from `(x0, dx, n)`.
   - Send `{x0, dx, n, y}` in the resampler patch; reconstruct `x` in the
     browser before `Plotly.restyle`.
   - Expected gain: halves the EEG/EMG payload and skips the WebGL
     x-buffer rebuild on apply. Biggest impact at wide zoom-outs where
     payload doubles to about 168 KB.

2. Send `y` as Float32 binary.
   - Once the final refresh uses the direct fetch endpoint, switch `y`
     arrays to `application/octet-stream` `Float32Array` to skip JSON
     encode/parse on both ends.
   - Pairs naturally with item 1.

3. A/B `hovermode`.
   - Quick timing pass with `hovermode` set to `"x"` or `"closest"` vs.
     the current `"x unified"` to see if unified-hover bookkeeping is
     part of the residual `Plotly.restyle` apply cost.

4. Compare `Plotly.react` vs `Plotly.restyle`.
   - Profile a versioned-data `Plotly.react` path against the current
     `Plotly.restyle` in `graphDirectRestyle.js` on the same recording.

5. Default perf logging to off in shipped config (final cleanup).
   - Flip `ENABLE_RESAMPLER_PERF_LOG` and
     `ENABLE_BROWSER_NAVIGATION_PERF_LOG` to `False` in `app_src/config.py`
     so the resampler callback stops paying the JSON-encode + summarize
     cost on every update for users; keep the env-var override path intact.
   - Deferred to last so each item above can be measured against the
     baseline with logs still on.

Measurement protocol:

- Use the 2026-05-25 Mac M4 baseline in
  `ui_response_time_optimization_progress.txt` as the anchor.
- Each item lands as its own commit so per-item before/after numbers stay
  attributable.

Possible later ideas:

- Consider precomputed downsample tiers only if on-demand resampling
  becomes a bottleneck again.

Do not revisit for now:

- Adaptive final refresh density by visible window width.
- Fast server trace updates during active movement.
- Visualization-only source downsampling to 128 Hz.

## Annotation Selection

Current status:

- Edge-triggered x-axis panning during annotation drag selection is implemented in
  `app_src/assets/annotationAutoPan.js`.
- The final selected `[start, end]` range is preserved for the existing annotation flow.
- Auto-pan direct trace refreshes use `/_sleep_scoring/resample` and browser-side
  `Plotly.restyle` so live selection does not compete with normal Dash graph updates.
- Browser-side merge buffers are capped so long auto-pan drags stay around `7k-8k`
  active points instead of growing without bound.
- This work still needs manual validation on top of the optimized
  `codex/next-level-navigation` stack after the branch integration.

Pre-ship validation:

- Re-test normal zooming, keyboard panning, mouse-drag panning, and reset.
- Re-test annotation click selection, normal drag selection, and drag-select auto-pan.
- Re-test mode switches and sampling levels `x0.5`, `x1`, `x2`, and `x4`.
- Watch for stale-trace snapback, lost final detail refresh, or annotation selection drift.

Guardrails:

- Keep normal click thin-box selection and drag-box selection intact.
- Keep the nonzero-`start_time` click-selection fix intact.
- Do not start with double-click or modifier-click gesture inference.

Possible later experiment:

- Revisit explicit full-bout selection with a right-click/context-menu style gesture.

## Statistical Model

Current status:

- `app_src/run_inference_stats_model.py` is the active app-side stats model.
- `scripts/visualize_low_band_wake_bouts.py` remains the broader experiment sandbox.
- Immediate goal: improve REM detection inside long Wake bouts.

Next experiment:

- Allow the REM detector to carve out a likely REM subsection from within a Wake bout
  instead of relabeling the entire Wake bout.
- Compare:
  - identifying a low-NE subsection before Wake-to-REM promotion
  - splitting a Wake-derived REM candidate after initial REM relabeling
- Focus on files where merged Wake currently swallows a smaller likely REM region.

Validation:

- Use side-by-side visual inspection in the app first.
- Prioritize Wake bouts containing likely REM subsections, post-REM Wake boundary
  placement, and files where merged Wake is too broad.

Guardrails:

- Do not destabilize current defaults while improving REM-in-Wake detection.
