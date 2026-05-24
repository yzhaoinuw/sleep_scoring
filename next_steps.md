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
- UI response optimization is paused until real user feedback after the next shipped version.

No active pre-ship experiment:

- Do not chase more UI response optimizations before shipping annotation auto-pan.
- Let users decide whether the current responsiveness is sufficient in practice.

Possible later ideas:

- Explore whether regular or partly regular `x` arrays can be derived client-side.
- Consider precomputed downsample tiers only if on-demand resampling becomes a bottleneck again.

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

Remaining polish:

- Clamp auto-pan lead requests at recording bounds.
  - When dragging briefly past the end of the recording, the lead request can go beyond
    available trace data.
  - Manual testing on the dev-line feature showed a momentary straight-line trace before
    the final replace refresh recovered.
  - The next small fix should clamp or skip out-of-bounds lead windows while still
    allowing the selection frontier to reach the true recording end.

Guardrails:

- Keep normal click thin-box selection and drag-box selection intact.
- Keep the nonzero-`start_time` click-selection fix intact.
- Do not start with double-click or modifier-click gesture inference.

Possible later experiment:

- Revisit explicit full-bout selection with a right-click/context-menu style gesture.

## Statistical Model

Immediate goal:

- Improve REM detection inside long Wake bouts without destabilizing current defaults.

Next experiment:

- Allow the REM detector to carve out a likely REM subsection from within a Wake bout
  instead of relabeling the entire Wake bout.
- Compare:
  - identifying a low-NE subsection before Wake-to-REM promotion
  - splitting a Wake-derived REM candidate after initial REM relabeling
- Focus on files where merged Wake currently swallows a smaller likely REM region.

Validation plan:

- Use side-by-side visual inspection in the app first.
- Prioritize Wake bouts containing likely REM subsections, post-REM Wake boundary
  placement, and files where merged Wake is too broad.

Notes:

- `app_src/run_inference_stats_model.py` is the active app-side stats model.
- `scripts/visualize_low_band_wake_bouts.py` remains the broader experiment sandbox.
