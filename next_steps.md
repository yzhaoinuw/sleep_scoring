# Next Steps

Use this as the forward-looking checklist. Completed experiments and measured
outcomes live in `work_log.md`.

## Installation Packaging

Current status:

- Windows users currently get the app as a PyInstaller-built zip containing
  `_internal/`, `app_src/`, `models/`, and `run_desktop_app.exe`.
- The current layout intentionally keeps `app_src/` beside the executable so
  app-code-only updates can be shipped as a small replacement-folder zip.
- Keep that lightweight `app_src/` update workflow. The packaging work should
  make it safer and more repeatable, not force every update into a full rebuild.
- First-pass packaging scripts now exist on `experiments/installation-packaging`;
  both the full app zip path and the `app_src` update zip path have been
  smoke-tested with tests skipped.
- The full app zip now removes Torch from `_internal/` while keeping sDREAMER
  code and checkpoint files in place. Users who need sDREAMER can add the
  optional `torch` folder under `_internal/`.
- The generated no-Torch full app zip was manually unzipped and launched
  successfully on Windows.

Immediate experiment:

- Apply a generated `app_src` update to an existing app folder and verify the
  patched app starts.
- Decide whether to add a double-click patch helper after seeing whether manual
  replacement remains confusing.
- If users need sDREAMER, keep the README add-on path centered on adding the
  optional `torch` folder under `_internal/`.

Release types:

- Use a full app package when `_internal/`, dependencies, Python, PyInstaller,
  `run_desktop_app.py`, model files, or runtime layout changes.
- Use an `app_src` update when changes are limited to tracked files under
  `app_src/` and do not add third-party dependencies or model/runtime changes.

Validation:

- Run the normal test suite before full packaging unless explicitly skipped.
- Verify release zips contain the expected top-level files and folders.
- Run `run_desktop_app.exe --smoke` from the full app build before zipping.
- Record the app version, git commit, branch, dependency snapshot, artifact
  name, and SHA256 hash in generated release metadata.
- Manually test the full package and at least one `app_src` update before relying
  on the workflow for a shipped release.

Possible later upgrades:

- Add a double-click launcher helper that unblocks downloaded files and starts
  the app.
- Consider an installer and code signing only after the zip workflow is boring
  and repeatable.

## Visualization Performance

Current status:

- Navigation now feels smooth across keyboard, mouse drag, wheel zoom, and reset.
- Server-side resampler work is no longer the main bottleneck.
- Final refresh payloads are compacted, usually around `95-110 KB`.
- Direct browser-side `Plotly.restyle` is the active final refresh path.
- Remaining cost is mostly Plotly/WebGL redraw time.
- Perf logging is off by default for shipped users; env-var overrides remain
  available for opt-in profiling.
- Mac M4 baseline captured on 2026-05-25 (see
  `ui_response_time_optimization_progress.txt`).

Active experiments:

- None before the next shipped build. The latest low-hanging UI optimization
  probes have been resolved; see "Do not revisit for now" below for each
  outcome.

Measurement protocol (for any future probe):

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
- Wrapping `x`/`y` as `Float32Array` before `Plotly.restyle`. Probed on
  2026-05-26 with no measurable change in `dash_apply` or auto-pan
  `apply` vs the baseline; Plotly already converts internally.
- Switching `hovermode` from `"x unified"` to `"x"`. Probed on
  2026-05-26. Apply time dropped by a real but modest ~5-10 ms across
  gesture types, but this drops the cross-subplot synchronized spike
  line and combined tooltip. Multiple users specifically requested
  the unified-crosshair behavior, so the UX trade-off is not worth
  the speedup. Do not revisit unless we find a way to keep the
  unified visual.
- Swapping `Plotly.restyle` for `Plotly.react` (with bumped
  `layout.datarevision`) in `graphDirectRestyle.js`. Probed on
  2026-05-26 and clearly regressed: `dash_apply` jumped by ~100-150 ms
  across native release, keyboard, and custom-drag gestures vs the
  baseline. Cause is that `react` re-diffs the full figure
  (spectrogram heatmap, sleep-score Heatmap, legend, layout) on every
  call, while `restyle` patches only the named props on the named
  trace indices. Auto-pan was unaffected because it lives in
  `annotationAutoPan.js` and has its own merge path.

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
- Broader experiment scripts are local-only and ignored under `scripts/`, not
  part of the shipped app source.
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

## Further Down The Line / Just A Thought

- Multi-session support on one computer is low priority. If ever needed, launch
  each app instance on its own free port and isolate cache/temp/video outputs per
  process/session; current user guidance is one app session per computer.
