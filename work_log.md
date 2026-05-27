# Work Log

Prepend new session notes to the top of this file. If you log multiple
sessions on the same calendar date, add a new `###` subsection under the
existing `## YYYY-MM-DD` header instead of starting a second header for the
same date.

Historical verification commands may include absolute paths from the original
development machine. When replaying or adapting them, keep the project folder
name `sleep_scoring` and conda environment name `sleep_scoring_dash3.0`, but
replace the user/home prefix and clone location with the collaborator's local
setup.

Reading note for agents: this file holds at most the 5 most recent unique
calendar dates. Older entries are rotated in chunks of 5 dates into
`work_log_archive/work_log_<earliest>_to_<latest>.md`. Default to reading the
two most recent dated entries; search older entries with targeted terms using
the `^## [0-9]{4}-[0-9]{2}-[0-9]{2}` anchor, or open the relevant archive file
by its date range. See `AGENTS.md` for the full rotation policy.

## 2026-05-27

### v0.16.1 Release Prep

- Added README documentation for right-click contiguous-segment selection in annotation mode.
- Bumped the app version from `v0.16.0` to `v0.16.1`.
- Aligned the legacy `setup.py` version with the release version.
- Added a `v0.16.1` changelog entry covering right-click segment selection, UI response work, annotation auto-pan responsiveness, and default-off performance logging.
- Confirmed before release that resampler/browser navigation performance logging remains off by default and can still be enabled with `SLEEP_SCORING_PROFILE_RESAMPLER=1`.

## 2026-05-26

### TypedArray Probe On Direct-Restyle Apply Path

- Tested whether wrapping `x` and `y` per-trace values as `Float32Array`
  inside `app_src/assets/graphDirectRestyle.js` before `Plotly.restyle`
  would reduce the residual apply cost. Browser-side change only; no
  server, transport, or endpoint changes.
- Apples-to-apples manual measurement against the 2026-05-25 Mac M4
  baseline (`830.mat`, ~15,295 s) showed no movement in apply time:
  - Native Plotly release at `x_width=15295 s`: `browser_total=261 ms`
    vs baseline 251-259 ms.
  - Native Plotly release at `x_width=457 s`: `browser_total=232-237 ms`
    vs baseline 224-295 ms.
  - Keyboard at `x_width=457 s`: `dash_apply=189-196 ms`,
    `browser_total=502-524 ms` vs baseline ~190 ms / 480-549 ms.
  - Annotation auto-pan at `x_width=446-457 s`: steady-state
    `apply=152-178 ms`, `browser_total=273-318 ms` vs baseline
    `apply=145-180 ms`, `browser_total=254-330 ms`.
- Conclusion: Plotly's WebGL path appears to convert plain JS arrays
  to typed arrays internally, so pre-wrapping does not help. Reverted
  the change rather than committing it. Logged under
  `next_steps.md` "Do not revisit for now" so future probes do not
  re-litigate this.
- Next probe queued: `hovermode` A/B. Switched
  `app_src/make_figure_dev.py` from `"x unified"` to `"x"` to test
  whether unified-hover bookkeeping is part of the residual apply
  cost. Pending manual measurement before commit-or-revert.

### Hovermode A/B (`"x unified"` vs `"x"`)

- Tested `app_src/make_figure_dev.py` hovermode switched from
  `"x unified"` to `"x"`. Note: this keeps a per-subplot vertical
  hover guide but drops the spike line synchronized across all four
  subplots and the combined multi-trace tooltip.
- Measurement against the 2026-05-25 Mac M4 baseline showed a small
  but consistent improvement on apply time:
  - Native Plotly release at `x_width=15295 s`: `browser_total=259 ms`
    vs baseline 251-259 ms (no change at this width).
  - Native Plotly release at `~457 s`: `browser_total=210-225 ms`
    vs baseline 224-295 ms (~5-15 ms lower).
  - Keyboard at `~457 s`: `dash_apply=180-186 ms`,
    `browser_total=497-511 ms` vs baseline ~190 ms / 480-549 ms
    (~5-10 ms apply drop).
  - Custom-drag at `~457 s`: `dash_apply=180-194 ms` vs the prior
    TypedArray-run baseline of 190-249 ms (~5-15 ms apply drop).
  - Annotation auto-pan steady-state at `~357 s`: `apply=149-171 ms`
    (~157 ms), `browser_total=268-316 ms` vs baseline `apply=145-180 ms`
    (~165 ms), `browser_total=254-330 ms` (~5-10 ms tighter).
- Conclusion: real but modest gain (~5-10 ms on apply). Multiple
  users specifically requested the unified-crosshair feature, so the
  UX trade-off does not justify the speedup. Reverted; logged under
  `next_steps.md` "Do not revisit for now" with the user-facing
  reason so future probes know it has been tried.
- Next planned probe: `Plotly.react` vs `Plotly.restyle` in
  `graphDirectRestyle.js` via a bumped `layout.datarevision`.

### Plotly.react vs Plotly.restyle Probe

- Swapped `Plotly.restyle` for `Plotly.react` in
  `app_src/assets/graphDirectRestyle.js`: mutated `plot.data` in place
  with the per-trace x/y/name/marker updates, bumped
  `plot.layout.datarevision`, and called
  `Plotly.react(plot, plot.data, plot.layout)`. No server change.
- Apples-to-apples measurement against the 2026-05-25 Mac M4 baseline
  showed a clear regression on every direct-restyle gesture:
  - Native Plotly release at `x_width=15295 s`: `dash_apply=379 ms`
    (`browser_total=405 ms`) vs baseline ~233-240 ms
    (`browser_total=251-259 ms`) — **+140 ms**.
  - Native Plotly release at `~65-500 s`: `dash_apply=314-347 ms`
    vs baseline ~190-205 ms — **+110-140 ms**.
  - Keyboard at `~269 s`: `dash_apply=325-331 ms`
    (`browser_total=649-657 ms`) vs baseline ~190 ms (480-549 ms total)
    — **+130-140 ms**.
  - Custom-drag at `~269 s`: `dash_apply=327-338 ms` vs baseline
    ~190-195 ms — **+130-140 ms**.
  - Annotation auto-pan: unchanged at `apply=149-171 ms` matched
    widths, confirming auto-pan uses a separate merge path in
    `annotationAutoPan.js` and never went through `graphDirectRestyle`.
- Conclusion: `Plotly.react` re-diffs the entire figure (4 subplots,
  spectrogram heatmap, sleep-score Heatmap, legend traces, layout) on
  every call. The `datarevision` bump tells react the data changed but
  does not skip the full reconciliation. `Plotly.restyle` with explicit
  trace indices patches only named props on named traces, which is
  much cheaper for this figure shape. Reverted; logged under
  `next_steps.md` "Do not revisit for now" with the cause.

### Default Perf Logging To Off In Shipped Config

- Flipped `ENABLE_RESAMPLER_PERF_LOG` and
  `ENABLE_BROWSER_NAVIGATION_PERF_LOG` to `False` in
  `app_src/config.py` so the resampler callback and browser navigation
  profiler stop paying the JSON-encode and summarize cost on every
  update for shipped users.
- Env-var override paths are intact. To re-enable per-run, set any of:
  - `SLEEP_SCORING_RESAMPLER_PERF_LOG=1` for resampler logs.
  - `SLEEP_SCORING_PROFILE_RESAMPLER=1` (umbrella; turns both on).
  - `SLEEP_SCORING_BROWSER_NAV_PERF_LOG=1` for browser-nav logs.
  - The kill switch `SLEEP_SCORING_PROFILE_RESAMPLER=0` still
    suppresses everything regardless of other env vars.
- Final cleanup item on this branch; with this and the revert above,
  the active-experiments list for Visualization Performance is now
  empty.

## 2026-05-25

### Mac M4 Baseline And Branch Setup

- Captured a Mac M4 Air baseline anchored to the same recording the Windows
  contributor used (`830.mat`, about 15,295 s) so cross-platform and
  cross-branch comparisons stay on a known footing.
  - Environment: MacBook Air (Mac16,13), Apple M4, 16 GB RAM, macOS 26.3.1,
    Safari/WebKit 26.3.1, Python 3.11.0, Dash 3.3.0, Plotly 6.5.0,
    plotly_resampler 0.11.0, pywebview 6.1.
  - Numbers and observations appended to
    `ui_response_time_optimization_progress.txt` (2026-05-25 Mac M4 baseline
    section).
- Cut `optimization/further_ui_speedup` from `dev` at
  `7a867bb Merge pull request #3 from yzhaoinuw/docs/agent-collab-polish` for
  the next round of UI response work.
- Updated `next_steps.md` Visualization Performance section with the active,
  ordered experiment list for this branch (Dash store bypass, client-side `x`
  synthesis, binary `y` transport, `hovermode` A/B, `Plotly.react` vs
  `Plotly.restyle`, perf-logging-default-off cleanup last so each preceding
  item can be measured against the baseline).

### Bypass Dash Store On Navigation Final Refresh

- Added a direct-fetch path for normal-navigation final refreshes that skips
  the Dash callback and store roundtrip:
  - `app_src/assets/graphFinalRefresh.js` exposes
    `window.sleepScoringFinalRefresh.tryDirectFetch(detail)` which fetches
    `/_sleep_scoring/resample?x0=...&x1=...&profile_id=...&mode=...&source=...`
    and applies via the existing `graphDirectRestyle` module.
  - `app_src/assets/graphRelayoutCoalescer.js` calls `tryDirectFetch` on
    final-mode coalesced events and, on success, skips dispatching the
    document event the Dash `graph-relayout-coalesced` listener picks up.
  - `app_src/app_dev.py` adds an `app.index_string` script tag that surfaces
    `window.sleepScoringConfig = {"directRestyleFinal": <bool>}` so the JS
    can decide whether to take the direct path or fall back to the Dash
    callback path.
  - `/_sleep_scoring/resample` now accepts optional `profile_id`, `mode`,
    `source` query params, runs the same `mark_navigation_profile_seen`
    plus `is_stale_navigation_profile` guards the Dash callback uses, and
    echoes a `profileMarker` back so the browser-side navigation profiler
    keeps emitting `[browser-nav]` lines that pair with `[resampler-direct]`
    lines by `browser_profile_id`.
- Annotation auto-pan, custom pointer pan, keyboard navigation, mode
  switches, sampling-level changes, and the
  `ENABLE_DIRECT_PLOTLY_RESTYLE = False` fallback are unchanged.
- Apples-to-apples measurement vs the 2026-05-25 Mac M4 baseline:
  - Native Plotly release at `x_width=15295 s`: new browser_total 257 ms vs
    baseline 251-259 ms — within noise.
  - Native Plotly release at `x_width=376 s`: new browser_total 223 ms vs
    baseline 224-295 ms — at the low end of the baseline distribution but
    only one sample.
  - Keyboard at `x_width=376 s`: new browser_total 500-501 ms vs baseline
    480-549 ms — essentially identical.
  - Annotation auto-pan at `x_width=376 s`: new fetch 124-157 / apply
    163-188 / browser_total 290-331 ms vs baseline fetch 110-145 / apply
    145-180 / browser_total 254-330 ms — within roughly 10-15 ms, no
    regression. An earlier post-change run at wider auto-pan window
    (`x_width=2165-4126 s`) showed apply 245-290 ms; that was the wider
    visible window, not the change.
- Prediction post-mortem:
  - The earlier 30-50 ms forecast was based on the gap between auto-pan
    apply (~155 ms on ~94 KB payloads) and direct-restyle apply (~245 ms on
    ~94 KB payloads) in the baseline.
  - The two paths are not apples-to-apples; auto-pan does merge/decimate
    work and skips profile-marker plumbing, so the measured gap conflated
    work with Dash-store overhead.
  - Kept the change anyway as structural cleanup; items 2-3 (client-side
    `x` synthesis and binary `y`) slot onto the direct-fetch path more
    naturally than onto the Dash-store path.
- Verification:
  - `python3 -m py_compile app_src/app_dev.py`
  - `node --check app_src/assets/graphRelayoutCoalescer.js`
  - `node --check app_src/assets/graphFinalRefresh.js`
  - Manual run on Mac confirmed `[resampler-direct] browser_profile_id=N`
    fires on each final refresh with no accompanying
    `POST /_dash-update-component` for the resampler callback.

### Revert Bypass Dash Store On Navigation Final Refresh

- Reverted the code portion of `bdfd36c` after apples-to-apples
  measurement on the 2026-05-25 Mac M4 baseline showed no
  `browser_total` change versus the prior direct-restyle path.
- Kept the baseline document, work-log archive rotation, and
  `next_steps.md` simplification from the same commit. Only the code
  files come back out:
  - `app_src/assets/graphFinalRefresh.js` removed
  - `app_src/assets/graphRelayoutCoalescer.js` restored to its
    pre-item-1 state
  - `app_src/app_dev.py` restored (drops the `app.index_string`
    `window.sleepScoringConfig` injection and the `profile_id`/`mode`/
    `source` query params plus `profileMarker` echo on
    `/_sleep_scoring/resample`)
- Corrected reasoning for skipping items 2-3:
  - MinMaxLTTB picks min/max within uniform buckets, so the resampled
    `x` array is not uniformly spaced; sending `(x0, dx, n)` would not
    represent the actual sampled positions.
  - Baseline shows `parse=0-1 ms` and `apply=145-180 ms` dominated by
    Plotly WebGL trace replacement. Transport is not the bottleneck,
    so neither item 2 (client-side `x` synthesis) nor item 3
    (binary `y`) would move `browser_total` meaningfully. For typical
    EEG/EMG magnitudes the JSON `y` array is already ~19 KB; a
    `Float32Array` would be 28 KB (larger).
- Next planned probe: TypedArray wrapping inside
  `graphDirectRestyle.js` (browser-side only, no server change) to
  test whether `Plotly.restyle` apply is faster with `Float32Array`
  inputs. See `next_steps.md`.
- Verification:
  - `python -m py_compile app_src/app_dev.py app_src/components_dev.py app_src/config.py`
  - `python -m pytest tests/test_app_helpers.py tests/test_smoke.py -q`
