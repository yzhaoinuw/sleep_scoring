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

## 2026-05-23

### Annotation Auto-Pan Bounds Clamp

- Followed up manual validation logs from the repaired integration branch:
  - normal navigation stayed on the optimized direct-restyle path
  - auto-pan direct resampler requests remained around `6-15 ms` server time
  - browser-side auto-pan merge buffers stayed bounded around `7k-8k` points
- Fixed the remaining edge behavior seen in the logs:
  - the direct resample endpoint clamps requested ranges before constructing patches
  - the graph figure stores recording x-bounds in layout meta for browser-side fetch clamping
- Followed up manual retest feedback that the first browser-side clamp was too aggressive:
  - restored the original auto-pan viewport motion and trim semantics
  - now only the fetch window is clamped in the browser before calling the direct endpoint
  - the pan loop can advance naturally while the server still avoids empty out-of-bounds patches
- Expected effect:
  - dragging past the left or right recording edge should no longer issue fully
    out-of-bounds resample requests such as negative-only x-ranges
  - selection can still reach the recording edge while trace refreshes stay within valid data

### Annotation Auto-Pan Integration Repair

- Corrected the integration direction after `dev` had advanced without the
  `codex/next-level-navigation` optimization stack:
  - switched back to `codex/next-level-navigation`
  - merged the newer `dev` auto-pan work into the optimization branch
  - resolved conflicts so the optimized final-refresh path remains the base
- Combined behavior:
  - normal navigation keeps the optimized final-only coalescer path
  - direct browser-side `Plotly.restyle` remains enabled for final resampler refreshes
  - annotation auto-pan suppresses normal relayout coalescing while selection is active
  - annotation auto-pan direct trace refreshes use the in-memory resampler through
    `get_fig_resampler()` instead of the old cache lookup path
  - direct auto-pan resampler patches now use the same numeric compaction as normal
    final refreshes
- Follow-up validation needed:
  - manually retest navigation and annotation auto-pan together on this branch before
    merging back to `dev`
  - pay special attention to stale-trace snapback, final-detail settling, and selection
    range drift during long auto-pan drags

### Annotation Drag Auto-Pan

- Added browser-side auto-pan for annotation drag selection:
  - dragging a selection beyond the left or right graph edge now pans the x-axis while
    preserving the selection range
  - the final selected range is sent through the existing annotation selection flow on
    mouse release
  - normal relayout coalescing is suppressed while annotation auto-pan is active so the
    live selection does not compete with regular navigation callbacks
- Added direct trace refreshes for auto-pan:
  - `/_sleep_scoring/resample` returns Plotly-resampler patches for browser-side refreshes
  - auto-pan requests a lead window in the drag direction and merges returned x/y data
    into the active graph
  - release uses a final replace refresh for the exact visible range
- Tuned the merge path after profiling:
  - bounded the browser-side merge buffer to prevent point-count growth during long
    auto-pan drags
  - long manual drags on the dev-line feature stayed around `7k-8k` active points
    instead of climbing into the `20k+` range
  - direct resampler server work was usually about `13-16 ms`, while browser merge
    apply was usually about `230-305 ms`
- Known follow-up:
  - dragging briefly past the recording end can request a lead window beyond available data
  - this may produce a momentary straight-line trace before the final replace refresh recovers
  - clamp the lead request to recording bounds in the next polish pass

## 2026-05-22

### UI Response Optimization Pause

- Decided to pause UI response optimization after the direct-restyle checkpoint:
  - the high-return changes have already landed
  - recent follow-up experiments show diminishing marginal returns
  - additional attempts now carry more risk of disrupting working navigation than likely speed benefit
- Current shipping plan:
  - keep direct browser-side `Plotly.restyle` as the active final refresh path
  - implement edge-triggered auto-pan while drag-selecting in annotation mode next
  - ship that version and let user feedback determine whether the current responsiveness is sufficient
- Deferred optimization ideas:
  - derive regular or partly regular `x` arrays client-side
  - revisit deeper Plotly/WebGL redraw avoidance only if user feedback says current responsiveness is not enough

### Minimal Direct Restyle Payload Experiment Dismissed

- Followed up the direct-restyle checkpoint with a smaller browser restyle payload, then reverted it:
  - added `ENABLE_DIRECT_RESTYLE_TRACE_NAME_UPDATES = False` in `app_src/config.py`
  - `build_direct_restyle_payload` now sends only direct trace data operations by default
  - trace `name` updates and layout meta are omitted from the direct-restyle payload
  - `[resampler]` logs now include `direct_ops`, which should be `6` for EEG/EMG/NE `x` and `y`
- Synthetic representative callback on `user_test_files/115_gs.mat`:
  - returned `applyPath=direct-restyle`
  - reduced direct operation count from `10` to `6`
  - confirmed operations were only `data[0].x/y`, `data[1].x/y`, and `data[6].x/y`
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\config.py app_src\components_dev.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_app_helpers.py tests\test_smoke.py -q`
  - ran bundled Node `--check` on `app_src\assets\graphDirectRestyle.js`
- Human-in-loop validation:
  - zooming, arrow-key panning, custom drag panning, annotation, and sampling-level changes felt smooth
  - logs showed `apply_path=direct-restyle` and `direct_ops=6`
  - `x1` payloads around `93-97 KB` still showed browser apply roughly `277-393 ms`
  - `x2` payloads around `175-177 KB` showed browser apply roughly `284-345 ms`
  - lower-density payloads around `59 KB` still showed browser apply roughly `269-311 ms`
- Current conclusion:
  - omitting trace-name/layout-meta restyle operations is behaviorally safe
  - timing gains are not clear; the remaining cost still looks dominated by Plotly/WebGL redraw
  - dismissed to avoid accumulating changes that do not clearly support the speed goal

### Direct Plotly Restyle Final Refresh Path

- Added a guarded direct browser restyle path for final resampler refreshes:
  - `ENABLE_DIRECT_PLOTLY_RESTYLE = True` in `app_src/config.py` routes the resampler callback to a hidden store instead of directly patching `graph.figure`
  - `app_src/assets/graphDirectRestyle.js` reads the serialized Dash Patch operations and applies trace `x`, `y`, and `name` updates with one `Plotly.restyle` call
  - `app_src/assets/graphNavigationProfiler.js` can now emit profile completion from direct-update promise completion if a normal `plotly_afterplot` marker path does not fire first
  - `app_src/components_dev.py` adds hidden stores for direct-restyle payload and status
  - `[resampler]` logs now include `apply_path=direct-restyle` or `apply_path=dash-figure-patch`
- Synthetic representative callback on `user_test_files/115_gs.mat`:
  - returned a `dict` payload with `applyPath=direct-restyle`
  - preserved full-density final payload around `94.4 KB`
  - emitted `10` serialized patch operations for the browser-side restyle path
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\config.py app_src\components_dev.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_app_helpers.py tests\test_smoke.py -q`
  - ran bundled Node `--check` on `app_src\assets\graphDirectRestyle.js` and `app_src\assets\graphNavigationProfiler.js`
  - confirmed Dash serves both assets and `_dash-layout` contains the direct-restyle stores
- Human-in-loop validation:
  - annotation, mode switch, and sampling-level changes (`x0.5`, `x1`, `x2`, `x4`) did not show stale-trace snapback or final-settle issues
  - `x1` / normal full-detail payloads around `94-100 KB` generally kept browser apply around roughly `270-370 ms`
  - `x4` payloads around `320-331 KB` generally kept browser apply around roughly `300-360 ms`, with one custom-drag outlier around `408 ms`
  - `x0.5` payloads around `59 KB` generally kept browser apply around roughly `269-311 ms`
- Current conclusion:
  - keep direct restyle as the active final refresh path because it works behaviorally and may shave a little overhead
  - the dominant remaining cost is still Plotly/WebGL redraw rather than Dash figure reconciliation

### Adaptive Final Refresh Density Experiment Dismissed

- Tried adaptive final-refresh density in `app_src/app_dev.py`, then reverted it:
  - tight windows kept the active full-detail density unchanged
  - broad final refreshes temporarily lowered only EEG, EMG, and NE `max_n_samples`
  - the active resampler trace limits were restored immediately after each patch was built
  - `[resampler]` logs included the active density band and trace limits
- The tested bands scaled from the user's active sampling setting:
  - `<=30 min`: full density
  - `<=2 h`: `0.75x`
  - `<=6 h`: `0.5x`
  - `>6 h`: `0.25x`
- Synthetic representative final update on `user_test_files/115_gs.mat`:
  - `300 s`: full density, `94.2 KB`
  - `1800 s`: full density, `95.1 KB`
  - `3600 s`: adaptive `0.75x`, `71.8 KB`
  - `10800 s`: adaptive `0.5x`, `49.5 KB`
  - `28800 s`: adaptive `0.25x`, `25.7 KB`
- Desktop-app profiling showed browser `dash_apply` remained noisy and mostly fixed-cost:
  - full density around `93-94 KB` still landed roughly `304-351 ms`
  - `0.75x` around `71-73 KB` landed roughly `294-402 ms`
  - `0.5x` around `48-51 KB` landed roughly `283-359 ms`
- Dismissal reason:
  - timing gains were inconsistent
  - the initial zoomed-out view looked visually worse and could give users the wrong impression
  - next optimization should focus on fixed Plotly/Dash redraw cost rather than lowering broad-window detail
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\config.py app_src\components_dev.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_app_helpers.py tests\test_smoke.py -q`

### Patch Payload Precision Compaction

- Added a conservative resampler patch compaction step in `app_src/app_dev.py`:
  - trace `x` arrays are rounded to 3 decimal places before Dash serializes the patch
  - trace `y` arrays are rounded to 7 decimal places before Dash serializes the patch
  - compaction runs after stale-update checks so stale patches do not pay extra work
- Synthetic representative final update on `user_test_files/115_gs.mat`:
  - initial `x5/y7` compaction dropped one representative payload from `184.6 KB` to `108.6 KB`
  - tightening x precision to milliseconds dropped a comparable representative payload to `99.2 KB`
- Follow-up tidy:
  - removed the dormant fast trace-update path now that active navigation is final-only and client-side
  - removed `fig_resampler_fast`, `FAST_NAVIGATION_N_SHOWN_SAMPLES`, `ENABLE_FAST_NAVIGATION_TRACE_UPDATES`, the hidden navigation-options bridge, and unused fast coalescer state
  - simplified in-memory resampler storage to a single active `FIG_RESAMPLER`
  - kept the profiling counters and latest-profile stale guard because they still explain overlap and browser timing in live logs
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py`
  - ran bundled Node `--check` on `app_src\assets\graphRelayoutCoalescer.js`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py app_src\config.py`
  - ran a synthetic compacted-patch size check confirming `99.2 KB`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_smoke.py -q`

### Final Refresh Cadence Optimization

- Updated `app_src/assets/graphRelayoutCoalescer.js` so native Plotly release events request the final detail refresh immediately instead of waiting for the normal idle fallback.
- Kept active movement client-side by having the browser coalescer read `ENABLE_FAST_NAVIGATION_TRACE_UPDATES` from hidden DOM configuration before sending fast server trace events.
- Added final-range duplicate suppression with a small movement tolerance so relayout echoes and unchanged ranges do not trigger redundant Dash/Plotly-resampler updates.
- Reset coalescer dispatch state when Dash replaces the graph so opening a new file is not blocked by stale duplicate-range memory.
- Area 1 cadence follow-up:
  - keyboard navigation now uses a shorter `120 ms` final-refresh settle window instead of the generic `450 ms` idle fallback
  - native Plotly release final refresh now uses a tiny `25 ms` debounce so same-frame release echoes can collapse before reaching Dash
- Verification:
  - ran bundled Node `--check` on `app_src\assets\graphRelayoutCoalescer.js`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\components_dev.py app_src\app_dev.py app_src\config.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_smoke.py -q`

## 2026-05-21

### Browser Navigation Profiling Prototype

- Created branch `codex/next-level-navigation` for the next navigation-performance push.
- Added an opt-in browser-side navigation profiler:
  - `app_src/assets/graphRelayoutCoalescer.js` now tags each coalesced relayout event with a browser profile id and `performance.now()` timestamps
  - `app_src/assets/graphNavigationProfiler.js` listens for Plotly `plotly_afterplot`, matches redraw completion back to the profile id carried by the Dash patch, and emits a compact `sleepgraphprofile` event
  - `app_src/components_dev.py` captures browser profile events through `graph-navigation-profile`
  - `app_src/app_dev.py` writes browser profile markers into resampler patches and prints `[browser-nav]` lines with coalescing, Dash/apply-to-afterplot, browser-total, frame-gap, and visible-range timing
- Profiling controls:
  - `ENABLE_BROWSER_NAVIGATION_PERF_LOG = True` in `app_src/config.py` enables browser navigation profiling
  - `ENABLE_RESAMPLER_PERF_LOG = True` in `app_src/config.py` enables server resampler logs and browser navigation logs
  - `SLEEP_SCORING_BROWSER_NAV_PERF_LOG=1` enables only browser navigation profiling
  - `SLEEP_SCORING_RESAMPLER_PERF_LOG=1` or legacy `SLEEP_SCORING_PROFILE_RESAMPLER=1` enables both server resampler logs and browser navigation logs
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py app_src\config.py run_desktop_app.py`
  - ran bundled Node `--check` on `app_src\assets\graphRelayoutCoalescer.js` and `app_src\assets\graphNavigationProfiler.js`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_smoke.py -q`
  - confirmed Dash serves the page and both profiler assets with HTTP 200 and that `_dash-layout` contains the profile event/store wiring
- Browser plugin note:
  - the in-app browser available in that agent environment blocked `localhost` / `127.0.0.1` navigation with `ERR_BLOCKED_BY_CLIENT` during this verification, so the live visual interaction pass still needs to be done in the desktop app or a normal browser.

### Area 1: Wasted Browser Redraw Suppression

- Added first low-risk stale-update suppression experiment:
  - browser coalescer now suppresses repeated fast dispatches when the x-range is unchanged or nearly unchanged
  - server callback tracks the newest browser profile id and returns `dash.no_update` for older/stale profile ids
  - stale drops are logged as `[resampler-stale]` when resampler profiling is enabled
  - opening a new file resets the profile-id guard
- Expected test signal:
  - fewer duplicate fast updates for the same or nearly same range
  - fewer unmatched fast `[resampler]` lines that never produce `[browser-nav]`
  - no loss of final detail refresh after idle/release

### Area 2: Final-Only Trace Refresh Experiment

- Added `ENABLE_FAST_NAVIGATION_TRACE_UPDATES` in `app_src/config.py`.
- Current experiment value is `False`:
  - fast relayout events still update Plotly axes immediately in the browser
  - fast server trace patches are skipped with `[resampler-skip]`
  - final idle/release patches still refresh the detailed traces
- Expected test signal:
  - active pan/zoom should feel smoother if fast trace patch redraws were the main interaction tax
  - logs should show many `[resampler-skip]` fast entries and only final `[browser-nav]` timing entries
  - final detail should still settle after stopping movement
- Human-in-loop result:
  - keyboard panning felt smooth
  - mouse dragging did not feel faster
  - conclusion: built-in Plotly drag/pan is likely a separate interaction bottleneck from server trace patch redraws

### Area 4: Custom Pointer Drag Pan Prototype

- Added `app_src/assets/graphCustomPointerPan.js` as a mouse-drag experiment:
  - only activates in Plotly pan mode
  - leaves annotation/select mode untouched
  - intercepts left-button pointer drag before native Plotly drag handling
  - computes a horizontal x-axis range shift from pointer movement
  - applies the range with a lean `Plotly.relayout` on `xaxis4.range`
  - requests final resampler refresh on pointer release through the existing coalescer path
- Expected test signal:
  - dragging should feel closer to keyboard panning if native Plotly drag was the bottleneck
  - logs should still show `[resampler-skip]` for active fast events and final `[browser-nav]` after release/idle
  - if drag direction feels inverted, flip the sign of the custom pan shift
  - if vertical EEG/EMG drag is missed, decide whether to restore native drag for y-axis or add custom y-pan later
- Human-in-loop result:
  - custom drag direction was correct
  - final trace settled correctly
  - custom drag still felt slightly slower than keyboard panning
  - vertical EEG/EMG drag was missing
- Follow-up patch:
  - coalescer now ignores Plotly relayout events during custom pointer drag
  - custom pointer drag requests final-only refresh on release with source `custom-drag`
  - custom pointer drag now pans EEG/EMG y-axis ranges when the drag starts inside those rows
- Expected follow-up signal:
  - drag movement should no longer produce active `[resampler-skip]` POSTs
  - final drag refresh should show `source=custom-drag`
  - EEG/EMG vertical drag should work again
- Human-in-loop result:
  - EEG vertical drag worked, EMG vertical drag did not
  - drag still felt slightly slower than keyboard panning
  - logs still showed active `[resampler-skip]` entries and final `source=plotly`
- Follow-up patch:
  - fixed custom y-axis candidates to EEG/EMG axes after accounting for the spectrogram secondary y-axis
  - added a temporary coalescer suppression window around custom `Plotly.relayout` calls and release so custom drag relayout echoes do not re-enter the normal Plotly coalescer path
- Expected follow-up signal:
  - EMG vertical drag should work
  - custom drag final should show `source=custom-drag`
  - drag movement should produce fewer or no `[resampler-skip]` entries
- Confirmed human-in-loop result:
  - EEG and EMG vertical drag both work
  - mouse drag panning is noticeably faster
  - custom drag final refresh logs now show `source=custom-drag`
  - custom drag final coalescing dropped from the normal roughly `450 ms` idle wait to roughly `2-12 ms`
  - custom drag final browser-total samples were roughly `304-400 ms`, compared with roughly `745-853 ms` for the normal Plotly/coalesced final path in the same run
  - final trace refresh still settles correctly after release
- Current conclusion:
  - Python/resampler callback time is no longer the navigation bottleneck
  - active movement is fastest when it stays client-side and avoids trace patch redraws
  - custom mouse drag is now close to keyboard panning because it bypasses native Plotly drag/coalescer churn and only requests one final detail refresh on release

## 2026-05-20

### Fast/Final Navigation Resampler Prototype

- Added a progressive navigation update prototype:
  - browser-side relayout coalescer now emits `mode="fast"` during active movement and `mode="final"` after release or idle
  - `fig_resampler_fast` is built and cached alongside the normal `fig_resampler`
  - fast transient updates use `512` shown samples for EEG, EMG, and NE
  - final updates keep the normal user-facing detail level
  - profiling logs now include update mode, cache key, and cache retrieval time
- Intended profiling signal:
  - active movement should show `mode=fast` and `cache_key=fig_resampler_fast`
  - idle/release correction should show `mode=final` and `cache_key=fig_resampler`
  - fast payload should be meaningfully smaller than the previous roughly 180 KB full-detail payload
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py app_src\make_figure_dev.py app_src\config.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_fft.py tests\test_smoke.py -q`
  - ran a synthetic `create_fig` check confirming both normal and fast resamplers are cached
  - ran a synthetic fast-mode `update_fig_resampler` check successfully
  - confirmed Dash serves `/assets/graphRelayoutCoalescer.js` with the new mode support
- Manual profiling after coalescer adjustment showed the fast path was working but still dominated by filesystem cache retrieval:
  - fast payload dropped to roughly 55-56 KB
  - fast callback totals were still often 300-500 ms while `cache_get` was roughly 230-680 ms

### In-Memory Resampler Storage Prototype

- Moved large Plotly-resampler objects out of Flask filesystem cache for the active desktop app path:
  - `app_src/app_dev.py` now stores `fig_resampler` and `fig_resampler_fast` in module-level process memory
  - opening a new file clears the in-memory resampler slots
  - relayout callbacks read the in-memory slots instead of deserializing from filesystem cache
  - profiling now reports `resampler_get` instead of `cache_get`
- Intended profiling signal:
  - `resampler_get` should be near zero compared with the previous 200-600 ms filesystem-cache retrieval cost
  - fast updates should keep the roughly 55 KB payload from the fast/final prototype
- Manual profiling after moving resamplers into process memory showed a large improvement:
  - `resampler_get=0.0 ms` across sampled fast and final updates
  - fast transient updates were roughly 11-15 ms total with a 55-56 KB payload
  - final full-detail updates were roughly 20-30 ms total with a 180-185 KB payload
  - before this change, the same fast/final path was roughly 300-500 ms because of filesystem cache retrieval
  - compared with the earlier full-detail path, active movement updates are now about 5-8x faster by callback time and about 3x smaller by payload
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_smoke.py -q`
  - ran a synthetic in-memory create/update/clear check successfully

### Visualization-Only EEG/EMG Downsampling Reverted

- Tested a first-pass display-only downsampling experiment for the active desktop visualization path, then reverted it:
  - EEG and EMG line traces were anti-aliased to 128 Hz with `scipy.signal.resample_poly`
  - raw EEG/EMG arrays remained the source for spectrogram generation, prediction, annotation saving, and file output
  - profiling did not show a meaningful interaction improvement
  - callback construction stayed around 55-80 ms
  - payload increased to roughly 200 KB because `x1` still displayed the same number of points and x-array serialization grew
- Updated `next_steps.md` to mark 128 Hz source downsampling as a tested dead end and move the next experiment back to "coarse while moving, detailed after idle/release."
- Verification:
  - during the experiment, ran a synthetic helper check confirming 512 Hz input was reduced to 128 Hz
  - during the experiment, ran a synthetic `make_figure` and `construct_update_data_patch` check successfully
  - after revert, ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\make_figure_dev.py app_src\config.py`
  - after revert, ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_fft.py tests\test_smoke.py -q`

### Relayout Coalescing Prototype

- Added browser-side coalescing for visualization trace updates:
  - `app_src/assets/graphRelayoutCoalescer.js` listens for Plotly `plotly_relayouting` and `plotly_relayout` events and emits one `sleepgraphrelayout` custom event after a short debounce or max wait
  - keyboard left/right panning now sends its requested x-range into the same coalescer instead of writing directly to `graph.relayoutData`
  - `app_src/components_dev.py` captures the coalesced custom event with `graph-relayout-coalesced`
  - `app_src/app_dev.py` now runs the Plotly-resampler update callback from the coalesced event instead of raw `graph.relayoutData`
- Intended behavior:
  - browser axes can still move immediately during pan/zoom
  - resampled EEG/EMG/NE trace payloads update from the latest coalesced x-range instead of every intermediate relayout event
  - repeated arrow-key panning is also rate-limited through the same path
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_fft.py tests\test_smoke.py -q`
  - imported `app_src.app_dev` successfully
  - confirmed Dash serves `/assets/graphRelayoutCoalescer.js` with HTTP 200
  - `node --check` could not run in that desktop agent environment because the bundled `node.exe` returned Access denied

### Navigation Profiling And Payload Reduction

- Added env-gated Plotly-resampler profiling in `app_src/app_dev.py`:
  - enabled with `SLEEP_SCORING_PROFILE_RESAMPLER=1`
  - logs update id, overlap/idle timing, callback construction time, payload encoding time, total callback time, payload size, x-range width, and a trace/property payload breakdown
- Measured pan/zoom updates on a user-style file:
  - original default `x1` updates were roughly 300 KB with 22 patch operations
  - callbacks usually did not backlog, but fast interactions could leave very small idle gaps or occasional overlap
  - payload was dominated by resampled EEG/EMG/NE arrays and theta/delta updates
- Reduced default-density update payload without making EEG/EMG sparse:
  - removed point markers from EEG, EMG, and NE traces while keeping black signal lines
  - removed duplicate theta/delta `customdata`
  - made theta/delta a static full-resolution trace so it no longer updates on every relayout
  - capped NE relayout updates at 1024 samples while keeping EEG/EMG at the default 2048
  - kept optional `x0.5` sampling level for fast navigation testing, but left `x1` as the default
- Observed profiling improvement:
  - default `x1` update payload dropped from about 308 KB to about 182 KB
  - patch operations dropped from 22 to 9
  - NE updates dropped from about 38 KB per x/y field to about 18-19 KB
- Current conclusion:
  - the lighter payload is real, but the subjective improvement may still be modest
  - remaining cost is mostly EEG/EMG x/y updates
  - next deeper optimization should consider debounced/coalesced relayout updates during active pan/zoom rather than reducing EEG/EMG detail by default

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\get_fft_plots.py app_src\make_figure_dev.py app_src\app_dev.py app_src\components_dev.py app_src\config.py`
- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests\test_fft.py tests\test_smoke.py -q`

## 2026-05-05

### Right-click Bout Selection Pilot

- Added a right-click bout-selection experiment to the active desktop app path:
  - `app_src/assets/graphContextMenu.js` captures graph `contextmenu` events and forwards Plotly x/y axis details to Dash
  - `app_src/components_dev.py` adds a `graph-contextmenu` event listener
  - `app_src/app_dev.py` handles the custom event in annotation/select mode and expands the selection to the contiguous scored or unscored bout around the clicked second
- Kept existing left-click thin-box selection and drag-box selection untouched.
- Right-click selection stores the same relative `[start, end]` range used by normal annotation and clip generation.
- Verification:
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py`
  - imported `app_src.app_dev` successfully
  - confirmed Dash serves `/assets/graphContextMenu.js` with HTTP 200
  - ran `node --check app_src\assets\graphContextMenu.js`

