# Work Log

Prepend new session notes to the top of this file.

Historical verification commands may include absolute paths from the original
development machine. When replaying or adapting them, keep the project folder
name `sleep_scoring` and conda environment name `sleep_scoring_dash3.0`, but
replace the user/home prefix and clone location with the collaborator's local
setup.

Reading note for agents: this file is prepended each session and can become
long. Start with only the two most recent dated entries, then search older
entries with targeted terms if the task needs deeper history.

## 2026-05-23

### Annotation Auto-Pan Bounds Clamp

- Followed up manual validation logs from the repaired integration branch:
  - normal navigation stayed on the optimized direct-restyle path
  - auto-pan direct resampler requests remained around `6-15 ms` server time
  - browser-side auto-pan merge buffers stayed bounded around `7k-8k` points
- Fixed the remaining edge behavior seen in the logs:
  - auto-pan lead/trim/replace requests now clamp to recording bounds
  - the direct resample endpoint also clamps requested ranges before constructing patches
  - the graph figure stores recording x-bounds in layout meta for browser-side clamping
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

## 2026-04-30

### Additional Notes

- Revisited the double-click idea after the last working `B`-armed experiment had been documented, but treated it as a pilot only and rolled all app-code experiments back at the end of the session.
- Tried a staged click-classifier approach in [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py):
  - first routed raw `graph.clickData` through a separate `check_click` path
  - piloted timer-based single-vs-double classification
  - confirmed that raw single clicks in annotation mode arrive reliably
  - did not get reliable double-click classification from sequential `clickData`
- Tried multiple browser-side double-click probes:
  - `dcc.Interval`-based wait-out logic
  - `setTimeout(...)`-based wait-out logic
  - native DOM `dblclick` listeners from `assets/`
  - Plotly `plotly_doubleclick`
  - `config["doubleClick"] = False` to disable Plotly's built-in double-click reset behavior
- What those pilots showed:
  - single clicks can be delayed and classified in clientside code
  - double clicks in graph whitespace can sometimes be seen by native listeners
  - double clicks inside actual subplots still do not reach the app in a useful, reliable way
  - disabling Plotly's built-in double-click behavior was not enough to make subplot double-click practical here
- Current conclusion:
  - a polished double-click interaction for subplot clicks is not impossible in theory
  - but it is not practical with the current Dash/Plotly event surfaces we tested
  - making it work would likely require a lower-level custom JS interaction layer on Plotly's internal drag/pointer surfaces
- Ended the session by restoring the app files to the last committed baseline.
- Important clarification:
  - the `B`-armed bout-select interaction was a documented working experiment
  - it is not the current committed app code
  - the current committed baseline is the nonzero-`start_time` click-selection fix plus the normal thin-box click and box-drag selection behavior

### Done Today

- Fixed a real annotation indexing bug in [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py):
  - `read_click_select` had been storing absolute plot times from single-click selection
  - `make_annotation` interprets the stored range as sleep-score indices
  - this only behaved correctly when `start_time == 0`
  - the fix was to store click-selected ranges relative to `metadata.start_time`, matching the existing box-select path
- Verified that bug on a nonzero-start `.mat` by:
  - applying the fix
  - reverting it temporarily so the old behavior could be reproduced
  - reapplying the fix after confirmation
- Committed and pushed the nonzero-start click-selection fix, then merged it to `dev` and `main`.
- Explored a new "select existing bout/unscored section" annotation workflow in the active app code:
  - focused on [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py)
  - used annotation mode as the feature boundary
  - preserved normal single-click thin-box selection and drag-box selection as the baseline behavior
- Landed on a working interaction for now:
  - press `M` to enter annotation mode
  - press `B` to arm a one-shot bout-select action
  - the next click expands to the full contiguous scored or unscored span around the clicked second
  - after that click, the app automatically returns to normal click selection
- Reverted a later refactor attempt and intentionally left the code at the earlier working `B`-armed version.

### What We Tried

- Double-click selection:
  - attractive from a UX perspective
  - not reliable in this Dash/Plotly setup
  - regular `clickData` timing heuristics were flaky
  - Plotly's own double-click behavior also competes with app-level logic
- DOM/EventListener-based double-click detection:
  - looked promising in theory
  - did not reliably see the graph-surface interaction we needed
- `Ctrl`/`Cmd` + click:
  - also looked promising in theory
  - modifier state and graph click state were not trustworthy enough together in this app
  - one iteration also interfered with existing keyboard-driven mode switching
- Segment-store refactor:
  - introduced `sleep-segments-store` and a separate `read_bout_select` path
  - conceptually cleaner for future features
  - the refactor led to callback/debug-mode problems and was backed out

### What Worked

- The nonzero-`start_time` fix for normal click selection is good and should stay.
- The simplest reliable select-existing-bout workflow so far is the explicit one-shot mode:
  - `M` enters annotation mode
  - `B` arms bout selection
  - one click selects the whole contiguous scored or unscored segment
  - the armed state resets immediately after use
- Keeping the bout-selection logic inside the existing `read_click_select` path was more reliable than trying to split it into multiple overlapping click callbacks.

### What Did Not Work Well

- Gesture inference in this app is harder than it first appears.
- Double-click and modifier-click ideas both ran into integration problems with Dash/Plotly event flow.
- The separate `sleep-segments-store` / `read_bout_select` refactor is not ready to trust yet.
- Debug-mode callback errors during that refactor were a sign to stop and return to the known-good baseline instead of continuing to pile changes on.

### Current Baseline

- The current committed app code now has:
  - the nonzero-`start_time` fix for click selection
  - normal single click thin-box selection
  - box drag selection
- The `B`-armed one-shot bout selection should be treated as a previously working experiment, not as the current committed baseline.
- If the faster segment-store architecture or explicit bout-select mode is revisited later, start from the current committed baseline and reintroduce the experimental pieces incrementally with debug mode on.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\app_dev.py app_src\components_dev.py` after the final rollback to the committed baseline.

## 2026-04-23

### Done Today

- Integrated the statistical Wake/REM model into the app as an alternative inference path:
  - created `app_src/run_inference_stats_model.py` from the visualization prototype and trimmed it into an app-shaped module
  - separated prediction from visualization with `predict_stats_model(...)`, `infer(...)`, and a developer-only `make_figure(...)`
  - moved the stats model out of `scripts/` and into `app_src/`
- Renamed the statistical model parameters to be easier to remember:
  - `spectrogram_sleep_wave_range`
  - `spectrogram_normalization_range`
  - `min_wake_duration`
  - `wake_merge_coefficient`
  - `min_rem_duration`
  - `rem_threshold_percentile`
  - `rem_threshold_comparison_percentile`
  - `ne_smoothing_window`
- Exposed only the intended user-facing stats-model controls in `app_src/config.py`:
  - `STATS_MODEL_WAKE_THRESHOLD`
  - `STATS_MODEL_MIN_WAKE_DURATION`
  - `STATS_MODEL_MIN_REM_DURATION`
- Added a config-level app model selector in `app_src/config.py`:
  - `SLEEP_SCORING_MODEL = "sdreamer"` or `"stats_model"`
  - the UI was left unchanged; selection is config-only for now
- Updated `app_src/inference.py` so:
  - `sdreamer` keeps the existing learned-model path
  - `stats_model` uses the new statistical inference path
  - legacy app postprocessing only applies to `sdreamer`
- Matched the app stats model to the validated `shape_test="none"` comparison behavior:
  - removed the REM shape gate from the new pipeline
  - left a short code comment explaining that this is intentional
- Verified that the app stats model matches the old visualization pipeline on `35_app13.mat` when compared against the `shape_test="none"` configuration.

### Next Steps

- Improve REM detection so it can relabel a subsection of a long Wake bout as REM instead of promoting the entire Wake bout to REM.
- Keep checking whether that subsection logic should happen:
  - before Wake-to-REM relabeling finishes
  - or as a follow-up split step after an initial REM candidate is found

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\config.py app_src\inference.py app_src\run_inference_stats_model.py`
- Compared `app_src/run_inference_stats_model.py` against `scripts/visualize_low_band_wake_bouts.py` on `user_test_files\35_app13.mat` and confirmed matching Wake/REM results when the old visualization uses `shape_test='none'`.

## 2026-04-22

### Done Today

- Changed the `thirds` REM shape test in `scripts/visualize_low_band_wake_bouts.py` from third medians to third means:
  - compares middle-third mean against left- and right-third means
  - makes `thirds` the default REM shape test after strict chord proved too restrictive and no shape test admitted too many imposters
  - keeps `--rem-shape-test chord`, `thirds`, and `none` available

### Next Steps

- Tidy the experimental algorithm before app integration:
  - rename tuning parameters so user-facing controls describe behavior rather than implementation details
  - decide which parameters should be exposed to users and which should remain developer/debug settings
  - integrate the Wake/REM/post-REM recovery logic into the app prediction path
  - ship an early test build so users can visually review the statistical model on real recordings

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with `--rem-shape-test thirds` and confirmed it writes an HTML plot.

## 2026-04-22

### Done Today

- Added `--rem-shape-test none` to `scripts/visualize_low_band_wake_bouts.py`:
  - skips the REM convex/chord/shape gate
  - keeps duration and global low-NE criteria active
  - supports quick diagnosis of whether candidate REM bouts are being rejected by shape testing

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with `--rem-shape-test none` and confirmed it writes an HTML plot.

## 2026-04-22

### Done Today

- Added a strict chord-based REM NE shape test to `scripts/visualize_low_band_wake_bouts.py`:
  - connects the first and last finite NE point in a candidate bout with a straight line
  - requires every finite interior NE point to sit on or below that line
  - makes chord the default REM shape test
  - keeps the previous thirds-median test available with `--rem-shape-test thirds`

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with default chord shape testing and confirmed it writes an HTML plot.

## 2026-04-22

### Done Today

- Added post-REM NE recovery splitting to `scripts/visualize_low_band_wake_bouts.py`:
  - after REM detection, each REM bout computes cumulative NE diff across the bout
  - after the NE trough, the first cumulative diff crossing above a small epsilon becomes the REM/Wake split
  - the pre-split segment remains REM and the recovery tail is added back as Wake
  - new CLI option is `--rem-recovery-epsilon-fraction`, defaulting to `0.02`

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with `--rem-recovery-epsilon-fraction 0.02` and confirmed it writes an HTML plot.

## 2026-04-22

### Done Today

- Changed REM low-NE candidate scoring in `scripts/visualize_low_band_wake_bouts.py` from `min`/`median` choices to a numeric bout percentile:
  - new CLI option is `--rem-low-ne-percentile`
  - `0` matches previous min-like behavior and `50` matches median-like behavior
  - global NE thresholding remains controlled separately by `--rem-global-low-percentile`

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with `--rem-low-ne-percentile 50` and confirmed it writes an HTML plot.

## 2026-04-21

### Done Today

- Replaced fixed Wake-bout gap merging in `scripts/visualize_low_band_wake_bouts.py` with a one-pass relative NREM-gap merge:
  - converts non-Wake/NREM gaps to Wake when gap duration is less than `nrem_gap_merge_ratio` times the sum of neighboring Wake durations
  - uses one Wake neighbor for edge gaps and two Wake neighbors for interior gaps
  - supports optional `max_nrem_gap_s`, defaulting to `None`
  - keeps short-Wake removal as a separate final cleanup step
- Updated CLI/direct-run parameters from `merge_gap_s` to `nrem_gap_merge_ratio` and `max_nrem_gap_s`.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran small synthetic checks confirming sum-neighbor merging and one-pass non-cascading behavior.

## 2026-04-21

### Done Today

- Extended `scripts/visualize_low_band_wake_bouts.py` from a Wake-only visualization into a Wake/REM experiment view:
  - keeps Wake detection modular with postprocessing toggles for short-bout removal and gap merging
  - adds optional NE-based Wake-to-REM relabeling controlled by `--no-rem-detection`
  - starts REM detection from Wake bouts at least 30 seconds long
  - supports global NE low-value tests by either `min` or `median`, controlled by `--rem-low-ne-stat`
  - supports a centered global moving average on NE before REM detection, controlled by `--rem-smoothing-window-s`
  - uses a simple thirds-based convexity check to require a valley-like NE shape
  - simplifies default output filenames to focus on the REM experiment instead of every Wake parameter
- Added the theta/delta ratio line back to the Wake/REM spectrogram subplot.
- Updated the shared theta/delta trace styling in `app_src/get_fft_plots.py` to use a black line at 50% opacity.
- Tried Plotly y-gridline overlays for the spectrogram, but reverted those changes after visual inspection showed they still did not draw above the heatmap reliably.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile app_src\get_fft_plots.py scripts\visualize_low_band_wake_bouts.py`
- Ran the Wake/REM visualization script on `user_test_files\115_gs.mat` with the current smoothed-min REM settings and confirmed it writes an HTML plot.
- Compared exploratory REM settings on `user_test_files\115_gs.mat`: unsmoothed min, median, and smoothed min.

## 2026-04-17

### Done Today

- Created branch `codex/statistical_model` from `origin/dev` to explore a simple statistical Wake-detection approach.
- Added `scripts/plot_low_band_spectrogram_distribution.py`:
  - loads a `.mat` file
  - reuses `app_src.get_fft_plots.get_fft_plots` so the feature is derived from the same smoothed dB spectrogram shown in the app
  - globally min-max normalizes the displayed spectrogram values to approximate the Viridis/Plotly visual normalization
  - averages the normalized `1-5 Hz` rows column-wise
  - writes an interactive HTML distribution plot
  - optionally splits the distribution by existing `sleep_scores` labels when they are present in the `.mat`
- Added `scripts/visualize_low_band_wake_bouts.py`:
  - thresholds the normalized `1-5 Hz` column mean into Wake-only predictions
  - writes a two-panel HTML visualization with EEG spectrogram and NE, matching the focused ChatGPT-style layout
  - overlays threshold-selected Wake predictions as an app-style sleep-score heatmap using the Wake color from `STAGE_COLORS`
  - enables scroll zoom in the generated Plotly HTML
  - includes the threshold and rule direction in the default output filename so multiple threshold experiments do not overwrite each other
- Initial visual inspection suggested `0.6` is a better threshold candidate than `0.8`.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\plot_low_band_spectrogram_distribution.py`
- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m py_compile scripts\visualize_low_band_wake_bouts.py`
- Ran the distribution script on `demo_data\COM5_bin1_gs.mat` and confirmed it wrote an HTML plot.
- Ran the Wake-bout visualization script on `demo_data\COM5_bin1_gs.mat` at multiple thresholds and confirmed:
  - Wake overlay color is `rgb(124, 124, 251)`
  - generated HTML includes `scrollZoom`
  - threshold-specific filenames are created
