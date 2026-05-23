# Codex Work Log

Prepend new session notes to the top of this file.

Rotation policy: keep the current month's entries here; rotate older entries to [`codex_work_log_archive.md`](codex_work_log_archive.md) at the start of each month (or whenever this file grows past roughly 200-300 lines). Last rotation: 2026-05-23.

## 2026-05-23

### Annotation Drag Auto-Pan

- Added browser-side auto-pan for annotation drag selection:
  - dragging a selection beyond the left or right graph edge now pans the x-axis while preserving the selection range
  - the final selected range is sent through the existing annotation selection flow on mouse release
  - normal relayout coalescing is suppressed while annotation auto-pan is active so the live selection does not compete with regular navigation callbacks
- Added direct trace refreshes for auto-pan:
  - `/_sleep_scoring/resample` returns Plotly-resampler patches for browser-side refreshes
  - auto-pan requests a lead window in the drag direction and merges returned x/y data into the active graph
  - release uses a final replace refresh for the exact visible range
- Tuned the merge path after profiling:
  - bounded the browser-side merge buffer to prevent point-count growth during long auto-pan drags
  - long manual drags now stayed around 7k-8k active points instead of climbing into the 20k+ range
  - direct resampler server work was usually about 13-16 ms, while browser merge apply was usually about 230-305 ms
- Added response-time logging for the current profiling pass:
  - server logs include `[resampler]` for normal Dash updates and `[resampler-direct]` for direct auto-pan refreshes
  - browser logs include `[browser-relayout]` and `[browser-autopan]` via `/_sleep_scoring/profile-log`
  - profiling is enabled by default through `PROFILE_RESAMPLER_UPDATES`; set `SLEEP_SCORING_PROFILE_RESAMPLER=0` to quiet it
- Known follow-up:
  - dragging briefly past the recording end can request a lead window beyond available data
  - this may produce a momentary straight-line trace before the final replace refresh recovers
  - clamp the lead request to recording bounds in the next polish pass
- Verification:
  - manually tested long-distance annotation auto-pan without observed selection failures
  - parsed the changed browser assets through dynamic import in the Codex Node REPL
  - ran `git diff --check`
  - ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests -q --basetemp .pytest_tmp\response_logging_buffer_cap -p no:cacheprovider`

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
  - `node --check` could not run in this Codex desktop environment because the bundled `node.exe` returned Access denied

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

