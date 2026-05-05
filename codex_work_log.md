# Codex Work Log

Prepend new session notes to the top of this file.

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
