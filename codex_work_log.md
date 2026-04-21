# Codex Work Log

Prepend new session notes to the top of this file.

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
