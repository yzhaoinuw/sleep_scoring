# Codex Work Log

Prepend new session notes to the top of this file.

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
