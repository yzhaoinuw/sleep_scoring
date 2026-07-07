# Sleep Scoring Project Overview

## What This Repo Is

This repository contains a desktop sleep scoring application built around a Dash UI embedded inside a `pywebview` window. The current active user-facing path starts at [`run_desktop_app.py`](run_desktop_app.py), then flows into the maintained modules under [`app_src`](app_src).

The app is designed to:

- Load EEG/EMG `.mat` files, with optional NE and video metadata
- Visualize spectrogram of EEG, EEG, EMG, NE, and sleep scores together
- Support manual annotation with keyboard shortcuts
- Optionally run automatic scoring using the statistical model or the `sdreamer` model in [`models/sdreamer`](models/sdreamer)
- Export edited annotations back to `.mat`
- Export sleep bout and summary statistics to Excel when scoring is complete
- Extract and play a matching video clip for a selected time region

## Implementation Cookbook

For a deeper, feature-by-feature account of *how* the interactive visualization is
built — file loading, the resampler figure, caching, zoom/pan coalescing, custom
pointer pan, drag-select with auto-pan, keypress annotation, undo, and the
clientside/asset-JS interaction layer — see [`dash_app_cookbook.md`](dash_app_cookbook.md).
It is organized as modular recipes, each pointing at the exact reference-app files,
so it doubles as a map from a feature to its implementation.

## Active Runtime Path

### 1. Desktop entrypoint

[`run_desktop_app.py`](run_desktop_app.py)

- Detects whether the app is running from source or as a packaged executable
- Adds the repo/app base directory to `sys.path`
- Imports:
  - [`app_src/app.py`](app_src/app.py)
  - [`app_src/config.py`](app_src/config.py)
  - [`app_src/__init__.py`](app_src/__init__.py)
- Starts the Dash server in a background thread
- Opens the UI in a native `pywebview` window

### 2. Main app package

[`app_src/app.py`](app_src/app.py) is a thin aggregator: importing it builds the app by pulling in the modules below (their imports register the Flask routes and Dash callbacks), and it re-exports `app` for `run_desktop_app.py`.

- [`app_src/server.py`](app_src/server.py): the Dash app and top-level layout, the filesystem cache in the system temp directory, `TEMP_PATH`/`VIDEO_DIR`, and the `run_inference` availability probe
- [`app_src/routes.py`](app_src/routes.py): raw Flask endpoints for direct browser-side resampling (`/_sleep_scoring/resample`) and mirrored profiling logs (`/_sleep_scoring/profile-log`)
- [`app_src/dialogs.py`](app_src/dialogs.py): native pywebview file dialogs for `.mat`, video, and save destinations
- [`app_src/session.py`](app_src/session.py): per-recording setup — cache initialization, temp-dir housekeeping, `.mat` metadata extraction, figure creation
- [`app_src/resampling.py`](app_src/resampling.py): the live resampler-figure store plus patch/profiling helpers shared by the resample route and the navigation callbacks
- [`app_src/callbacks/`](app_src/callbacks): Dash callbacks, one module per concern — `clientside` (registrations for the in-browser callbacks; their JavaScript lives in [`app_src/assets/clientsideCallbacks.js`](app_src/assets/clientsideCallbacks.js)), `loading`, `navigation`, `prediction`, `saving`, `video`

Together these cover: validating selected `.mat` contents, building the interactive Plotly figure, tracking annotation history for undo and crash recovery, and handling prediction requests, saving, and video clip preparation.

Important active imports:

- [`app_src/components.py`](app_src/components.py)
- [`app_src/make_figure.py`](app_src/make_figure.py)
- [`app_src/inference.py`](app_src/inference.py) if Torch is available
- [`app_src/postprocessing.py`](app_src/postprocessing.py)
- [`app_src/make_mp4.py`](app_src/make_mp4.py)

### 3. UI component layer

[`app_src/components.py`](app_src/components.py)

- Defines the home screen button to choose a `.mat` file
- Defines hidden stores used by callbacks
- Defines the graph, modals, save/undo controls, video controls, and prediction modal
- Uses `dash_extensions.EventListener` to capture keyboard input for annotation and mode switching

### 4. Figure generation

[`app_src/make_figure.py`](app_src/make_figure.py)

- Pads or initializes `sleep_scores` to match recording duration
- Builds a four-row Plotly layout:
  - EEG spectrogram + theta/delta ratio
  - EEG trace
  - EMG trace
  - NE trace
- Overlays sleep score heatmaps on EEG, EMG, and NE rows
- Uses `plotly-resampler` for responsive navigation on large signals
- Pulls FFT/spectrogram content from [`app_src/get_fft_plots.py`](app_src/get_fft_plots.py)

[`app_src/get_fft_plots.py`](app_src/get_fft_plots.py)

- Computes a spectrogram using `scipy.signal.ShortTimeFFT`
- Restricts display to 0-30 Hz
- Computes the theta/delta ratio line
- Applies configurable smoothing and colors from [`app_src/config.py`](app_src/config.py)

### 5. Prediction path

[`app_src/inference.py`](app_src/inference.py)

- Selects the inference backend from `SLEEP_SCORING_MODEL` in [`app_src/config.py`](app_src/config.py)
- Writes `sleep_scores` and `confidence` back into the in-memory `mat`
- Optionally postprocesses predictions

Routing:

- Statistical Wake/REM model: [`app_src/run_inference_stats_model.py`](app_src/run_inference_stats_model.py)
- sDREAMER with NE present: [`app_src/run_inference_ne.py`](app_src/run_inference_ne.py)
- sDREAMER without NE: [`app_src/run_inference_sdreamer.py`](app_src/run_inference_sdreamer.py)

Shared preprocessing:

- [`app_src/preprocessing.py`](app_src/preprocessing.py)

What preprocessing does:

- Trims missing labels at sequence edges
- Standardizes channels if requested
- Resamples EEG/EMG to 512 Hz when needed
- Reshapes signals into per-second model inputs
- Builds NE windows at 10 Hz for the NE-aware model

### 6. Model files

[`models/sdreamer`](models/sdreamer)

Relevant files:

- [`models/sdreamer/n2nSeqNewMoE2.py`](models/sdreamer/n2nSeqNewMoE2.py): EEG/EMG-only model
- [`models/sdreamer/n2nBaseLineNE.py`](models/sdreamer/n2nBaseLineNE.py): EEG/EMG/NE model
- [`models/sdreamer/checkpoints`](models/sdreamer/checkpoints): checkpoint files expected by the inference scripts
- [`models/sdreamer/layers`](models/sdreamer/layers): transformer and patch-encoding submodules

This model stack is active but optional. The statistical model runs without Torch; sDREAMER requires the optional Torch runtime.

### 7. Postprocessing and export

[`app_src/postprocessing.py`](app_src/postprocessing.py)

- Converts dense sleep score arrays into contiguous bout tables
- Applies heuristic cleanup rules to predictions
- Derives summary stats and transition counts
- Standardizes NE before saving if present

Heuristics include:

- Relabeling short Wake segments when EMG evidence is weak
- Removing short SWS between Wake segments
- Removing short REM bouts
- Validating REM transitions, optionally using NE

### 8. Video support

[`app_src/make_mp4.py`](app_src/make_mp4.py)

- Uses the `imageio-ffmpeg` bundled executable
- Cuts a selected `[start, end]` time range into an `.mp4`
- Stores clips under [`app_src/assets/videos`](app_src/assets/videos)

Within the app:

- The selected time window comes from the selection callbacks in [`app_src/callbacks/clientside.py`](app_src/callbacks/clientside.py); the clip callbacks live in [`app_src/callbacks/video.py`](app_src/callbacks/video.py)
- The clip is shown in a Dash modal using `dash_player`

## User Data Expectations

The current app expects MATLAB `.mat` files with:

Required fields:

- `eeg`
- `eeg_frequency`
- `emg`

Optional fields:

- `ne`
- `ne_frequency`
- `sleep_scores`
- `start_time`
- `video_name`
- `video_path`
- `video_start_time`

The README is still the best end-user description of expected inputs and runtime setup.

## Sample Data and Testing Assets

### `user_test_files/`

[`user_test_files`](user_test_files)

This is the main sandbox for trying the app and for understanding data variants. It contains:

- Rawish `.mat` files for manual testing
- Gold-standard / predicted variants such as:
  - `*_gs.mat`
  - `*_sdreamer_3class.mat`
  - `*_sdreamer_ne_3class.mat`
  - `*_post.mat`
- Example source videos (`.avi`)
- Subfolders with multi-bin or user-specific examples
- An `archive/` subfolder with older test artifacts

Good representative files to start with:

- [`user_test_files/115_gs.mat`](user_test_files/115_gs.mat)
- [`user_test_files/F268_FP-Data.mat`](user_test_files/F268_FP-Data.mat)
- [`user_test_files/788_bin1_gs.mat`](user_test_files/788_bin1_gs.mat)

### `tests/`

[`tests`](tests)

Current tests focus on the active modules:

- smoke/import coverage
- preprocessing behavior
- postprocessing behavior
- FFT helper behavior
- app helper functions (`session.py`, `resampling.py`, and callbacks in `callbacks/saving.py` / `callbacks/video.py`)
- clientside callback behavior (`tests/js/`, a jest suite for `app_src/assets/clientsideCallbacks.js`; requires Node.js, run with `cd tests/js && npm ci && npm test`)

These tests reinforce that the `app_src` app package, `components.py`, `make_figure.py`, `preprocessing.py`, and `postprocessing.py` are the main maintained path.

## Repo Structure Map

High-level map:

```text
sleep_scoring/
|- run_desktop_app.py                # active desktop entrypoint
|- README.md                         # user/developer setup and usage notes
|- pyproject.toml                    # package metadata and dependencies
|- app_src/
|  |- __init__.py                    # version
|  |- config.py                      # UI / FFT / inference config
|  |- app.py                         # thin aggregator (importing it registers everything)
|  |- server.py                      # Dash instance, cache, components, runtime paths
|  |- routes.py                      # raw Flask endpoints (resample, profile-log)
|  |- dialogs.py                     # native pywebview file dialogs
|  |- session.py                     # per-recording setup helpers
|  |- resampling.py                  # resampler figure store + patch helpers
|  |- callbacks/                     # Dash callbacks, one module per concern
|  |  |- clientside.py               # clientside registrations (JS in assets/)
|  |  |- loading.py / navigation.py / prediction.py / saving.py / video.py
|  |- components.py                  # active UI components
|  |- make_figure.py                 # active figure builder
|  |- get_fft_plots.py               # spectrogram + theta/delta helper
|  |- inference.py                   # inference router
|  |- run_inference_stats_model.py   # statistical Wake/REM model runner
|  |- run_inference_sdreamer.py      # EEG/EMG model runner
|  |- run_inference_ne.py            # EEG/EMG/NE model runner
|  |- preprocessing.py               # model input preparation
|  |- postprocessing.py              # heuristic cleanup + stats
|  |- make_mp4.py                    # video clip extraction
|  |- assets/
|  |  |- clientsideCallbacks.js      # clientside callback implementations
|  |  |- graph*.js / annotationAutoPan.js / closeWindow.js  # interaction scripts
|  |  |- videos/
|- models/
|  |- sdreamer/
|  |  |- n2nSeqNewMoE2.py
|  |  |- n2nBaseLineNE.py
|  |  |- layers/
|  |  |- checkpoints/
|- user_test_files/                  # sample MAT inputs and example videos
|- tests/                            # pytest suite for active modules
|- packaging/
|  |- windows/
|  |  |- app.spec                    # active Windows PyInstaller spec
|  |  |- make_full_app_zip.ps1       # full Windows app zip builder
|  |  |- make_app_update_zip.ps1     # app_src-only update zip builder
|  |  |- smoke_check_release.ps1     # release folder checks
|  |  |- release_helpers/            # files copied into full app zips
|- demo_data/                        # small demo example(s)
|- archive/                          # older model/app code and checkpoints
|- msda_version1.1/                  # older model lineage
|- build/                            # generated packaging/test staging output
|- dist/                             # generated PyInstaller app folders
|- release_artifacts/                # generated release zips and sidecars
|- environment.yml / requirements.txt
```

## What Looks Active vs. Legacy

### Active / relevant now

- [`run_desktop_app.py`](run_desktop_app.py)
- [`app_src/app.py`](app_src/app.py), [`app_src/server.py`](app_src/server.py), [`app_src/routes.py`](app_src/routes.py)
- [`app_src/dialogs.py`](app_src/dialogs.py), [`app_src/session.py`](app_src/session.py), [`app_src/resampling.py`](app_src/resampling.py)
- [`app_src/callbacks/`](app_src/callbacks)
- [`app_src/components.py`](app_src/components.py)
- [`app_src/make_figure.py`](app_src/make_figure.py)
- [`app_src/get_fft_plots.py`](app_src/get_fft_plots.py)
- [`app_src/preprocessing.py`](app_src/preprocessing.py)
- [`app_src/inference.py`](app_src/inference.py)
- [`app_src/run_inference_stats_model.py`](app_src/run_inference_stats_model.py)
- [`app_src/run_inference_sdreamer.py`](app_src/run_inference_sdreamer.py)
- [`app_src/run_inference_ne.py`](app_src/run_inference_ne.py)
- [`app_src/postprocessing.py`](app_src/postprocessing.py)
- [`app_src/make_mp4.py`](app_src/make_mp4.py)
- [`models/sdreamer`](models/sdreamer)
- [`packaging/windows`](packaging/windows)
- [`user_test_files`](user_test_files)
- [`tests`](tests)

### Likely older or secondary

- [`archive`](archive)
- [`msda_version1.1`](msda_version1.1)
- [`build`](build)
- [`dist`](dist)
- [`release_artifacts`](release_artifacts)

## Practical Mental Model

If you only want to understand the current product, read files in this order:

1. [`README.md`](README.md)
2. [`run_desktop_app.py`](run_desktop_app.py)
3. [`app_src/server.py`](app_src/server.py) (then [`app_src/app.py`](app_src/app.py) to see how the pieces assemble)
4. [`app_src/components.py`](app_src/components.py)
5. [`app_src/callbacks/`](app_src/callbacks) — pick the module for the concern you care about
6. [`app_src/make_figure.py`](app_src/make_figure.py)
7. [`app_src/get_fft_plots.py`](app_src/get_fft_plots.py)
8. [`app_src/inference.py`](app_src/inference.py)
9. [`app_src/preprocessing.py`](app_src/preprocessing.py)
10. [`app_src/postprocessing.py`](app_src/postprocessing.py) and [`app_src/make_mp4.py`](app_src/make_mp4.py)

## Questions Worth Clarifying Later

These are not blockers, just places where your intent would be useful if we keep documenting or refactoring:

- Which sample `.mat` files you consider the best canonical fixtures
- Whether `archive/` and `msda_version1.1/` should be documented as historical research lineage
