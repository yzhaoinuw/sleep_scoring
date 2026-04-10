# Sleep Scoring Project Overview

## What This Repo Is

This repository contains a desktop sleep scoring application built around a Dash UI embedded inside a `pywebview` window. The current active user-facing path starts at [`run_desktop_app.py`](C:\Users\yzhao\python_projects\sleep_scoring\run_desktop_app.py), then flows into the newer `*_dev` modules under [`app_src`](C:\Users\yzhao\python_projects\sleep_scoring\app_src).

The app is designed to:

- Load EEG/EMG `.mat` files, with optional NE and video metadata
- Visualize spectrogram of EEG, EEG, EMG, NE, and sleep scores together
- Support manual annotation with keyboard shortcuts
- Optionally run automatic scoring using the `sdreamer` model in [`models/sdreamer`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer)
- Export edited annotations back to `.mat`
- Export sleep bout and summary statistics to Excel when scoring is complete
- Extract and play a matching video clip for a selected time region

## Active Runtime Path

### 1. Desktop entrypoint

[`run_desktop_app.py`](C:\Users\yzhao\python_projects\sleep_scoring\run_desktop_app.py)

- Detects whether the app is running from source or as a packaged executable
- Adds the repo/app base directory to `sys.path`
- Imports:
  - [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py)
  - [`app_src/config.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\config.py)
  - [`app_src/__init__.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\__init__.py)
- Starts the Dash server in a background thread
- Opens the UI in a native `pywebview` window

### 2. Main app module

[`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py) is the main active application.

Key responsibilities:

- Creates the Dash app and top-level layout
- Manages a filesystem cache in the system temp directory
- Opens native file dialogs for `.mat`, `.avi`, `.mp4`, and save destinations
- Validates selected `.mat` contents
- Builds the interactive Plotly figure
- Tracks annotation history for undo and crash recovery
- Handles prediction requests, saving, and video clip preparation

Important active imports:

- [`app_src/components_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\components_dev.py)
- [`app_src/make_figure_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_figure_dev.py)
- [`app_src/inference.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\inference.py) if Torch is available
- [`app_src/postprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\postprocessing.py)
- [`app_src/make_mp4.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_mp4.py)

### 3. UI component layer

[`app_src/components_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\components_dev.py)

- Defines the home screen button to choose a `.mat` file
- Defines hidden stores used by callbacks
- Defines the graph, modals, save/undo controls, video controls, and prediction modal
- Uses `dash_extensions.EventListener` to capture keyboard input for annotation and mode switching

### 4. Figure generation

[`app_src/make_figure_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_figure_dev.py)

- Pads or initializes `sleep_scores` to match recording duration
- Builds a four-row Plotly layout:
  - EEG spectrogram + theta/delta ratio
  - EEG trace
  - EMG trace
  - NE trace
- Overlays sleep score heatmaps on EEG, EMG, and NE rows
- Uses `plotly-resampler` for responsive navigation on large signals
- Pulls FFT/spectrogram content from [`app_src/get_fft_plots.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\get_fft_plots.py)

[`app_src/get_fft_plots.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\get_fft_plots.py)

- Computes a spectrogram using `scipy.signal.ShortTimeFFT`
- Restricts display to 0-30 Hz
- Computes the theta/delta ratio line
- Applies configurable smoothing and colors from [`app_src/config.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\config.py)

### 5. Prediction path

[`app_src/inference.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\inference.py)

- Selects the inference backend based on whether NE exists in the `.mat`
- Writes `sleep_scores` and `confidence` back into the in-memory `mat`
- Optionally postprocesses predictions

Routing:

- With NE present: [`app_src/run_inference_ne.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\run_inference_ne.py)
- Without NE: [`app_src/run_inference_sdreamer.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\run_inference_sdreamer.py)

Shared preprocessing:

- [`app_src/preprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\preprocessing.py)

ChatGPT scoring scaffold notes:

- [`app_src/chatgpt_inference.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\chatgpt_inference.py) is the placeholder backend entrypoint for the future ChatGPT scoring path
- [`app_src/chatgpt_tools.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\chatgpt_tools.py) contains deterministic helper functions for snapshots, interval inspection, score lookup, and score editing
- [`app_src/chatgpt_scoring_guidance.md`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\chatgpt_scoring_guidance.md) is the first draft of the model guidance prompt, including the current transition rules

What preprocessing does:

- Trims missing labels at sequence edges
- Standardizes channels if requested
- Resamples EEG/EMG to 512 Hz when needed
- Reshapes signals into per-second model inputs
- Builds NE windows at 10 Hz for the NE-aware model

### 6. Model files

[`models/sdreamer`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer)

Relevant files:

- [`models/sdreamer/n2nSeqNewMoE2.py`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer\n2nSeqNewMoE2.py): EEG/EMG-only model
- [`models/sdreamer/n2nBaseLineNE.py`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer\n2nBaseLineNE.py): EEG/EMG/NE model
- [`models/sdreamer/checkpoints`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer\checkpoints): checkpoint files expected by the inference scripts
- [`models/sdreamer/layers`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer\layers): transformer and patch-encoding submodules

This model stack is active but optional. The app still runs without Torch; in that case the prediction button is disabled.

### 7. Postprocessing and export

[`app_src/postprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\postprocessing.py)

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

[`app_src/make_mp4.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_mp4.py)

- Uses the `imageio-ffmpeg` bundled executable
- Cuts a selected `[start, end]` time range into an `.mp4`
- Stores clips under [`app_src/assets/videos`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\assets\videos)

Within the app:

- The selected time window comes from annotation selection in `app_dev.py`
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

[`user_test_files`](C:\Users\yzhao\python_projects\sleep_scoring\user_test_files)

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

- [`user_test_files/115_gs.mat`](C:\Users\yzhao\python_projects\sleep_scoring\user_test_files\115_gs.mat)
- [`user_test_files/F268_FP-Data.mat`](C:\Users\yzhao\python_projects\sleep_scoring\user_test_files\F268_FP-Data.mat)
- [`user_test_files/788_bin1_gs.mat`](C:\Users\yzhao\python_projects\sleep_scoring\user_test_files\788_bin1_gs.mat)

### `tests/`

[`tests`](C:\Users\yzhao\python_projects\sleep_scoring\tests)

Current tests focus on the active modules:

- smoke/import coverage
- preprocessing behavior
- postprocessing behavior
- FFT helper behavior
- a few helper functions in `app_dev.py`

These tests reinforce that `app_dev.py`, `components_dev.py`, `make_figure_dev.py`, `preprocessing.py`, and `postprocessing.py` are the main maintained path.

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
|  |- app_dev.py                     # active Dash app
|  |- components_dev.py              # active UI components
|  |- make_figure_dev.py             # active figure builder
|  |- get_fft_plots.py               # spectrogram + theta/delta helper
|  |- inference.py                   # inference router
|  |- run_inference_sdreamer.py      # EEG/EMG model runner
|  |- run_inference_ne.py            # EEG/EMG/NE model runner
|  |- preprocessing.py               # model input preparation
|  |- postprocessing.py              # heuristic cleanup + stats
|  |- make_mp4.py                    # video clip extraction
|  |- assets/
|  |  |- closeWindow.js
|  |  |- videos/
|  |- app.py                         # older app implementation
|  |- components.py                  # older UI implementation
|  |- make_figure.py                 # older figure implementation
|  |- app_background_callback.py     # older/experimental callback path
|  |- sketch_*.py / refactor_*.py    # experiments and scratch work
|- models/
|  |- sdreamer/
|  |  |- n2nSeqNewMoE2.py
|  |  |- n2nBaseLineNE.py
|  |  |- layers/
|  |  |- checkpoints/
|- user_test_files/                  # sample MAT inputs and example videos
|- tests/                            # pytest suite for active modules
|- demo_data/                        # small demo example(s)
|- archive/                          # older model/app code and checkpoints
|- msda_version1.1/                  # older model lineage
|- build/                            # PyInstaller build artifacts
|- dist/                             # packaged app artifacts
|- *.spec                            # PyInstaller specs
|- environment.yml / requirements.txt
```

## What Looks Active vs. Legacy

### Active / relevant now

- [`run_desktop_app.py`](C:\Users\yzhao\python_projects\sleep_scoring\run_desktop_app.py)
- [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py)
- [`app_src/components_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\components_dev.py)
- [`app_src/make_figure_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_figure_dev.py)
- [`app_src/get_fft_plots.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\get_fft_plots.py)
- [`app_src/preprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\preprocessing.py)
- [`app_src/inference.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\inference.py)
- [`app_src/run_inference_sdreamer.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\run_inference_sdreamer.py)
- [`app_src/run_inference_ne.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\run_inference_ne.py)
- [`app_src/postprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\postprocessing.py)
- [`app_src/make_mp4.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_mp4.py)
- [`models/sdreamer`](C:\Users\yzhao\python_projects\sleep_scoring\models\sdreamer)
- [`user_test_files`](C:\Users\yzhao\python_projects\sleep_scoring\user_test_files)
- [`tests`](C:\Users\yzhao\python_projects\sleep_scoring\tests)

### Likely older or secondary

- [`app_src/app.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app.py)
- [`app_src/components.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\components.py)
- [`app_src/make_figure.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_figure.py)
- [`app_src/app_background_callback.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_background_callback.py)
- [`archive`](C:\Users\yzhao\python_projects\sleep_scoring\archive)
- [`msda_version1.1`](C:\Users\yzhao\python_projects\sleep_scoring\msda_version1.1)
- [`build`](C:\Users\yzhao\python_projects\sleep_scoring\build)
- [`dist`](C:\Users\yzhao\python_projects\sleep_scoring\dist)
- scratch files like `sketch_*`, `refactor_*`, and other experimental helpers under `app_src`

## Practical Mental Model

If you only want to understand the current product, read files in this order:

1. [`README.md`](C:\Users\yzhao\python_projects\sleep_scoring\README.md)
2. [`run_desktop_app.py`](C:\Users\yzhao\python_projects\sleep_scoring\run_desktop_app.py)
3. [`app_src/app_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\app_dev.py)
4. [`app_src/components_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\components_dev.py)
5. [`app_src/make_figure_dev.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_figure_dev.py)
6. [`app_src/get_fft_plots.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\get_fft_plots.py)
7. [`app_src/inference.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\inference.py)
8. [`app_src/preprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\preprocessing.py)
9. [`app_src/postprocessing.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\postprocessing.py)
10. [`app_src/make_mp4.py`](C:\Users\yzhao\python_projects\sleep_scoring\app_src\make_mp4.py)

## Questions Worth Clarifying Later

These are not blockers, just places where your intent would be useful if we keep documenting or refactoring:

- Whether `app.py` should now be treated as fully deprecated
- Which sample `.mat` files you consider the best canonical fixtures
- Whether `build/` and `dist/` should stay in-repo or be ignored/cleaned up in the future
- Whether `archive/` and `msda_version1.1/` should be documented as historical research lineage
