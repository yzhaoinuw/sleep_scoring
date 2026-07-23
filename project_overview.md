# Sleep Scoring Project Overview

This document maps the current GitHub repository and active runtime. End-user
installation, usage, and input-file instructions belong in
[README.md](README.md); contribution workflow belongs in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Product and repository boundary

Sleep Scoring is a Dash application embedded in a native `pywebview` window.
It can:

- visualize EEG, EMG, and optional norepinephrine (NE) signals;
- display a spectrogram, sleep scores, and aligned video;
- support keyboard-driven manual annotation and undo;
- run the statistical model or optional sDREAMER automatic scoring;
- save annotations to `.mat` and export completed summaries to Excel.

The public repository intentionally does **not** contain private recordings,
example lab data, videos, sDREAMER checkpoint files, or generated package
outputs. In particular, these local or generated paths are ignored:

- `user_test_files/` and `demo_data/`
- `models/sdreamer/checkpoints/`
- `app_src/assets/videos/`
- `build/`, `dist/`, and `release_artifacts/`
- local archives, caches, and scratch files

The packaged Windows distribution contains the runtime files and model
checkpoints needed by that distribution. See the README for access and setup.

## Runtime flow

```text
run_desktop_app.py
  -> claim a free desktop-window slot
  -> run the packaged-app update check when applicable
  -> import app_src/app.py
       -> create the Dash/Flask app in app_src/server.py
       -> register routes and callbacks
  -> start the local server
  -> open the native pywebview window
```

[`run_desktop_app.py`](run_desktop_app.py) is the only supported desktop
entrypoint. It detects source versus packaged execution, starts the Dash server
in a background thread, and opens the UI on localhost.

Up to three app processes can run side by side. Each claims one port beginning
at 8050 and receives its own window slot. Only slot 0 checks for packaged
source updates, so `app_src/` is not replaced while another window is using it.

## Active code map

### Application shell

| Path | Responsibility |
| --- | --- |
| [`app_src/app.py`](app_src/app.py) | Thin import aggregator that assembles the app |
| [`app_src/server.py`](app_src/server.py) | Dash/Flask instance, layout, filesystem cache, and per-window runtime paths |
| [`app_src/routes.py`](app_src/routes.py) | Resampling, profiling-log, and peer-window Flask endpoints |
| [`app_src/dialogs.py`](app_src/dialogs.py) | Native `.mat`, video, and save dialogs |
| [`app_src/session.py`](app_src/session.py) | Recording setup, metadata extraction, cache initialization, recovery, and same-file peer checks |
| [`app_src/resampling.py`](app_src/resampling.py) | Live resampler-figure store and patch/profiling helpers |

### UI and interaction

| Path | Responsibility |
| --- | --- |
| [`app_src/components.py`](app_src/components.py) | Home screen, graph, controls, modals, and hidden state stores |
| [`app_src/callbacks/`](app_src/callbacks) | Loading, navigation, prediction, saving, video, and clientside callback registration |
| [`app_src/assets/clientsideCallbacks.js`](app_src/assets/clientsideCallbacks.js) | Browser-side navigation and annotation callbacks |
| [`app_src/assets/`](app_src/assets) | Pointer pan, auto-pan, graph update, context-menu, profiling, and window-close scripts |

Keyboard events enter through `dash_extensions.EventListener`. The callback
modules validate selected `.mat` files, build and update the figure, track
annotation history and recovery state, save results, and prepare video clips.

For a deeper feature-by-feature explanation, see
[dash_app_cookbook.md](dash_app_cookbook.md).

### Figure generation

[`app_src/make_figure.py`](app_src/make_figure.py) creates the interactive
Plotly figure and overlays sleep-score heatmaps. The standard four-row layout
contains:

1. EEG spectrogram and theta/delta ratio
2. EEG trace
3. EMG trace
4. NE trace

[`app_src/get_fft_plots.py`](app_src/get_fft_plots.py) computes the 0-30 Hz
spectrogram and theta/delta ratio with `scipy.signal.ShortTimeFFT`.
`plotly-resampler` keeps navigation responsive on long signals, while
[`app_src/config.py`](app_src/config.py) holds user-adjustable display and
inference settings.

### Prediction

[`app_src/inference.py`](app_src/inference.py) routes to the backend selected by
`SLEEP_SCORING_MODEL`:

| Backend | Active path |
| --- | --- |
| Statistical Wake/REM model | [`app_src/run_inference_stats_model.py`](app_src/run_inference_stats_model.py) |
| sDREAMER with EEG/EMG | [`app_src/run_inference_sdreamer.py`](app_src/run_inference_sdreamer.py) |
| sDREAMER with EEG/EMG/NE | [`app_src/run_inference_ne.py`](app_src/run_inference_ne.py) |

[`app_src/preprocessing.py`](app_src/preprocessing.py) standardizes and reshapes
model inputs, resamples EEG/EMG to 512 Hz for sDREAMER, and builds 10 Hz NE
windows for the NE-aware model. [`app_src/mat_utils.py`](app_src/mat_utils.py)
normalizes shared `.mat` metadata details such as the `ne_frequency` /
`fp_frequency` alias.

The tracked [`models/sdreamer/`](models/sdreamer) package contains the model
definitions and transformer layers. Its checkpoint directory is intentionally
excluded from GitHub and supplied separately for installations that use
sDREAMER. The statistical model does not need Torch or checkpoint files.

### Saving and video

[`app_src/postprocessing.py`](app_src/postprocessing.py) converts dense scores
into contiguous bouts, applies prediction cleanup rules, derives summary
statistics and transition counts, and standardizes NE before saving when
present.

[`app_src/make_mp4.py`](app_src/make_mp4.py) uses the bundled
`imageio-ffmpeg` executable to cut the selected time range from an aligned
video. Generated clips live in ignored per-window subfolders under
`app_src/assets/videos/`; they are runtime output, not repository content.

## User-data contract

The app reads `.mat` files created by the
[preprocess_sleep_data](https://github.com/yzhaoinuw/preprocess_sleep_data)
workflow. The required fields are:

- `eeg`
- `eeg_frequency`
- `emg`

Optional fields include `ne`, `ne_frequency` or `fp_frequency`,
`sleep_scores`, `start_time`, `video_name`, `video_path`, and
`video_start_time`. README.md is the source of truth for the user-facing input
contract and sampling-rate behavior.

Runtime data stays local to the user's computer:

- saving annotations modifies the selected `.mat` file;
- completed scoring can create an Excel summary at a user-selected location;
- cache, crash recovery, and generated clips are stored in ignored local
  paths;
- each desktop-window slot isolates its cache, recovery state, and generated
  clips.

## GitHub-visible structure

This map intentionally lists tracked, maintained content rather than every
folder that may exist in a developer's checkout:

```text
sleep_scoring/
|- run_desktop_app.py              # desktop entrypoint
|- app_src/                        # active application package
|  |- callbacks/                   # server-side and clientside registrations
|  |- assets/                      # tracked browser interaction scripts
|  |- server.py, routes.py         # Dash/Flask shell and endpoints
|  |- session.py, resampling.py    # recording and live-figure state
|  |- components.py                # UI definitions
|  |- make_figure.py               # Plotly figure builder
|  |- inference.py                 # prediction router
|  |- mat_utils.py                 # shared MAT metadata helpers
|  |- preprocessing.py             # model input preparation
|  |- postprocessing.py            # prediction cleanup and exports
|  |- make_mp4.py                  # aligned video extraction
|- models/sdreamer/                # model definitions; checkpoints excluded
|- tests/                          # Python and clientside JavaScript tests
|- packaging/windows/              # full-package and source-update builders
|- paper/                          # JOSS manuscript sources
|- README.md                       # installation, usage, and input contract
|- CONTRIBUTING.md                 # contributor workflow
|- dash_app_cookbook.md            # detailed implementation recipes
|- pyproject.toml                  # package metadata and dependencies
|- environment.yml                # portable Conda environment
```

## Suggested reading order

To understand the current product:

1. [README.md](README.md)
2. [`run_desktop_app.py`](run_desktop_app.py)
3. [`app_src/server.py`](app_src/server.py) and
   [`app_src/app.py`](app_src/app.py)
4. [`app_src/components.py`](app_src/components.py)
5. the relevant module in [`app_src/callbacks/`](app_src/callbacks)
6. [`app_src/make_figure.py`](app_src/make_figure.py)
7. [`app_src/inference.py`](app_src/inference.py)
8. [`app_src/mat_utils.py`](app_src/mat_utils.py) and
   [`app_src/preprocessing.py`](app_src/preprocessing.py)
9. [`app_src/postprocessing.py`](app_src/postprocessing.py) or
   [`app_src/make_mp4.py`](app_src/make_mp4.py)

Use [packaging/windows/README.md](packaging/windows/README.md) for release
artifact construction and [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration
and verification.
