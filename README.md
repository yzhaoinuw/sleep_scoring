# Sleep Scoring

[![Agent Collab Treaty](https://raw.githubusercontent.com/yzhaoinuw/agent_collab_treaty/main/assets/treaty-adopted.svg)](https://github.com/yzhaoinuw/agent_collab_treaty)

A desktop app for viewing EEG, EMG, and optional norepinephrine (NE) signals,
manually annotating sleep stages, checking aligned video, and optionally
generating automatic sleep scores.

## Contents

- [Install](#install)
- [Before Your First Session](#before-your-first-session)
- [Use The App](#use-the-app)
- [Input Files](#input-files)
- [Developer Documentation](#developer-documentation)
- [Citation](#citation)

## Install

### Choose An Installation

| You are... | Recommended installation | What you need |
| --- | --- | --- |
| A Windows user who wants to run the app without Git, Python, or Conda | [Packaged Windows app](#packaged-windows-app) | Access to the private distribution folder; request access from Yue Zhao if needed |
| A Windows user who wants to inspect or modify the code | [Run from source](#run-from-source-windows-or-macos) | Git and Miniconda |
| A macOS user | [Run from source](#run-from-source-windows-or-macos) | Git and Miniconda; no packaged macOS build is currently provided |
| A contributor | [Run from source](#run-from-source-windows-or-macos) | Git, Miniconda, and the checks in [CONTRIBUTING.md](CONTRIBUTING.md) |

The packaged Windows app is the simplest route for most users. The source
installation is public and cross-platform, but it requires command-line setup.
The source version has been tested on macOS Tahoe.

### Packaged Windows App

1. Open the private
   [sleep_scoring_project distribution folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g).
   If you cannot open it, request access from Yue Zhao, the repository
   maintainer.
2. Download `sleep_scoring_app_vX.Y.Z-windows.zip`, where `X.Y.Z` is the
   current version.
3. Extract the zip and move the extracted app folder onto your computer.
4. Double-click `unblock_app.cmd`. It removes Windows download blocking from
   the app files and then starts `run_desktop_app.exe`.

After the first launch, you can start the app with either `unblock_app.cmd` or
`run_desktop_app.exe`.

<details>
<summary>Troubleshoot the extracted folder layout</summary>

The app folder should directly contain:

- `_internal/`
- `app_src/`
- `models/`
- `unblock_app.cmd`
- `run_desktop_app.exe`

Some extraction tools create an extra nested folder with the same name. If
that happens, move the inner app folder to the location where you want to keep
it and remove the empty outer wrapper.

</details>

#### Packaged App Updates

The Windows app checks the latest GitHub Release when it starts and may update
compatible `app_src/` files before the window opens. If the check is offline,
fails, finds an incompatible update, or detects local edits to files it would
replace, the app still opens normally.

Dependency, model, launcher, or packaged-runtime changes require a new full
Windows zip from the private distribution folder.

### Run From Source (Windows Or macOS)

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
and [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html),
then run:

```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
cd sleep_scoring
conda env create -f environment.yml
conda activate sleep_scoring_dash3.0
python run_desktop_app.py
```

Activate `sleep_scoring_dash3.0` again whenever you open a new terminal before
running the app.

To update an existing source installation:

```bash
git pull
conda env update -f environment.yml
conda activate sleep_scoring_dash3.0
```

The source checkout uses the statistical automatic-scoring model by default,
which does not require PyTorch or separately distributed model checkpoints.

#### Optional sDREAMER Setup

sDREAMER is not required for visualization, annotation, video, saving, or the
default statistical model.

- **Packaged Windows app:** download `torch.zip` from the private distribution
  folder, extract it, and copy its contents directly into the app's
  `_internal/` folder. After copying, `_internal/torch/` should exist.
- **Source installation:** install the PyTorch build recommended for your
  computer from [pytorch.org](https://pytorch.org/get-started/locally/), then
  run `pip install timm==1.0.22 einops==0.8.1`. The sDREAMER checkpoint files
  are not stored in this public repository; request them from Yue Zhao and
  place them in `models/sdreamer/checkpoints/`.

Set `SLEEP_SCORING_MODEL = "sdreamer"` in `app_src/config.py`, then restart the
app.

## Before Your First Session

- For the packaged app, run it from a folder on your computer rather than
  directly from OneDrive, a network drive, or the downloaded zip.
- You can open up to three app windows on one computer, but the same `.mat`
  file cannot be open in two windows at once.
- If the graph feels slow, close unnecessary browser tabs and other
  resource-heavy applications.

## Use The App

### Start The App And Open A Recording

- **Packaged Windows app:** double-click `unblock_app.cmd` or
  `run_desktop_app.exe`.
- **Source installation:** activate the Conda environment and run
  `python run_desktop_app.py`.

Select a `.mat` file to visualize its EEG, EMG, and optional NE signals. The
app has two interaction modes:

- **Navigation mode:** pan and zoom the plots.
- **Annotation mode:** select time ranges and assign sleep stages.

Press <kbd>M</kbd> to switch modes.

### Navigate And Zoom

Every newly opened recording starts in navigation mode.

- Drag left or right on a plot to pan horizontally, or use the left and right
  arrow keys.
- Drag vertically on the EEG or EMG plot to pan its Y-axis.
- Scroll over a plot to zoom.
- Scroll over the spectrogram or NE plot to zoom only the X-axis.
- Scroll just to the left of a Y-axis to zoom only that axis.
- Use **Reset Axes** in the graph's upper-right mode bar to restore the view.

The spectrogram Y-axis is fixed. The NE Y-axis is adjustable by default; set
`FIX_NE_Y_RANGE = True` in `app_src/config.py` if you want to lock it.

https://github.com/user-attachments/assets/d0daa3ff-18dc-43bb-beb3-742209ae5f60

### Annotate Sleep Stages

In annotation mode:

- Click a region, then press <kbd>1</kbd>, <kbd>2</kbd>, or <kbd>3</kbd> to
  assign a sleep stage.
- Drag to select a wider region. Dragging beyond the visible edge auto-pans the
  graph so you can continue the selection.
- Right-click inside a scored or unscored segment to select that entire
  contiguous segment.
- Select an existing annotation and assign a new score to overwrite it.
- Use **Undo Annotation** below the graph to undo the most recent annotation.

https://github.com/user-attachments/assets/1c513a72-53be-440a-aaa8-c52e0ffc64d4

### Check Aligned Video

In annotation mode, select a region shorter than 300 seconds and click
**Check Video** above the graph.

The first time you check video for a recording, the app may ask you to locate
the matching `.avi` file. If the video was found during
[preprocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev),
the app displays that saved path to help you find it.

### Generate Automatic Scores

Choose the backend in `app_src/config.py`:

```python
SLEEP_SCORING_MODEL = "stats_model"  # or "sdreamer"
```

The statistical model works with the standard installation. Its user-facing
settings are also in `app_src/config.py`:

- `STATS_MODEL_WAKE_THRESHOLD`
- `STATS_MODEL_MIN_WAKE_DURATION`
- `STATS_MODEL_MIN_REM_DURATION`

After [setting up sDREAMER](#optional-sdreamer-setup), switch to annotation
mode and click **Generate Predictions**. Prediction runs in the background;
when it finishes, you can correct the result manually or undo it.

### Save Sleep Scores

Click **Save Annotations** below the graph. A native Save dialog opens with the
current `.mat` filename suggested. The app writes to the destination you
choose; it replaces the original recording only if you select that file and
confirm the replacement.

If any recording segment remains unscored, the app reports the first unscored
range as `[start, end] (duration s)`, even if you cancel the save dialog. Once
the recording is completely scored, the app also offers to export sleep bouts
and summary statistics to an Excel file.

Automatic scoring may leave a few seconds unscored at the end because of the
model's input sequence length. Score that remainder manually before exporting
the Excel summary.

https://github.com/user-attachments/assets/2c08644e-cd0e-4f37-8912-da17ab6c9456

### Use Multiple Windows And Crash Recovery

Launch the app again to open as many as three independent windows. The second
and third windows show `(2)` and `(3)` in their title bars.

- A recording already open in one window cannot be loaded in another.
- Video clips and saved video associations are isolated by window.
- Only the first window checks for app updates.
- Unsaved crash recovery is also isolated by window position. Relaunch windows
  in their original order and reopen the same recording in the matching
  position before opening a different file. Opening a different file clears
  that window's recovery state.

## Input Files

The app opens MATLAB `.mat` files produced from raw recordings by the
[preprocess_sleep_data](https://github.com/yzhaoinuw/preprocess_sleep_data)
workflow.

Required fields:

| Field | Type |
| --- | --- |
| `eeg` | 1 x *N* single |
| `eeg_frequency` | double |
| `emg` | 1 x *N* single |

<details>
<summary>Optional fields and timing details</summary>

| Field | Type |
| --- | --- |
| `ne` | 1 x *M* single |
| `ne_frequency` (alias: `fp_frequency`) | double |
| `sleep_scores` | single |
| `start_time` | double |
| `video_name` | char |
| `video_path` | char |
| `video_start_time` | double |

- `start_time` can be nonzero when a recording longer than 12 hours was split
  into shorter files.
- `video_path` is the `.avi` path found during preprocessing.
- `video_start_time` is the video TTL onset measured on the EEG acquisition
  side, such as Viewpoint or Pinnacle.

</details>

Visualization supports variable EEG/EMG sampling rates through
`eeg_frequency` and variable NE sampling rates through `ne_frequency` or
`fp_frequency`; EMG is assumed to share `eeg_frequency`. The statistical model
uses those frequencies directly. sDREAMER resamples EEG/EMG to 512 Hz for
prediction, while NE-aware sDREAMER expects NE at 10 Hz.

## Developer Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): contribution workflow, source setup, and
  checks
- [project_overview.md](project_overview.md): current architecture and
  repository boundaries
- [dash_app_cookbook.md](dash_app_cookbook.md): feature-by-feature
  implementation recipes
- [packaging/windows/README.md](packaging/windows/README.md): Windows release
  packaging and update assets

## Citation

If you use this app in research, use GitHub's **Cite this repository** button
or the repository's [CITATION.cff](CITATION.cff) file to obtain an APA or
BibTeX entry.

A JOSS paper is in preparation in [paper/paper.md](paper/paper.md). Once it is
published, please cite the paper instead.
