[![Agent Collab Treaty](https://raw.githubusercontent.com/yzhaoinuw/agent_collab_treaty/main/assets/treaty-adopted.svg)](https://github.com/yzhaoinuw/agent_collab_treaty)

# Installation

## Windows Users

1. Go to the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g) on OneDrive. Contact Yue if you can't access it.
2. Download **_sleep_scoring_app_vx.zip_**. The **x** in **vx** denotes the current version. If you unzip it to the same location as the zip file, you may end up with a nested folder, such as a _sleep_scoring_app_vx/_ folder inside another _sleep_scoring_app_vx/_ folder. If this happens, move the inner folder somewhere else and delete the outer one to avoid confusion.
3. Check that inside the unzipped sleep_scoring_app_vx directory, you have:
   - **__internal/_**
   - **_app_src/_**
   - **_models/_**
   - **_unblock_app.cmd_**
   - **_run_desktop_app.exe_**
4. Double click **_unblock_app.cmd_**. It unblocks the downloaded app files if Windows marked them as blocked, then starts **_run_desktop_app.exe_**.

### Updates

Auto-update-enabled Windows builds check the latest GitHub Release when the app starts and may update **_app_src/_** before the window opens. If the update check fails, is offline, is incompatible, or detects local edits in files it needs to replace, the app opens normally and you can use a new full zip instead.

Dependency, model, or packaged runtime changes still require downloading a new full app zip.

## Mac Users

The app has been tested on macOS Tahoe. To download, follow [Build From Source](#build-from-source-run-using-anaconda).

## Before Usage

- For best performance, copy the unzipped app folder to your own computer before running it. The app includes the sDREAMER model files but does not include the optional sDREAMER Torch runtime needed for deep-learning automatic scoring.
- Use only one Sleep Scoring App session per computer.
- If the graph feels slow, close unnecessary browser tabs and other heavy apps before scoring.

# Usage

To open the app, double click **_unblock_app.cmd_** and it will open the app's home page. After the folder has been unblocked, double clicking **_run_desktop_app.exe_** directly also works. Select a .mat file to visualize its EEG, EMG, and/or NE signals. There are two modes: [**navigation/panning mode**](#navigation) and [**annotation mode**](#annotation). To swap between them, press M on the keyboard.

## Navigation

Every time you open a new mat file, you are in **navigation/panning mode** initially. When in this mode:

- Click-and-drag left/right on any plot to pan horizontally. You can also press the left or right arrow key to move horizontally. This may come especially handy when you are in annotation mode.
- On the EEG and EMG plots, you may also drag up/down to pan vertically.

> Note: The spectrogram and NE plots are vertically fixed, so they only allow horizontal panning.

### Zooming

To zoom in or out:

- Hover your cursor over a plot and scroll your mouse wheel.
- To zoom X-axis only, hover over the spectrogram or NE plot.
- To zoom Y-axis only, move the cursor slightly to the left of the plot's Y-axis before scrolling.
- Zooming inside the EEG and EMG plot interior zooms both axes.

To reset the view, click Reset Axes in the mode bar in the upper-right above the graph. The mode bar may be hidden but will appear if your cursor is within the graph area.

> Note: The spectrogram and the NE plot are fixed on the Y-axis, so you can only zoom on the X-axis on them. You can change this behavior for the NE plot. Open **_config.py_** in **_app_src/_** and change the line `FIX_NE_Y_RANGE = False` to `FIX_NE_Y_RANGE = True`.

https://github.com/user-attachments/assets/d0daa3ff-18dc-43bb-beb3-742209ae5f60

## Annotation

In annotation mode, you can annotate sleep scores or use [automatic sleep scoring](#automatic-sleep-scoring). You can also [check video](#check-video).

- Click a region to highlight it with a thin selection box.
- Assign a score by pressing 1, 2, or 3 on your keyboard. The selected region will be colored correspondingly.
- To annotate a wider region, click-and-drag to draw a wider box.
- If you keep dragging past the left or right edge of the visible graph, the view will auto-pan so you can continue selecting a longer region without leaving annotation mode.
- You can overwrite existing annotations by selecting and reassigning them.
- Use the Undo Annotation button in the bottom-left below the graph to undo the last annotation. This button only appears when there is something to undo.

To customize the colors used for Wake, NREM, REM, and MA, edit
`SLEEP_STAGE_COLORS` in _app_src/config.py_ and restart the app. Each value can
be a Plotly-compatible color such as `"#4477AA"` or `"rgb(68, 119, 170)"`.

https://github.com/user-attachments/assets/1c513a72-53be-440a-aaa8-c52e0ffc64d4

### Select a Contiguous Segment

While in annotation mode, right-click inside an existing scored or unscored segment to select that whole contiguous segment. You can then press 1, 2, or 3 to assign a score to the selected segment.

### Check Video

While in annotation mode:

- If your selected region spans less than 300 seconds, click the **Check Video** button in the upper-left above the graph to open and play the corresponding video clip.

> Note: If this is your first time checking video for a given .mat file, you will be prompted to choose the corresponding .avi file. If the .avi file was found during [preprocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev), the app will show the file path to help you find it.

### Automatic Sleep Scoring

Automatic scoring can use either the statistical Wake/REM model or sDREAMER. To switch models, open _app_src/config.py_ and set `SLEEP_SCORING_MODEL` to either `"stats_model"` or `"sdreamer"`.

The statistical model does not need the optional sDREAMER Torch runtime. Its user-facing settings are in _app_src/config.py_:

- `STATS_MODEL_WAKE_THRESHOLD`
- `STATS_MODEL_MIN_WAKE_DURATION`
- `STATS_MODEL_MIN_REM_DURATION`

For sDREAMER, the Windows app zip includes the model files, but not the optional Torch runtime. To enable sDREAMER:

- Download *torch.zip* from the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g).
- Unzip it, ensuring it does not remain nested inside another folder.
- Copy all folders and files from the unzipped runtime directly into the app's _internal/_ folder. After copying, _internal/_ should contain _torch/_.
- Reopen the app and it should be enabled automatically.

After enabling:

- While in annotation mode, click the **Generate Predictions** button in the upper-right above the graph. This will load the model and run prediction in the background.
- You can track progress in the command-line window.
- When finished, the prediction will appear on the graph.
- You can annotate to correct it, or undo the prediction and annotate manually.

## Save Sleep Scores

Click the **Save Annotations** button in the bottom-left below the graph to save your sleep scores. They will be saved directly into the original .mat file.

If any part of the recording is still unscored, the app reports the first unscored range in the annotation message as `[start, end] (duration s)`, even if you cancel the .mat save dialog. Score that range and save again.

If the .mat file has been sleep scored completely, you will also be prompted to export sleep bouts and simple statistics to an Excel file.

> Note: When you use automatic sleep scoring, the last few seconds may be unscored because of the deep learning model's input sequence length. To get the sleep bout Excel file, manually score the last few seconds.

https://github.com/user-attachments/assets/2c08644e-cd0e-4f37-8912-da17ab6c9456

## Multiple Windows

You can open up to three app windows at the same time, for example to compare two recordings side by side. Just launch the app again while it is already running; each window is fully independent. The second and third windows show a number in their title bar, e.g. `(2)`.

- The same .mat file cannot be open in two windows at once. If you select a file that another window already has open, the app shows a reminder instead of loading it — pick a different file, or close it in the other window first.
- Video clips and each recording's video association are tracked per window.
- When more than one window is open, only the first window checks for app updates at startup.

## Additional Notes

- Crash recovery is stored separately for each window position. Relaunch windows
  in their original order and open the same .mat file in the matching position
  before opening any other file: the first window has no number, the second
  shows `(2)`, and the third shows `(3)`. Opening a different file clears that
  window's unsaved recovery, including a file with the same name in a different
  folder.

---

# Developer Notes

For collaboration workflow, branch habits, test expectations, and documentation conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Windows Packaging

Windows packages are built with the scripts in [`packaging/windows/`](packaging/windows/) instead of running PyInstaller by hand. For a full app zip, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1
```

The script runs tests, builds with PyInstaller, copies `app_src/`, `models/`, and the starter command into the release folder, smoke-checks the packaged app, and writes the zip plus manifest/hash files under `release_artifacts/`. For code-only releases after an auto-update-enabled full build, see [`packaging/windows/README.md`](packaging/windows/README.md).

## Input File

The input files to the app must be `.mat` (MATLAB) files created from raw recording files with the [preprocess_sleep_data](https://github.com/yzhaoinuw/preprocess_sleep_data) MATLAB preprocessing workflow. The `.mat` files contain the following fields.

#### Required fields

| Field Name | Data Type |
| --- | --- |
| **_eeg_** | 1 x *N* single |
| **_eeg_frequency_** | double |
| **_emg_** | 1 x *N* single |

#### Optional fields

| Field Name | Data Type |
| --- | --- |
| *ne* | 1 x *M* single |
| *ne_frequency* (alias: *fp_frequency*) | double |
| *sleep_scores* | single |
| *start_time* | double |
| *video_name* | char |
| *video_path* | char |
| *video_start_time* | double |

**Sampling-rate note:** Visualization supports variable EEG/EMG sampling rates through `eeg_frequency` and variable NE sampling rates through `ne_frequency` (or its `fp_frequency` alias, for recordings whose upstream pipeline names the fiber-photometry sampling rate that way); EMG is assumed to share `eeg_frequency`. Automatic scoring with `stats_model` uses those frequencies directly. sDREAMER can also accommodate non-512 Hz EEG/EMG by resampling them to 512 Hz for prediction, but NE-aware sDREAMER expects NE at 10 Hz.

**Explanations**

1. *start_time* is not *0* if the .mat file came from a longer recording (>12 hours) that was segmented into 12-hour or shorter bins.
2. *video_path* is the .avi path found during preprocessing.
3. *video_start_time* is the TTL pulse onset found on the EEG side, such as Viewpoint or Pinnacle.

## Build From Source (Run Using Anaconda)

There are two preparation steps that you need to follow before using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instructions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
2. Get Git if you haven't. You need it to download the repo and to get updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git.

### Download source code

```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
```

This command downloads the source code into the directory where it is run. You can move the source code folder anywhere you like afterwards. Then use `cd` in your command line to change to the folder where you placed the **_sleep_scoring/_** folder.


### Set up the environment

After you have done the prep work above, open your Anaconda terminal or Anaconda Powershell Prompt. To recreate the project environment, run this from the `sleep_scoring/` folder:

```bash
conda env create -f environment.yml
conda activate sleep_scoring_dash3.0
```

In the future, every time before you run the app, make sure you activate this environment.


If you want to use sDREAMER, install the PyTorch build recommended for the target computer from https://pytorch.org/get-started/locally/, then install the sDREAMER helper packages:

```bash
pip install timm==1.0.22 einops==0.8.1
```

### Running the app

Last step, type:

```bash
python run_desktop_app.py
```

to run the app.

### Updating the app

When there's an update announced, it's straightforward to get the update from source. Have the environment activated, cd to the source code folder, then type:

```bash
git pull
```

If dependencies have changed, reinstall:

```bash
pip install -e .
```

# Citation

If you use this app in your research, please cite it. The repository includes a
[`CITATION.cff`](CITATION.cff) file, so you can use GitHub's **"Cite this
repository"** button (top right of the repo page) to get an APA or BibTeX
entry.

A JOSS paper describing the software is in preparation (see
[`paper/paper.md`](paper/paper.md)); once it is published, please cite the paper
instead.
