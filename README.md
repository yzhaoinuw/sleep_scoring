# Installation

## Windows Users

1. Go to the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g) on OneDrive. Contact Yue if you can't access it.
2. Download **_sleep_scoring_app_vx.zip_**. The **x** in **vx** denotes the current version. If you unzip it to the same location as the zip file, you may end up with a nested folder, such as a _sleep_scoring_app_vx/_ folder inside another _sleep_scoring_app_vx/_ folder. If this happens, move the inner folder somewhere else and delete the outer one to avoid confusion.
3. Check that inside the unzipped sleep_scoring_app_vx directory, you have:
   - **__internal/_**
   - **_app_src/_**
   - **_models/_**
   - **_run_app.exe_**
4. After you download and unzip, open PowerShell and run:

```powershell
cd PATH_TO_YOUR_APP_FOLDER
Get-ChildItem -Recurse | Unblock-File
```

The first line navigates to the unzipped app folder. Replace `PATH_TO_YOUR_APP_FOLDER` with the actual path to the app folder on your computer. The second line unblocks the webview dependencies that provide the app window. Windows may block those files when the zip is downloaded from OneDrive; without unblocking them, the app may not run.

## Mac Users

The app has been tested on macOS Tahoe. To download, follow [Build From Source](#build-from-source-run-using-anaconda).

# Usage

To open the app, double click **_run_desktop_app.exe_** and it will open the app's home page. Select a .mat file to visualize its EEG, EMG, and/or NE signals. There are two modes: [**navigation/panning mode**](#navigation) and [**annotation mode**](#annotation). To swap between them, press M on the keyboard.

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

https://github.com/user-attachments/assets/1c513a72-53be-440a-aaa8-c52e0ffc64d4

### Select a Contiguous Segment

While in annotation mode, right-click inside an existing scored or unscored segment to select that whole contiguous segment. You can then press 1, 2, or 3 to assign a score to the selected segment.

### Check Video

While in annotation mode:

- If your selected region spans less than 300 seconds, click the **Check Video** button in the upper-left above the graph to open and play the corresponding video clip.

> Note: If this is your first time checking video for a given .mat file, you will be prompted to choose the corresponding .avi file. If the .avi file was found during [preprocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev), the app will show the file path to help you find it.

### Automatic Sleep Scoring

Automatic scoring is no longer included by default. To enable it:

- Download *torch.zip* from the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g).
- Unzip it, ensuring it does not remain nested inside another folder.
- Place it directly inside _internal/ and **NOT** inside any subfolder. Review the [Installation](#installation) section if needed.
- Reopen the app and it should be enabled automatically.

After enabling:

- While in annotation mode, click the **Generate Predictions** button in the upper-right above the graph. This will load the model and run prediction in the background.
- You can track progress in the command-line window.
- When finished, the prediction will appear on the graph.
- You can annotate to correct it, or undo the prediction and annotate manually.

## Save Sleep Scores

Click the **Save Annotations** button in the bottom-left below the graph to save your sleep scores. They will be saved directly into the original .mat file. If the .mat file has been sleep scored completely, you will also be prompted to export sleep bouts and simple statistics to an Excel file.

> Note: When you use automatic sleep scoring, the last few seconds may be unscored because of the deep learning model's input sequence length. To get the sleep bout Excel file, manually score the last few seconds.

https://github.com/user-attachments/assets/2c08644e-cd0e-4f37-8912-da17ab6c9456

## Additional Notes

- If the app crashes before you get to save your sleep scores, don't panic. Reopen the app and open the **SAME** mat file that you were just working on to recover your work. Note that you **MUST** open the **SAME** file that you were working on when the app crashed. If you open any other file, you will lose your unsaved work for good.

---

# Developer Notes

For collaboration workflow, branch habits, test expectations, and documentation conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Input File

The input files to the app must be .mat (matlab) files, and contain the following fields.

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
| *ne_frequency* | double |
| *sleep_scores* | single |
| *start_time* | double |
| *video_name* | char |
| *video_path* | char |
| *video_start_time* | double |

**Explanations**

1. *start_time* is not *0* if the .mat file came from a longer recording (>12 hours) that was segmented into 12-hour or shorter bins.
2. *video_path* is the .avi path found during preprocessing.
3. *video_start_time* is the TTL pulse onset found on the EEG side, such as Viewpoint or Pinnacle.

## Build From Source (Run Using Anaconda)

There are two preparation steps that you need to follow before using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instructions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
2. Get Git if you haven't. You need it to download the repo and to get updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git.

#### Download source code

```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
```

This command downloads the source code into the directory where it is run. You can move the source code folder anywhere you like afterwards. Then use `cd` in your command line to change to the folder where you placed the **_sleep_scoring/_** folder.

#### Download model checkpoints

To use automatic sleep scoring, download the checkpoints of the trained model from the [OneDrive folder](https://uofr-my.sharepoint.com/:f:/r/personal/yzhao38_ur_rochester_edu/Documents/sleep_scoring_project?csf=1&web=1&e=Kw7OEB). Then, unzip if needed, and place the checkpoints in **_models/sdreamer/_** in the app folder.

#### Set up the environment

After you have done the prep work above, open your Anaconda terminal or Anaconda Powershell Prompt and create an environment with Python 3.11. For consistency with the project agent notes, use the environment name `sleep_scoring_dash3.0`.

```bash
conda create -n sleep_scoring_dash3.0 python=3.11
```

Then activate the environment:

```bash
conda activate sleep_scoring_dash3.0
```

In the future, every time before you run the app, make sure you activate this environment. Next, when you are in the *sleep_scoring/* folder, install the app with all dependencies including PyTorch for automatic sleep scoring. You only need to do this once.

```bash
pip install -e ".[ml]"
```

#### Running the app

Last step, type:

```bash
python run_desktop_app.py
```

to run the app.

#### Updating the app

When there's an update announced, it's straightforward to get the update from source. Have the environment activated, cd to the source code folder, then type:

```bash
git pull
```

If dependencies have changed, reinstall:

```bash
pip install -e ".[ml]"
```
