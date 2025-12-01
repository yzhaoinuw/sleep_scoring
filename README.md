## 📦 Installation
1. Go to the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g) on OneDrive. Contact Yue if you can't access it.
2. Download **_sleep_scoring_app_vx.zip_** (the "**x**" in the suffix "**vx**" denotes the current version). Note that if you unzip it to the same location where the zip file is, you may end up with a nested folder, ie., a _sleep_scoring_app_vx/_ folder inside another _sleep_scoring_app_vx/_ folder. If this happens, move the inner folder somewhere else and delete the outer one to avoid confusion.  
2. Inside the unzipped sleep_scoring_app_vx directory, you should see:
- **__internal/_**
- **_app_src/_**
- **_models/_**
- **_run_app.exe_**


## 🎹 Usage 
To open the app, double click **_run_app.exe_** and it will open the app's home page in a tab in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.

#### ↔️ Navigation
Every time you open a new mat file, you are in the **navigation/panning mode** initially. When in this mode, 
- Click-and-drag left/right on any plot to pan horizontally. You can also press ⬅️ or ➡️ on the keyboard to move horizontally. This may come especially handy when you are in annotation mode (see [Annotation](#%EF%B8%8F-annotation)) 
- On the EEG and EMG plots, you may also drag up/down to pan vertically.

> Note: The spectrogram and NE plots are vertically fixed, so they only allow horizontal panning.

#### 🔎 Zooming
To zoom in or out, 
- Hover your cursor over a plot and scroll your mouse wheel.
- To zoom X-axis only, hover over the spectrogram or NE plot.
- To zoom Y-axis only, move the cursor slightly to the left of the plot’s Y-axis before scrolling.
- Zooming inside EEG and EMG plot interior zooms both axes.

To reset the view, click Reset Axes in the mode bar (upper-right, above the graph). The mode bar may be hidden but will appear if your cursor is within the graph area.

> Note: The spectrogram and the NE plot are fixed on the Y-axis, so you can only zoom on the X-axis on them. **If you want to change this behavior, contact Yue for an easy hack.** 

https://github.com/user-attachments/assets/d0daa3ff-18dc-43bb-beb3-742209ae5f60

#### ✏️ Annotation
Press M to switch to **annotation mode**.
- Click a region to highlight it with a thin selection box.
- Assign a score by pressing 1, 2, or 3 on your keyboard. The selected region will be colored correspondingly.
- To annotate a wider region, click-and-drag to draw a wider box.
- You can overwrite existing annotations by selecting and reassigning them.
- Use the Undo Annotation button (bottom-left, below the graph) to undo up to the three most recent annotations. This button only appears when there is something to undo.

> Note: To return to navigation/panning mode, press M again.

https://github.com/user-attachments/assets/1c513a72-53be-440a-aaa8-c52e0ffc64d4

#### 🎥 Check video
While in annotation mode,
- If your selected region spans less than 300 seconds, click **Check Video** button (upper-left, above the graph)to open and play the corresponding video clip.

> Note: If this is your first time checking video for a given .mat file, you will be prompted to upload the corresponding .avi file. If the .avi file was found during [procrocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev), the app will show the file path to help you find it.


#### 🚗 Automatic sleep scoring
Automatic scoring is no longer included by default. To enable it,
- Download *torch.zip* from the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g).
- Unzip it (ensure it does not remain nested inside another folder).
- Place it directly inside _internal/ and **NOT** inside any subfolder (please review the [Installation Section](#-installation)).
- Reopen the app and it should be enabled automatically.

After enabling:
- While in annotation mode, click **Generate Predictions** button (upper-right, above the graph). This will load the model and run prediction in the background.
- You can track progress in the command-line window.
- When finished, the prediction will appear on the graph.
- You can annotate to correct it, or undo the prediction and annotate manually.

#### 📝 Save sleep scores
Click **Save Annotations** button (bottom-left, below the graph) to save your sleep scores. They will be saved directly into the original .mat file. If the .mat file has been sleep scored completely, you will also be prompted to export sleep bouts and simple statistics to an Excel file.
> Note: When you use the automatic sleep scoring, it's likely that the last few seconds are not sleep scored being the residue of the deep learning model's input sequence. To get the sleep bout Excel file, you need so manually score the last few seconds.

https://github.com/user-attachments/assets/2c08644e-cd0e-4f37-8912-da17ab6c9456

#### 📓 Additional note
- If your browser does not prompt you for a save location, it may have automatically downloaded to the Downloads folder. Adjust your browser settings if needed.
- Closing other tabs in the same browser window may improve performance.
- If the app crashes on you before you get to save your sleep scores, don't panic. Reopen the app and open the **SAME** mat file that you were just working on to recover your work. Note that you **MUST** open the **SAME** file that you were working on when the app crashed. If you open any other file, you will lose your unsaved work for good.

---
# Developer Notes
## Input File 
The input files to the app must be .mat (matlab) files, and contain the following fields.

#### Required fields
| Field Name          | Data Type      |
| --------------------|----------------|
| **_eeg_**           | 1 x *N* single |
| **_eeg_frequency_** | double         |
| **_emg_**           | 1 x *N* single |

#### Optional fields
| Field Name         | Data Type      | 
| -------------------|----------------|
| *ne*               | 1 x *M* single | 
| *ne_frequency*     | double         |
| *sleep_scores*     | single         |
| *start_time*       | double         |
| *video_name*       | char           |
| *video_path*       | char           | 
| *video_start_time* | double         |

**Explanations**

 1. *start_time* is not *0* if the .mat file came from a longer recording (>12 hours) that was segmented into 12-hour or less bins.
 2. *video_path* is the .avi path found during preprocessing.
 3. *video_start_time* is the TTL pulse onset found on the EEG side (such as Viewpoint, Pinnacle).
 

## Build From Source (Run Using Anaconda)
There are two preparation processes that you need to do before using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instrcutions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

2. Get Git if you haven't. You need it to download the repo and to get updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git.

#### Download source code
```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
```
In whatever directory you run this command will download the source code there. You can place the source code folder anywhere you like afterwards. Then use the command `cd`, which stands for change directory, in your command line to change to where you place the *sleep_scoring_app_vx/* folder. 

#### Set up the environment
After you have done the prep work above, open you Anaconda terminal or Anaconda Powershell Prompt, create an environment with Python 3.10
```bash
conda create -n sleep_scoring python=3.10
```
Then, activate the sleep_scoring environment by typing
```bash
conda activate sleep_scoring
```
In the future, every time before you run the app, make sure you activate this environment. Next, When you are in the *sleep_scoring_app_vx/* folder, install all the dependencies for the app. You only need to do it once.
```bash
pip install -r requirements.txt 
```

#### Running the app
Last step, type
```bash
python main.py
```
to run the app.

#### Updating the app
When there's an update announced, it's straightforward to get the update from source. Have the environment activated, cd to the source code folder, then type
```bash
git pull origin dev
```
