# Changelog

This file summarizes changes that affect people using the app. Maintainer-only
implementation and release details are recorded in `work_log.md`.

## v0.17.1

- Releases that require a new full package no longer appear as failed automatic
  updates. The app now links directly to GitHub Releases when a manual download
  is required.
- Windows downloads now use the explicit package name
  `sleep_scoring_app_v0.17_full.zip`.
- Existing v0.17.0 installations need to download this full Windows package
  once to receive the updated startup behavior.

## v0.17.0

- Automatic update checks are more reliable and run at most once per day during
  normal startup.
- Startup messages now show the installed version and, when applicable, the
  available version.
- The Windows download is named `sleep_scoring_app_v0.17_full.zip` and extracts
  to `sleep_scoring_app_v0.17`. The exact patch version remains visible in the
  terminal and app window title.
- Sleep-stage colors can be customized with `STAGE_COLORS` in
  `app_src/config.py`.
- Existing v0.16.x installations need to download this full Windows package
  once before receiving future compatible updates.

## v0.16.8

- Excel sleep-bout exports now retain manually scored microarousals and include
  them in duration, count, percentage, and transition statistics.
- Existing annotation files can be reopened and saved again to create a
  corrected spreadsheet without rescoring the recording.

## v0.16.7

- Added optional color customization for Wake, NREM, REM, and MA stages.
- Lightweight updates preserve existing settings. Installations without custom
  stage colors continue to use the standard palette.

## v0.16.5

- Added support for up to three independent app windows for comparing and
  annotating recordings side by side.
- Each window now keeps its own recovery data and video selection, and the app
  warns when the same MAT file is already open in another window.
- Added numbered window titles, tiling-friendly window sizing, and a file
  selection button that remains readable when resized.
- Added startup checks for compatible Windows app updates. If an update cannot
  be applied safely, the app still opens normally.
- Added optional sDREAMER support through a separate `torch.zip` download while
  keeping the main Windows package smaller and ready for the statistical model.

## v0.16.4.post1

- Added an initial Windows startup check for compatible app updates.

## v0.16.4

- When scoring is incomplete, Save Annotations now reports the first unscored
  range with its start time, end time, and duration—even if the save dialog is
  canceled.

## v0.16.3

- Added support for MAT files that use `fp_frequency` as the NE or
  fiber-photometry sampling-rate field.
- Saving annotations preserves the original sampling-rate field names.

## v0.16.2

- Improved video alignment for recordings with positive or negative video
  offsets.
- Added clear messages when the selected EEG range falls outside the available
  video.

## v0.16.1

- Added right-click selection of contiguous scored or unscored segments in
  annotation mode.
- Improved navigation and annotation auto-panning for large recordings.

## v0.16.0

- Added a statistical Wake/REM scoring option alongside sDREAMER. Select the
  model with `SLEEP_SCORING_MODEL` in `app_src/config.py`.
- Added configurable theta/delta trace color and opacity. The default is now a
  less visually intrusive semi-transparent black line.

## v0.15.5

- Fixed EEG spectrogram alignment for non-integer sampling rates.
- Fixed EEG and EMG time axes so traces use their true sample times rather than
  a rounded recording duration.

## v0.15.1

- Improved annotation speed.
- Simplified Undo to one step.
- Added the video filename to the video window.
- Added macOS support, tested on macOS Tahoe 26.0.1.
- Automatic scoring now updates sleep scores without redrawing the entire
  figure.
- Prevented the graph area from collapsing when a different MAT file is opened.

## v0.15.0-dev

- File selection now uses the desktop file dialog, reducing the wait when
  opening large recordings.
- Improved app responsiveness and added figure customization options.

## v0.14.0-beta

- Recordings now open directly in the visualization view. Automatic scoring is
  available from **Generate Predictions** in Annotation mode when an optional
  prediction model is installed.
- The Windows app can be used without a prediction model. To enable automatic
  scoring, place the optional `torch` folder inside `_internal`.
- Added precise annotation at zoomed-out scales: click a point in Annotation
  mode to select a narrow region whose width adapts to the zoom level.
- Integrated the spectrogram as the first subplot so it pans and zooms with the
  other signals, and removed the Confidence subplot.
- Added a button for selecting a different video for the current MAT file.
- Improved video playback speed and reliability, and added MP4 support.
- Added recovery of unsaved annotations after an unexpected shutdown. Recovery
  requires opening the same MAT file first on the next launch.
- Renamed the SWS label to NREM.

## v0.13.0

- Added video checking.
- Corrected the theta/delta ratio orientation.

## v0.12.1

- Added a **Show/Hide Spectrogram** button.
- Added the `m` keyboard shortcut for switching between Pan and Annotation
  modes.

## v0.12.0

- Added a spectrogram and theta/delta ratio above the signal plots.

## v0.12.0-dev

- Added an sDREAMER model specialized for recordings with NE data; the app
  selects it automatically when NE is available.
- Adjusted optional postprocessing to retain more detected REM bouts.
- Added a figure-title indicator showing which model and postprocessing setting
  were used.

## v0.11.0

- Added an EEG spectral-density plot for selections of up to 300 seconds.
- Improved support for large recordings and added a loading progress indicator.
- Added hour markers and hour-based hover times to the recording timeline.
- Preserved the true recording start time for MAT files that begin after the
  first source-data bin.

## v0.10.1

- Fixed spreadsheet-export errors when the entire recording contains only one
  predicted sleep stage.
- Enabled sleep-bout spreadsheet exports for manually scored recordings.
- Windows installations can receive smaller app updates without downloading a
  complete package each time.

## v0.10.0

- Added optional postprocessing to correct unlikely REM transitions and very
  short bouts.
- Prediction results now open immediately in an interactive view for review,
  annotation, and saving.
- Saving annotations now creates a spreadsheet containing chronological sleep
  bouts plus summary and transition statistics.
- EEG, EMG, and NE signals are aligned to their shared available duration.
- Fixed errors when annotating beyond the recording or saving with unscored
  sections.
- Fixed sleep scores disappearing after the sampling level changes.
- Added one-second boundaries to EEG, EMG, and NE signals.
- Removed the MSDA model.

## v0.9.0-beta

- Updated the sDREAMER model for improved scoring performance. See the
  [model notes](https://docs.google.com/document/d/1pj3fm7cJ2eW6XDKuYYW0IRGrYlmgvDCexqE3yngLl4c/edit?usp=sharing).

## v0.8.0

- Fixed signal alignment for recordings with non-integer sampling rates.
- Moved the filename into the figure title.
- Added compatibility with flattened signals and pre-downsampled NE data from
  the updated preprocessing workflow.
- Updated sleep-stage colors to match the Viewpoint app.

## v0.7.2

- Added visualization for unscored and manually scored recordings. MAT files
  without sleep scores open with Wake as the initial label; files without
  confidence values open with zero confidence.

## v0.7.0

- Annotation selections now clear after scoring, making it faster to create
  smaller selections in the same area.
- Simplified hover information and made it semi-transparent so it blocks less
  of the plot.
- Removed the redundant Autoscale button; use **Reset Axes** instead.
- Added a save reminder when exiting the app.
- Added automatic EEG and EMG resampling so models can handle recordings above
  512 Hz, including 610 Hz data.
- NE signals are now displayed at 10 Hz.
- Added visible time divisions to the confidence heatmap.
- Improved annotation responsiveness.

## v0.6.0

- Added a three-class scoring model for recordings without NE data.
- Increased the initial visualization sampling level to 4,000 points, with an
  optional 16,000-point view.
- Corrected one-second annotations so the second containing most of the
  selection receives the chosen score.
- Changed annotation shortcuts from `0`–`3` to `1`–`4`.
- Added left- and right-arrow panning in Annotation mode.
- Changed the MA stage color to yellow.

## v0.5.4

- Added Micro-Arousal (MA) annotation.
- Time axes now display complete second values instead of abbreviated values
  such as `16.455K`.
- Added a visible filename reminder while reviewing prediction results.
- Fixed Undo consuming its single history step when an annotation did not
  actually change the prediction.
