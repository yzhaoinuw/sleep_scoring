---
title: 'sleep_scoring: An interactive desktop application for rodent EEG/EMG sleep scoring with optional deep-learning-assisted prediction'
tags:
  - Python
  - sleep
  - electroencephalography
  - EEG
  - EMG
  - rodent
  - neuroscience
  - deep learning
  - annotation
  - norepinephrine
authors:
  - name: Yue Zhao
    orcid: 0000-0002-0819-5012
    corresponding: true
    affiliation: 1
  # TODO: add co-authors (lab members who contributed code, models, or data)
affiliations:
  - name: University of Rochester, Rochester, NY, USA   # TODO: confirm/expand
    index: 1
date: 2 June 2026
bibliography: paper.bib
---

# Summary

`sleep_scoring` is an open-source desktop application for visualizing, manually
annotating, and automatically scoring rodent sleep stages from
electroencephalogram (EEG), electromyogram (EMG), and—optionally—fiber-photometry
norepinephrine (NE) recordings. The software loads MATLAB `.mat` recordings,
presents a synchronized, interactive view of the EEG spectrogram, raw
EEG/EMG/NE traces, and any existing sleep scores, and lets a human scorer label
segments as Wake, Slow-Wave Sleep (SWS), or REM using keyboard shortcuts. When
the optional PyTorch dependency is installed, a built-in transformer model
(sDREAMER) can score a recording automatically; the same UI is then used to
inspect and correct the predicted labels. Finished annotations are written back
into the original `.mat` file, and per-bout and per-stage summary statistics are
exported to Excel. The application is distributed as a single-folder
PyInstaller bundle for Windows and installs from source via `pip` on macOS.

# Statement of need

Manual sleep scoring of rodent polysomnography is a standard step in many
systems-neuroscience experiments, but it is slow and tedious: a typical
12-hour recording requires several hours of expert annotation in short (1–4 s)
epochs. Existing tools fall into two broad camps. Commercial packages such as
SleepSign and Neuroscore offer well-designed interactive interfaces but are
closed-source, vendor-locked, and do not natively support newer fiber-photometry
signals such as NE. Open-source alternatives—including AccuSleep
[@barger2019accusleep], SPINDLE [@miladinovic2019spindle], somnotate
[@miladinovic2022somnotate], Visbrain Sleep [@combrisson2019visbrain], and
SleepEEGpy—variously provide either an interactive viewer or an automated
classifier, but rarely both in a single end-user workflow, and none of the
widely-used ones accept an NE channel as input.

`sleep_scoring` was developed to close this gap for a single laboratory's
day-to-day work and has since been adopted more broadly across a research
program studying noradrenergic regulation of sleep. It combines (i) an
interactive Dash/Plotly UI embedded in a `pywebview` desktop window, (ii) a
transformer-based automatic scorer (sDREAMER) with two variants—EEG/EMG-only
and EEG/EMG/NE—and (iii) on-demand playback of the behavior-video clip
corresponding to any selected time window, all in a single double-clickable
application that does not require its users to write code. The result is a
tight predict → audit → correct → export loop that a non-programmer experimenter
can complete on a recording in one sitting.

# Key features

The active runtime path centers on `run_desktop_app.py`, which starts a
background Dash server and opens the UI in a native window. The figure layer
renders a four-row, time-locked view: the EEG spectrogram with an overlaid
theta/delta ratio, the raw EEG trace, the EMG trace, and the NE trace when
present, each with a colored sleep-score heatmap rendered on top. The
`plotly-resampler` package [@vanderdonckt2022plotlyresampler] keeps interaction
responsive on multi-hour recordings.

Two interaction modes—navigation and annotation—are toggled with a single key
press. In annotation mode, clicking selects an epoch, dragging selects a wider
window with automatic horizontal scrolling at the viewport edge, and a right
click selects the entire contiguous segment under the cursor. Pressing `1`,
`2`, or `3` assigns Wake, SWS, or REM to the current selection. An undo stack
and a filesystem-backed crash-recovery cache protect in-progress work.

When PyTorch is available, a Generate Predictions button routes the recording
to either the EEG/EMG-only model or the NE-aware model based on the contents
of the file. Predictions are then post-processed by a small set of rule-based
cleanup heuristics that remove physiologically implausible short bouts and
validate REM transitions, optionally using the NE signal as additional
evidence. On save, the resulting bout table and per-stage summary statistics
are written to an Excel workbook alongside the updated `.mat`.

A short user-selected time window can be cut from the synchronized behavior
video on demand using a bundled `imageio-ffmpeg` binary, allowing scorers to
confirm ambiguous epochs against animal behavior without leaving the app.

# Implementation

The application is implemented in Python 3.11. The UI is built on Dash 3 with
`dash-bootstrap-components`, `dash-extensions`, and `dash-player`, and is
hosted in a `pywebview` window so that end users see a normal desktop
application rather than a browser tab. Signal processing relies on
`scipy.signal.ShortTimeFFT`, NumPy [@harris2020numpy], and pandas
[@mckinney2010pandas]. The deep-learning component is implemented in PyTorch
[@paszke2019pytorch] using `timm` and `einops`. The repository ships with a
`pytest` suite covering preprocessing, postprocessing, FFT helpers, and
app-level utilities, and a representative set of sample `.mat` files for
manual testing.

# Acknowledgments

We thank <!-- TODO: list lab PI, data contributors, model contributors,
funding sources, and grant numbers --> for their support during the
development of this software.

# References
