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
the optional PyTorch dependency is installed, the app can hand a recording to
the externally-developed sDREAMER transformer model
[@TODO_sdreamer_citation] for automatic scoring; the same UI is then used to
inspect and correct the predicted labels. Finished annotations are written back
into the original `.mat` file, and per-bout and per-stage summary statistics are
exported to Excel. The application is distributed as a single-folder
PyInstaller bundle for Windows and installs from source via `pip` on macOS.

# Statement of need

`sleep_scoring` was developed for the U19 BrainFlowZZZ research program, which
studies noradrenergic regulation of sleep using simultaneous EEG, EMG, and
fiber-photometry recordings of cortical norepinephrine (NE) in mice.
Recordings in this program are acquired with Viewpoint (EEG/EMG) and TDT
(NE photometry) hardware, preprocessed by a companion pipeline into MATLAB
`.mat` files with a fixed field layout, and scored at 1-second epochs to
match the temporal resolution at which the program reasons about NE
dynamics—a finer granularity than the 4- to 10-second epochs assumed by most
existing rodent sleep scoring tools. Manual scoring at this resolution is
correspondingly slow, and the existing software landscape did not fit the
program's needs end-to-end.

Commercial packages such as SleepSign and Neuroscore are closed-source,
vendor-locked to specific acquisition hardware, and do not natively accept an
NE channel. Open-source alternatives—including AccuSleep
[@barger2019accusleep], SPINDLE [@miladinovic2019spindle], somnotate
[@miladinovic2022somnotate], Visbrain Sleep [@combrisson2019visbrain], and
SleepEEGpy—variously provide either an interactive viewer or an automated
classifier, but rarely both in one workflow, and none accept NE as an input
feature.

`sleep_scoring` is therefore opinionated about its input format and epoch
length, and external adopters will, at present, need recordings shaped to
match the BrainFlowZZZ preprocessing pipeline (or a thin adapter that
produces the same `.mat` field layout). Beyond that constraint, the
contributions intended to be reusable outside the program are (i) the
interactive Dash/Plotly + `pywebview` annotation UI, including its
keyboard-driven epoch selection, contiguous-segment selection, and
undo/crash-recovery behavior; and (ii) the integrated behavior-video clip
extraction synchronized to the annotation selection. The sDREAMER scorer
itself is developed and published separately within the same research
program [@TODO_sdreamer_citation]; this app integrates both its EEG/EMG-only
and NE-aware variants behind a one-button prediction flow, with a small set
of rule-based post-processing heuristics layered on top, but does not claim
the model as its own contribution. The result is a tight predict →
audit → correct → export loop that a non-programmer experimenter in the
program can complete on a recording in one sitting, and that we expect other
rodent sleep groups running comparable polysomnography plus NE photometry
experiments to find useful as a starting point.

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
to one of the two sDREAMER variants—EEG/EMG-only or NE-aware—based on the
contents of the file. Predictions returned by the upstream model are then
post-processed by a small set of rule-based cleanup heuristics that remove
physiologically implausible short bouts and validate REM transitions,
optionally using the NE signal as additional evidence. On save, the resulting bout table and per-stage summary statistics
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
[@mckinney2010pandas]. The upstream sDREAMER model
[@TODO_sdreamer_citation] is vendored under `models/sdreamer/` and loaded via
PyTorch [@paszke2019pytorch] using `timm` and `einops` when the optional `ml`
dependencies are installed; the app itself adds only the prediction routing,
post-processing heuristics, and integration with the annotation UI. The repository ships with a
`pytest` suite covering preprocessing, postprocessing, FFT helpers, and
app-level utilities, and a representative set of sample `.mat` files for
manual testing.

# Acknowledgments

We thank <!-- TODO: list lab PI, data contributors, model contributors,
funding sources, and grant numbers --> for their support during the
development of this software.

# References
