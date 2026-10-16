---
title: 'Sleep Scoring App: Interactive review and correction of rodent sleep annotations with synchronized electrophysiology, photometry, and behavior video'
tags:
  - Python
  - sleep
  - sleep scoring
  - electroencephalography
  - EEG
  - EMG
  - fiber photometry
  - norepinephrine
  - rodent
  - neuroscience
  - annotation
authors:
  - name: Yue Zhao
    orcid: 0000-0002-0819-5012
    corresponding: true
    affiliation: 1
  # TODO: add co-authors only if they meet the JOSS authorship bar (code, or
  # active project direction); supplying data alone goes in Acknowledgments.
affiliations:
  - name: University of Rochester, Rochester, NY, USA   # TODO: confirm/expand
    index: 1
date: 14 October 2026
bibliography: paper.bib
---

# Summary

Sleep Scoring App
([`sleep_scoring`](https://github.com/yzhaoinuw/sleep_scoring)) is a local
desktop application for sleep scoring rodent recordings: labeling each
second as wakefulness (Wake), non-rapid-eye-movement (NREM) sleep, REM
sleep, or a brief microarousal (MA), from brain electrical activity (EEG)
and muscle activity (EMG). It
brings EEG, EMG, an EEG spectrogram, an optional fiber-photometry
norepinephrine (NE) signal, and behavior-video clips of a selected interval
into one shared scoring workflow. Users can label intervals manually or
start from sleep scores predicted by an automatic scoring model, then revise
individual seconds or whole bouts; predicted sleep scores never overwrite
the user's own labels. The
default scoring model is a rule-based algorithm that adapts to
expert-labeled examples: it fits the user's labels on the current recording
while changing its other predictions as little as possible.
Completed scores are saved back
to the recording and exported as bout tables with stage and transition
statistics.

# Statement of need

Sleep Scoring App was developed for researchers studying how brain state
regulates cerebrospinal-fluid transport during sleep, in
[Project 2](https://www.urmc.rochester.edu/research/u19/project-2) of the
BrainFlowZZZ program at the University of Rochester. That work
relates sleep states to cortical NE dynamics, which oscillate during NREM
sleep and shape sleep architecture [@kjaerby2022norepinephrine] and may drive
the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].
<!-- TODO(authors): confirm with the Kjaerby/Hauglund co-authors that these
two citations frame the program fairly. Neither study's scoring used this
application: Kjaerby et al. scored in Viewpoint's SleepScore (Methods), and
users report no publications from the app's scores yet (2026-10-15). -->
Such questions depend on brief events, such as microarousals and state
transitions, that span a few seconds. Scoring them means examining the same
short interval across several physiological signals and the animal's
movement, and correcting boundaries across multi-hour recordings; without an
integrated tool, this means moving among a signal viewer, predicted scores,
and separate video software. One-second annotation is a requirement of this
research, not a claim that boundaries or model estimates are accurate to
one second. Sleep Scoring App is intended for
experimenters scoring rodent EEG/EMG, particularly with aligned NE
photometry and behavior video.

The current version takes as input a MATLAB file produced from the raw
recordings by a companion preprocessing pipeline,
[`preprocess_sleep_data`](https://github.com/yzhaoinuw/preprocess_sleep_data).
Other laboratories may need a thin adapter to this file layout
until later versions accept more general input formats.

# State of the field

Several open tools support rodent sleep scoring.
[AccuSleep](https://github.com/zekebarger/AccuSleep) and its Python
successor [AccuSleePy](https://github.com/zekebarger/AccuSleePy)
[@barger2019accusleep] pair a manual-labeling interface with a
neural-network scoring model. [SPINDLE](https://sleeplearning.ethz.ch)
[@miladinovic2019spindle] provides end-to-end learned scoring across
laboratories and species.
[Somnotate](https://github.com/paulbrodersen/somnotate)
[@brodersen2024somnotate] scores with linear discriminant analysis and a
hidden Markov model. [Visbrain Sleep](https://github.com/EtienneCmb/visbrain)
[@combrisson2019visbrain] is a general polysomnography viewer and hypnogram
editor.
<!-- TODO(verify): confirm each characterization against the cited paper and
current documentation, including epoch-length options, and add commercial
packages only with a citable source. As of 2026-10-07 the AccuSleep,
AccuSleePy, and somnotate documentation describe EEG/EMG(/LFP) inputs and do
not mention photometry or video. Scoring methods checked 2026-10-14:
AccuSleep CNN on EEG/EMG (PLOS ONE 2019); SPINDLE CNN+HMM on 2 EEG + 1 EMG,
4 s epochs (PLoS Comput Biol 2019); somnotate LDA+HMM on EEG/LFP/EMG
spectrograms (repo README); Visbrain detects spindles/K-complexes/slow
waves/REMs but does not stage. Viewpoint SleepScore uses an undisclosed
"probability method" on EEG/EMG (product page); check its manual before
extending the NE claim to it. sDREAMER's NE-aware variant does use NE, so
keep the claim scoped to the tools listed here. -->

These tools center on electrophysiology and, for most, on automatic sleep
scoring, which draws on EEG, EMG, or LFP features alone. None of their
scoring methods uses NE photometry, although NE dynamics have been proposed
as markers for distinguishing wake, microarousals, NREM, and REM sleep
[@andersen2023photometry]. Beyond automatic scoring, our work needed a
workspace for labeling and correcting scores, in which NE photometry and
video checks inform scoring. Project 2 previously used Viewpoint's commercial
[SleepScore](https://www.viewpoint.fr/product/rodent/sleep-monitoring/sleepscore), which did not integrate NE
photometry well; that gap motivated a dedicated application.
<!-- History from the Project 2 users (2026-10-14); they confirm the wording
in the full-draft review. SleepScore supports synchronized video, so the
stated gap is NE photometry integration only. -->


# Software design

The application is designed around five tasks: (1) zero-setup installation
and automatic updates, (2) visualizing and annotating at scale, (3)
automatic scoring that keeps and learns from expert labels, (4) checking
movement against video, and (5) saving, recovery, and export. Smooth
interactive visualization is central to these tasks, so the application is
built with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly]: a local web page, shown in a
native window via [pywebview](https://pywebview.flowrl.com) and backed by a
Python server on the same computer. In what follows, "interface" means this
embedded web page and its JavaScript, and "server" the local Python process.

## 1. Zero-setup installation and automatic updates

Many experimenters do not program, so we packaged the Windows application
with [PyInstaller](https://pyinstaller.org) to install like conventional
desktop software: users download a release, extract it, and double-click to
open it, with no Python installation, environment setup, or command line.
Our first users ran Windows; macOS support came later, and because a
comparable package would need Apple's code signing and notarization, macOS
users currently install from source. Packaging alone would have required a
rebuild and a fresh download for every change, so we kept the application's
own source code outside the packaged runtime. The packaged application
therefore updates itself: at startup it applies bug fixes, new features, and
other updates, keeping the user's edited settings.[^deps] We extracted this
updater into a reusable, dependency-free package,
[`desktop_app_source_updater`](https://github.com/yzhaoinuw/desktop_app_source_updater)
[@desktop_app_source_updater].

[^deps]: Updates that change the packaged dependencies still require
downloading a new release.

## 2. Visualizing and annotating at scale

The EEG spectrogram (with a theta/delta ratio), EEG, EMG, and NE share one
time axis, and the score is drawn as an overlay on every signal row
(\autoref{fig:adaptation}A), so each signal can be used on its own to
examine or label sleep states. We built shortcut gestures so that navigation
and annotation take as few steps as possible. For example, a click selects
an interval too narrow to drag at the current zoom, a right-click selects a
whole contiguous bout, and a drag that reaches the edge of a zoomed-in view
keeps panning, so a selection can extend arbitrarily far without losing
detail. A selection is then labeled with a single key press.

In a Dash application, the interface draws the figures while the data and
computation stay on the server: each interaction is sent to the server,
which returns updated figure data. Plotly alone becomes unresponsive when
asked to draw the millions of samples in a multi-hour recording, so the
application decimates long signals on demand with [Plotly
Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]: the server keeps the full-resolution
signals and sends the interface only enough points to draw the visible
range, so zooming in or panning needs new data from the server. In feedback
sessions, users of earlier versions experienced lags of one to two seconds
while navigating, which disrupted scoring. The library's stock update path
requests that data on every zoom or pan event, which can fall behind
continuous movement. We replaced that path, following one principle for
responsiveness: anything that responds directly to the user's hand runs in
the interface, because each round trip to the server adds delay, and the
server is asked only for signal data the interface does not yet have.
Zooming, panning, selecting, and labeling therefore run in the interface.
While the user zooms or pans continuously, the application waits for the
movement to pause before requesting new traces, drops requests already
overtaken by newer movement, and swaps in only the changed trace data rather
than redrawing the figure. Applying a label needs no server round trip. In
manual timing tests on a 2.86-hour recording with v0.17.4, the median time
from a navigation request to refreshed traces was 212 ms for mouse panning
and 329 ms for keyboard navigation on a Windows laptop, and 64 and 181 ms on
an Apple M4 laptop; much of the keyboard time is deliberate waiting for
repeated key presses to settle. On the same Windows laptop, an earlier
version of this path took about 935 ms. None of these is a controlled
benchmark.
<!-- Source: paper/latency_measurements.md (v0.17.4, 2026-10-08/09; same
recording at Sampling Level x1; first three updates per task excluded;
34–69 updates per task at 256- to 380-second views; 95th percentiles
Windows 371/239 ms, Mac 191/69 ms for keyboard/mouse). Windows and Mac runs
differ in machine and viewport; do not frame the difference as a speedup or
platform ranking. The 935 ms figure is from
ui_response_time_optimization_progress.txt (May 2026, Windows i9 laptop): an
earlier app version, not stock Plotly Resampler, under unmatched conditions;
keep it framed as development history, not a before/after benchmark. The
one-to-two-second lags were seen by the maintainer while watching users
navigate over screen share in feedback meetings (no written record). -->

This interaction design is not specific to sleep; we extracted it into a
runnable template for general multichannel time series
[@timeseries_app_cookbook].

## 3. Automatic scoring that keeps and learns from expert labels

The application keeps user labels in a per-second array separate from the
displayed scores and places them over every prediction, so both scoring
models keep them as-is.

We implemented the default scoring model as a rule-based algorithm that
encodes the stage criteria used by Project 2 researchers
[@kjaerby2022norepinephrine]: it identifies Wake from EEG spectral power and
REM from sustained low NE, and without usable NE it identifies only Wake and
NREM. Its thresholds adapt to each recording.
Each run treats the user's labels (a single labeled interval is enough) as
ground truth and chooses the settings that best fit them while changing the
rest of the predictions as little as possible (\autoref{fig:adaptation}).
Adaptation therefore needs no baseline built from many labeled recordings.
Agreement with the user's labels shows only that the fit succeeded, not how
accurate the predictions are elsewhere in the recording.
<!-- Rule origin: Project 2 researchers agreed the rules as a group. Kjaerby
et al. (Methods) define Wake/NREM/REM from EEG and EMG in 1 s epochs and MA
as wake bouts under 15 s, and note NE is lowest in REM; the app's numeric
thresholds are not in that paper, so keep the citation at the level of
stage criteria. Detail deliberately left to the README (2026-10-15): Wake
uses a normalized 1-7 Hz EEG feature; REM needs bouts of at least 30 s
below recording-wide and within-bout NE percentiles; five settings are
fitted (Wake threshold, minimum Wake and REM durations, two NE percentiles),
scored by average per-stage error plus a penalty for changing unlabeled
predictions; MA labels are kept but not fitted. -->

Alternatively, the deep-learning model sDREAMER [@chen2023sdreamer], in a
version adapted by the author
([`sdreamer_flow`](https://github.com/yzhaoinuw/sdreamer_flow)), runs when
the optional [PyTorch](https://pytorch.org) [@paszke2019pytorch]
dependencies are installed; it does not adapt to user labels but keeps them
in the same way.

![**Expert-guided adaptation.** One interval shown three times: **(A)**
predicted stages over the spectrogram, EEG, EMG, and NE; **(B)** a
50-second expert NREM correction; **(C)** predictions after calibration on
228 s of expert labels (178 s Wake outside this view, 50 s NREM), which
change beyond the outlined interval.\label{fig:adaptation}](figures/figure1_adaptation.png)

## 4. Checking movement against video

EMG shows the muscle activity that helps distinguish sleep states, and the
behavior video can corroborate it. For a selected interval of up to five
minutes, enough for events that span seconds and quick to extract, the
application cuts the matching clip from the video with
[FFmpeg](https://ffmpeg.org) and plays it in the same window
(\autoref{fig:video}).

![**Selection-linked video review.** **(A)** A right-click selects the
bout at 9176–9272\ s, and Check Video (outlined) opens its clip. **(B)** The
96-second clip plays beside the signals.\label{fig:video}](figures/figure2_video_review.png)

## 5. Saving, recovery, and export

Undo reverts the most recent change to the scores, and filesystem-backed
recovery restores unsaved scores when a file is reopened after a crash.
Saving a complete recording exports bout, stage, and transition statistics;
otherwise, saving reports the first unscored segment so the user can
complete it.

# Research impact statement

Researchers in Project 2 of the University of Rochester BrainFlowZZZ program
have been the application's main users since 2023, using it to score their
rodent EEG/EMG/NE recordings. The program's channel for the application has
about 30 members; five users who reported their use have together scored
about 1,200 hours of recordings, a lower bound on total use. Results scored
with the application have been presented in conference posters.
<!-- Source: replies from five users, 2026-10-15/16: ~300 h, ~300 h, ~250 h,
~200 h, and 50-100 recordings at a typical ~3 h (150-300 h); about
1,200-1,350 h in total. No publications or datasets from these scores yet, and no known users
outside the program. TODO(authors): name studies, preprints, or datasets
scored with the app once published. -->
The repository provides releases, documentation, tests, demonstration
videos, and archived versions [@sleepscoring_zenodo].

# AI usage disclosure

From 2023 through 2025, the author asked web-based ChatGPT and Claude
coding questions (model versions not recorded). Since 2026, coding agents
have assisted with code, tests, documentation, and release checks: Claude
Code (Claude Opus 4.5–5.5 and Fable 5) and OpenAI Codex (GPT-5 and GPT-6),
with sessions recorded in the repository's work log (`treaty_docs/`), kept
under the author's
[Agent Collab Treaty](https://github.com/yzhaoinuw/agent_collab_treaty)
conventions for human–agent collaboration. Claude Code and Codex
also helped plan, draft, and review this manuscript. The author made the
architectural and design decisions, reviewed and tested all AI-assisted code
before release, and reviewed, revised, and verified the manuscript against
the software and cited sources.
<!-- TODO(author): confirm review is complete before submission. -->

# Acknowledgments

This work was supported by the BRAIN Initiative of the US National Institutes
of Health (U19NS128613).

<!-- TODO: thank the lab PI, data contributors, and model contributors by
name. The funding sentence above is the PI-approved wording and should be
kept verbatim. -->

# References
