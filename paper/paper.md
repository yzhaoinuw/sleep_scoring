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
while changing the rest of its predictions as little as possible.
Completed scores are saved back
to the recording and exported as bout tables with stage and transition
statistics.

# Statement of need

Sleep Scoring App was developed for researchers studying how brain state
regulates cerebrospinal-fluid transport during sleep, in
[Project 2](https://www.urmc.rochester.edu/research/u19/project-2) of the
NIH BRAIN Initiative U19 program at the University of Rochester. That work
relates sleep states to cortical NE dynamics, which oscillate during NREM
sleep and shape sleep architecture [@kjaerby2022norepinephrine] and drive
the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].
<!-- TODO(authors): confirm with the Kjaerby/Hauglund co-authors that these
two citations frame the program fairly. Neither study's scoring used this
application: Kjaerby et al. scored in Viewpoint's SleepScore (Methods), and
users report no publications from the app's scores yet (2026-10-15). -->
Such questions depend on brief events, such as microarousals and state
transitions, that span a few seconds. Scoring them means reading the same
short interval against several physiological signals and the animal's
behavior, and correcting boundaries across multi-hour recordings; without an
integrated tool, this means moving among a signal viewer, predicted scores,
and separate video software. One-second annotation is a design requirement
of this research, not a claim that physiological boundaries or model
estimates are accurate to one second. Sleep Scoring App is intended for
experimenters scoring rodent EEG/EMG, particularly with aligned NE
photometry and behavior video.

The current version takes as input a MATLAB file produced from the raw
recordings by a companion preprocessing pipeline,
[`preprocess_sleep_data`](https://github.com/yzhaoinuw/preprocess_sleep_data).
EEG and EMG are recorded by the same device and share a sampling rate, while
the optional photometry signal comes from a separate device and usually has
its own. Other laboratories may need a thin adapter to this file layout
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
workspace for labeling and correcting scores, in which NE photometry informs
scoring and video checks are tied to the
selected interval. Project 2 previously used Viewpoint's commercial
[SleepScore](https://www.viewpoint.fr/product/rodent/sleep-monitoring/sleepscore), which did not integrate NE
photometry well; that gap motivated a dedicated application.
<!-- History from the Project 2 users (2026-10-14); they confirm the wording
in the full-draft review. SleepScore supports synchronized video, so the
stated gap is NE photometry integration only. -->


# Software design

Built with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly], the application is a local
web page shown in a native window via
[pywebview](https://pywebview.flowrl.com), backed by a server on the same
computer. Recordings open by path through native file
dialogs and never leave the machine. In
what follows, "interface" means this embedded web page and its JavaScript,
and "server" the local Python process. One principle runs through the
design: anything that responds directly to the user's hand runs in the
interface, because each trip to the server adds delay, and the server is
asked only for signal data the interface does not yet have. The design is
organized around five tasks: (1) inspecting and correcting at the needed
scale, (2) automatic scoring that keeps and learns from expert labels, (3)
checking behavior against video, (4) protecting and completing work, and (5)
installing and updating.

## 1. Inspecting and correcting at the needed scale

The EEG spectrogram (with a theta/delta ratio), EEG, EMG, and NE share one
time axis, and the score is drawn as an overlay on every signal row
(\autoref{fig:adaptation}A), so a boundary can be read against each
physiological signal. Moving through and labeling this view rests on
two modes, keyboard panning, three forms of selection, and one labeling
step.

The mouse does different jobs when navigating and when annotating, so these
are separate modes, switched with one key. In navigation mode, dragging pans
the view in large strides and scrolling zooms. In annotation mode the mouse
selects intervals instead, so the left and right arrow keys pan at an
adaptive pace that scales with zoom.
Annotation mode offers three selections. A click selects a short interval
scaled to the current zoom, for ranges too narrow to drag at that scale; a
dragged box selects a wider range; and a right-click selects the whole
contiguous bout under the cursor. When a drag reaches the viewport edge, the
view pans automatically and newly revealed signal is streamed in, so long
selections keep boundary detail. All selections feed one labeling step:
keys `1`–`4` assign Wake, NREM, REM, or MA, and `0` clears.

Plotly alone becomes unresponsive when asked to draw the millions of samples
in a multi-hour recording, so the application decimates long signals on
demand with [Plotly Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]. Plotly Resampler computes what to draw
for a range, but its stock update path asks the server for new traces on
every zoom or pan event, which can fall behind continuous movement. The
application keeps that computation and replaces the update path with its
own, following the principle stated in [Software design](#software-design).
Zooming, panning, selecting, and labeling are handled by JavaScript in the
interface. While the user zooms or
pans continuously, the application waits for the movement to stop or pause
before requesting new traces, and drops requests already overtaken by newer
movement. Only the changed trace data are then sent and swapped in, rather
than the whole figure being redrawn. Applying a label repaints the overlay
with no server round trip at all. The result is smoother navigation with
less lag on multi-hour recordings. In manual timing tests on a 2.86-hour recording with v0.17.4,
the median time from a navigation request to refreshed traces was 212 ms for
mouse panning and 329 ms for keyboard navigation on a Windows laptop, and 64
and 181 ms on an Apple M4 laptop; much of the keyboard time is deliberate
waiting for repeated key presses to settle. These are local instrumented
measurements, not a controlled benchmark.
<!-- Source: paper/latency_measurements.md (v0.17.4, 2026-10-08/09; same
recording at Sampling Level x1; first three updates per task excluded;
34–69 updates per task at 256- to 380-second views; 95th percentiles
Windows 371/239 ms, Mac 191/69 ms for keyboard/mouse). Windows and Mac runs
differ in machine and viewport; do not frame the difference as a speedup or
platform ranking. -->

This interaction design is not specific to sleep. We extracted it into a
runnable template for general multichannel time series
[@timeseries_app_cookbook]; adapting it mainly means editing one
data-loading function and one label configuration.

## 2. Automatic scoring that keeps and learns from expert labels

The application keeps user labels in a per-second array separate from the
displayed scores and places them over every prediction, so both scoring
models described below keep them as-is.

The default scoring model is a rule-based algorithm. It identifies
Wake from a normalized 1–7 Hz EEG spectral feature, keeping Wake bouts of
at least 5 s by default, and relabels candidate bouts of at least 30 s as
REM when NE falls below
recording-wide and within-bout percentile thresholds; without usable NE, it
identifies only Wake and NREM. Each run treats user-labeled Wake, NREM, and
REM intervals (a single labeled interval is enough) as ground truth
(\autoref{fig:adaptation}) and searches fixed candidate values for five
parameters: the Wake threshold, the minimum Wake and REM durations, and the
two NE percentiles. Each candidate is scored by its average error across the
labeled stages plus a penalty for changing unlabeled predictions. MA labels
are kept but not used for fitting. Adaptation is per recording, because
recordings vary widely, and needs no baseline built from many labeled
recordings. Fitting examples does not
establish held-out accuracy.

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

## 3. Checking behavior against video

For a selected interval of up to five minutes, the application cuts the
matching clip from the behavior video with [FFmpeg](https://ffmpeg.org),
and plays it in the same window
(\autoref{fig:video}). The cap suits checks of events that span seconds and
keeps clip extraction quick.

![**Selection-linked behavioral review.** **(A)** A right-click selects the
bout at 9176–9272\ s, and Check Video (outlined) opens its clip. **(B)** The
96-second clip plays beside the signals.\label{fig:video}](figures/figure2_video_review.png)

## 4. Protecting and completing work

Four further features protect and complete work: undo, crash recovery,
side-by-side windows, and checks on save. Undo reverts the most recent
change to the scores, and filesystem-backed recovery restores unsaved scores
when a file is reopened after a crash. Up to three isolated windows allow
side-by-side comparison of different recordings. Saving a complete
recording exports bout, stage, and transition statistics; otherwise, saving
reports the first unscored segment so the user can complete it.

## 5. Installing and updating

Windows users receive a packaged application that runs without a Python
installation; macOS users install from source. The packaged application
keeps itself current: it checks GitHub for a compatible release and applies
it before the window opens, keeping the user's edited settings, so fixes
reach users automatically without a reinstall. We extracted this updater into a reusable, dependency-free package,
[`desktop_app_source_updater`](https://github.com/yzhaoinuw/desktop_app_source_updater)
[@desktop_app_source_updater].

# Research impact statement

Researchers in Project 2 of the University of Rochester U19 program have been
the application's main users since 2023, using it to score and correct their
rodent EEG/EMG/NE recordings. The program's channel for the application has
about 30 members; four users who reported their use have together scored
about 1,000 hours of recordings, a lower bound on total use. Results scored
with the application have been presented in conference posters.
<!-- Source: replies from four users, 2026-10-15: ~300 h, ~300 h, ~250 h,
and 50-100 recordings at a typical ~3 h (150-300 h); about 1,000-1,150 h in
total. No publications or datasets from these scores yet, and no known users
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
