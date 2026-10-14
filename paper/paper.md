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
desktop application for *sleep scoring* rodent recordings: labeling each
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
two citations frame the program fairly, and whether either study's scoring
used this application. -->
Such questions depend on brief events, such as microarousals and
state transitions, that span a few seconds. Scoring them requires reading
the same short interval against several physiological signals and the
animal's behavior, and correcting
boundaries, often across multi-hour recordings.

Without an integrated tool, this review means moving among a signal viewer,
predicted sleep scores, and separate video software, and redrawing boundaries
that a scoring model or a previous pass got wrong. Sleep Scoring App is
intended for experimenters scoring rodent EEG/EMG, particularly those with
aligned NE photometry and behavior video.

Its input contract is deliberately narrow. Recordings are MATLAB files with
a documented field layout, produced by a companion preprocessing pipeline,
[`preprocess_sleep_data`](https://github.com/yzhaoinuw/preprocess_sleep_data);
EEG and EMG share a sampling rate and the optional photometry signal carries
its own. Other laboratories need a thin adapter to that layout; the
application is not a general acquisition-format importer. One-second
annotation is a design requirement of the motivating research, not a claim
that physiological boundaries or model estimates are accurate to one second.

# State of the field

Several open tools support rodent sleep scoring.
[AccuSleep](https://github.com/zekebarger/AccuSleep) and its Python
successor [AccuSleePy](https://github.com/zekebarger/AccuSleePy)
[@barger2019accusleep] combine a manual-labeling
interface with a neural-network classifier for EEG/EMG and configurable
brain states. [SPINDLE](https://sleeplearning.ethz.ch)
[@miladinovic2019spindle] provides end-to-end learned
scoring across laboratories and species.
[Somnotate](https://github.com/paulbrodersen/somnotate)
[@brodersen2024somnotate] scores vigilance states with linear
discriminant analysis and a hidden Markov model, with simple interfaces for
refining annotations.
[Visbrain Sleep](https://github.com/EtienneCmb/visbrain)
[@combrisson2019visbrain] offers a
general hypnogram viewer and editor for polysomnography.
<!-- TODO(verify): confirm each characterization against the cited paper and
current documentation, including epoch-length options, and add commercial
packages only with a citable source. As of 2026-10-07 the AccuSleep,
AccuSleePy, and somnotate documentation describe EEG/EMG(/LFP) inputs and do
not mention photometry or video. -->

These tools center on electrophysiology and, for most, on automatic sleep
scoring. Beyond automatic scoring, our work needed a workspace for labeling
and correcting scores, in which NE photometry informs scoring rather than
being an extra trace, video checks are tied to the selected interval, and
predicted sleep scores keep the user's labels as-is. Project 2 researchers previously scored with Viewpoint's
commercial [SleepScore](https://www.viewpoint.fr/product/rodent/sleep-monitoring/sleepscore),
which did not integrate fiber-photometry signals such as NE well; that gap
motivated a dedicated application, whose interaction design is documented
for reuse.
<!-- History from the Project 2 users (2026-10-14); they confirm the wording
in the full-draft review. SleepScore supports synchronized video, so the
stated gap is NE photometry integration only. -->


# Software design

Built with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly], the application runs as a
desktop program: its interface is a web page rendered inside a
native window (via [pywebview](https://pywebview.flowrl.com)) by the operating system's embedded web
engine, WebView2 on Windows or WebKit on macOS, and talks to a server
running on the same computer. No internet connection is needed: recordings
open by path through native file dialogs and never leave the machine. The
only network use is an optional update check and opt-in usage reporting. In
what follows, "interface" means this embedded web page and its JavaScript,
and "server" the local Python process. The design is organized around four
tasks: (1) inspecting and correcting at the needed scale, (2) automatic
scoring that keeps and learns from expert labels, (3) checking behavior
against video, and (4) protecting and completing work.

## 1. Inspecting and correcting at the needed scale

The EEG spectrogram (with a theta/delta ratio), EEG, EMG, and NE share one
time axis, and the score is drawn as an overlay on every signal row
(\autoref{fig:adaptation}A), so a boundary can be read against
each physiological signal. Users switch between navigation and annotation with
one key. Selections can be a zoom-adaptive click, a dragged box, or a
right-click that selects the whole contiguous bout under the cursor; when a
drag reaches the viewport edge, the view pans automatically and newly
revealed signal is streamed in, so long selections keep boundary detail.
All selection forms feed one labeling step: keys `1`–`4` assign Wake,
NREM, REM, or MA, and `0` clears.

Drawing every sample of a multi-hour recording would stall the interface, so
long signals are decimated on demand by
[Plotly Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]. The application uses that library only
to compute what to draw for a range, and replaces its stock update path with
its own, which keeps interaction state in the interface and full-resolution
data on the server. Gestures, panning, and labeling run as JavaScript in the
interface. Navigation events are coalesced, so the server refreshes traces
after a gesture is released or pauses rather than on every frame, and
refreshes overtaken by newer navigation are discarded. Updates are applied as
trace patches rather than figure rebuilds. During a selection drag, auto-pan
instead refreshes newly revealed signal repeatedly through a dedicated
endpoint outside the Dash callback graph. Applying a label repaints the
overlay in the interface without a server round trip.

This interaction design does not depend on the sleep domain. We extracted it
into a separate, runnable template [@timeseries_app_cookbook] that applies
the same navigation, selection, and labeling design to general multichannel
time series, demonstrated on synthetic data; adapting it to a new signal
type centers on one data-loading function and one label configuration.

## 2. Automatic scoring that keeps and learns from expert labels

The application keeps user labels in their own per-second array, separate
from the displayed scores and empty wherever the user has not labeled. When
automatic scoring is confirmed, the app snapshots these labels and places
them over the model's predicted scores, so the user's labels are kept as-is
with either of the two scoring models described below.

The default scoring model is a rule-based algorithm. It identifies
Wake from a normalized 1–7 Hz EEG spectral feature with minimum bout
lengths, and relabels candidate bouts as REM when NE falls below
recording-wide and within-bout percentile thresholds; without usable NE, it
identifies only Wake and NREM. Each run treats user-labeled Wake, NREM, and
REM intervals (a single labeled interval is enough) as ground truth
(\autoref{fig:adaptation}) and searches fixed candidate values for five
parameters: the Wake threshold, the minimum Wake and REM durations, and the
two NE percentiles. Each candidate is scored by its average error across the
labeled stages plus a penalty for changing unlabeled predictions; ties go to
values nearest the defaults. The chosen values are reported. MA labels are
kept but not used for fitting. Adaptation applies to the current recording
only: recordings vary widely, and fitting each one directly avoids needing
many labeled recordings to establish a baseline. Fitting examples does not
establish held-out accuracy.

As an alternative, the application provides sDREAMER [@chen2023sdreamer], an
externally developed deep-learning sleep scoring model, which runs when the
optional [PyTorch](https://pytorch.org) [@paszke2019pytorch] dependencies are
installed. It is used as published, without adaptation, and its predicted
scores keep user labels in the same way.

![**Expert-guided adaptation of sleep-stage predictions.** All panels show the
same interval. **(A)** Predicted stages with the spectrogram, EEG, EMG, and NE
signals. **(B)** A 50-second expert NREM correction. **(C)** Predictions
after adaptive calibration using 228 s of expert labels
(178 s Wake, outside this view, and 50 s NREM); predictions change beyond the outlined labeled
interval.\label{fig:adaptation}](figures/figure1_adaptation.png)

## 3. Checking behavior against video

For a selected interval of up to five minutes, the application cuts the
matching clip from the behavior video with [FFmpeg](https://ffmpeg.org),
validates the recording-to-video offset, and plays it in the same window
(\autoref{fig:video}). The cap suits checks of events that span seconds and
keeps clip extraction quick.

![**Selection-linked behavioral review.** **(A)** A right-click selects the
contiguous bout at 9176–9272\ s, and Check Video (both outlined) opens its
clip. **(B)** The corresponding 96-second behavior clip, opened in the application for review alongside the physiological
signals.\label{fig:video}](figures/figure2_video_review.png)

## 4. Protecting and completing work

One-step undo and filesystem-backed recovery protect in-progress work; up
to three isolated windows allow side-by-side comparison, refusing the same
file twice. Saving reports the first unscored gap; a complete recording also
exports bout, stage, and transition statistics, including microarousals.

# Research impact statement

Researchers in Project 2 of the University of Rochester U19 program have been
the application's main users since 2023, using it to score and correct their
rodent EEG/EMG/NE recordings.
<!-- TODO(authors): add the users' rough estimate of total recording hours
scored. The opt-in usage counter (README badge) is recent and covers only
opted-in copies, so it undercounts total use. Name studies, preprints, or
datasets scored with the app once confirmed. -->
It integrates with a companion preprocessing pipeline that converts raw
acquisition output into its input format, and the repository provides
packaged Windows releases, macOS source installation, documentation, tests,
demonstration videos, and archived versions [@sleepscoring_zenodo].

Manual timing tests on a 2.86-hour recording show the interaction design is
practical on long data. With v0.17.4, the median time from the
application receiving a navigation request to the refreshed traces was 329 ms
for keyboard navigation and 212 ms for mouse panning on a Windows laptop
(95th percentiles 371 and 239 ms), and 181 and 64 ms on an Apple M4 laptop
(95th percentiles 191 and 69 ms), over 34–69 updates per task at 256- to
380-second views. Server work took under 13 ms of each update, and much of
the keyboard time is deliberate coalescing of repeated key presses. These are
local instrumented measurements rather than a controlled benchmark.
<!-- Source: paper/latency_measurements.md (v0.17.4, 2026-10-08/09; same
recording at Sampling Level x1; first three updates per task excluded).
Windows and Mac runs differ in machine and viewport; do not frame the
difference as a speedup or platform ranking. -->

# AI usage disclosure

From 2023 through 2025, the maintainer used the web versions of ChatGPT
(OpenAI) and Claude (Anthropic) for coding questions and suggestions while
developing the software; specific model versions were not recorded. Since
2026, AI coding agents working in the repository have assisted with code
generation, refactoring, tests, documentation, and release checks: Claude
Code (Claude Opus 4.5–5.5 and Fable 5) and OpenAI Codex (GPT-5 and GPT-6),
with sessions recorded in the repository's work log. Claude Code and Codex
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
