---
title: 'Sleep Scoring App: Interactive review and correction of rodent sleep annotations with synchronized electrophysiology, photometry, and behavior video'
# TODO(name): the paper now uses "Sleep Scoring App" as the software's name;
# the repository and package stay `sleep_scoring`. Align the CITATION.cff and
# Zenodo titles with this name before submission.
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
  # TODO: add co-authors (lab members who contributed code, models, or data)
affiliations:
  - name: University of Rochester, Rochester, NY, USA   # TODO: confirm/expand
    index: 1
date: 7 October 2026
bibliography: paper.bib
---

# Summary

Sleep research in rodents depends on *sleep scoring*: labeling each moment
of a recording as wakefulness, non-rapid-eye-movement (NREM) sleep, or REM
sleep, plus brief microarousals (MA), from brain electrical activity (EEG)
and muscle activity (EMG). Sleep Scoring App
([`sleep_scoring`](https://github.com/yzhaoinuw/sleep_scoring)) is a local desktop application
for reviewing and correcting these annotations at one-second resolution. It
brings EEG, EMG, an EEG spectrogram, an optional fiber-photometry
norepinephrine (NE) signal, and behavior-video clips of a selected interval
into one shared scoring workflow. Users can label intervals manually or
generate automatic scores, then revise individual seconds or whole bouts;
labels a user has supplied are kept as-is in every automatic result. The
default automatic scorer is a rule-based algorithm that adapts to
expert-labeled examples: it treats the user's labels on the current
recording as ground truth and adjusts its parameters to best match them.
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
Questions at this level depend on brief events, such as microarousals and
state transitions, that span a few seconds. Scoring them requires reading
the same short interval against several physiological signals and the
animal's behavior, and correcting
boundaries, often across multi-hour recordings.

Without an integrated tool, this review means moving among a signal viewer,
a classifier's output, and separate video software, and redrawing boundaries
that an automatic scorer or a previous pass got wrong. Sleep Scoring App is
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
[@brodersen2024somnotate] classifies vigilance states with linear
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

These tools center on electrophysiology and, for most, on classification.
Our need was the inverse emphasis: a correction workspace in which NE
photometry informs scoring rather than being an extra trace, video checks
are tied to the selected interval, and automatic scoring keeps user-supplied
labels as-is. Extending an existing tool would have meant changing its
central data model (a single label stream over EEG/EMG inputs) as well as
its interaction layer. A focused application was the smaller change, and its
interaction design is documented for reuse.
<!-- TODO(authors): this build-vs-contribute rationale is inferred from the
design, not from project history. Confirm or replace it with the actual
reasons a new application was built in 2023. -->


# Software design

The application is built with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly], web technologies,
but runs as a desktop program. Its interface is a web page rendered inside a
native window (via [pywebview](https://pywebview.flowrl.com)) by the operating system's embedded web
engine, WebView2 on Windows or WebKit on macOS, and talks to a server
running on the same computer. No internet connection is needed: recordings
open by path through native file dialogs and never leave the machine. The
only network use is an optional update check and opt-in usage reporting. In
what follows, "interface" means this embedded web page and its JavaScript,
and "server" the local Python process. The design is organized around three
tasks: (1) inspecting and correcting at the needed scale, (2) automatic
scoring that keeps and learns from expert labels, and (3) checking behavior
and completing a recording.

## 1. Inspecting and correcting at the needed scale

EEG spectrogram with a
theta/delta ratio, EEG, EMG, and NE share one time axis, and the score is
drawn as an overlay on every signal row, so a boundary can be read against
each physiological signal. Users switch between navigation and annotation with
one key. Selections can be a zoom-adaptive click, a dragged box, or a
right-click that selects the whole contiguous bout under the cursor; when a
drag reaches the viewport edge, the view pans automatically and newly
revealed signal is streamed in, so long selections keep boundary detail.
All selection forms feed one labeling step, keys `1`–`4` and `0` for clear.

Long signals are decimated on demand by
[Plotly Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]. The application uses that library only
to compute what to draw for a range, and owns how updates flow. The main
tradeoff is to keep interaction state in the interface and full-resolution
data on the server. Gestures, panning, and labeling run as JavaScript in the
interface. Navigation events are coalesced, so the server refreshes traces
after a gesture is released or pauses rather than on every frame, and
refreshes overtaken by newer navigation are discarded. Updates are applied as
trace patches rather than figure rebuilds. During a selection drag, auto-pan
instead refreshes newly revealed signal repeatedly through a dedicated
endpoint outside the Dash callback graph. Applying a label repaints the
overlay in the interface without a server round trip.

This interaction layer does not depend on the sleep domain. We extracted it
into a separate, runnable template [@timeseries_app_cookbook] that applies
the same navigation, selection, and labeling design to synthetic
multichannel data; adapting it to a new signal type centers on one
data-loading function and one label configuration.

## 2. Automatic scoring that keeps and learns from expert labels

The
application keeps the labels a user has supplied in a sparse layer, separate
from the displayed scores. When an automatic scoring run is confirmed, the
app snapshots this layer and places it over the backend's output, so labels
supplied to that run are kept as-is, whichever backend is used. The tradeoff
is provenance: scores already saved in an opened file also enter the layer
and are treated as user labels, even if they were earlier automatic results;
users clear intervals they want rescored.

The default backend is a rule-based algorithm that adapts to expert-labeled
examples, and needs no GPU or deep-learning runtime. Its rules identify
Wake from a normalized 1–7 Hz EEG spectral feature with minimum bout
lengths, and relabel candidate bouts as REM when NE falls below
recording-wide and within-bout percentile thresholds; without usable NE, it
identifies only Wake and NREM. Before each run, the algorithm treats the
user's Wake, NREM, and REM labels on the current recording, however few
(one is enough), as ground truth and automatically adjusts its five
parameters, searching a fixed set of candidate values, to best match those
examples; ties favor the
defaults, and the chosen values are displayed. MA labels are kept but not
used for fitting. We chose this recording-specific, inspectable adaptation
over persistent model training: it is fast, does not change the defaults for
the next recording, and matching the supplied examples is not a claim of
held-out accuracy. The externally developed sDREAMER model
[@chen2023sdreamer] can be selected instead, with optional
[PyTorch](https://pytorch.org) [@paszke2019pytorch] dependencies; it does not adapt, but its output keeps
user labels in the same way.

## 3. Checking behavior and completing a recording

For a selected interval of
up to five minutes, the application cuts the matching clip from the behavior
video with [FFmpeg](https://ffmpeg.org), validates the recording-to-video offset, and plays it in
the same window. One-step undo and filesystem-backed recovery protect
in-progress work; up to three isolated windows allow side-by-side
comparison, refusing the same file twice. Saving reports the first unscored
gap; a complete recording also exports bout, stage, and transition
statistics, including microarousals. Packaged Windows installations receive
compatible updates that preserve supported settings.

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

Manual timing tests on a 4.25-hour recording show the interaction design is
practical on long data. On a Windows laptop, optimizing the update pipeline
reduced the time from the end of a navigation gesture to the refreshed traces
from about 935 ms to about 300–370 ms; server work was 14–17 ms of that, and
the remainder is plot redrawing in the interface. On an Apple M4 laptop, refreshes after drag
panning took about 190–300 ms, and live auto-pan updates during a selection
about 260–310 ms. These are local manual measurements rather than a
controlled benchmark.
<!-- Source: ui_response_time_optimization_progress.txt (2026-05-23/25).
The measured pipeline matches the shipped code (the later Dash-store bypass
was reverted). Consider rerunning on the current release before submission. -->

# AI usage disclosure

From 2023 through 2025, the maintainer used the web versions of ChatGPT
(OpenAI) and Claude (Anthropic) for coding questions and suggestions while
developing the software; specific model versions were not recorded. Since
2026, AI coding agents working in the repository have assisted with code
generation, refactoring, tests, documentation, and release checks: Claude
Code (Claude Opus 4.5–5.5 and Fable 5) and OpenAI Codex (GPT-5 and GPT-6),
with sessions recorded in the repository's work log. Claude Code and Codex
also helped plan, draft, and review this manuscript. The authors made the
architectural and design decisions, reviewed and tested all AI-assisted code
before release, and reviewed, revised, and verified the manuscript against
the software and cited sources.
<!-- TODO(authors): confirm author review is complete before submission. -->

# Acknowledgments

This work was supported by the BRAIN Initiative of the US National Institutes
of Health (U19NS128613).

<!-- TODO: thank the lab PI, data contributors, and model contributors by
name. The funding sentence above is the PI-approved wording and should be
kept verbatim. -->

# References
