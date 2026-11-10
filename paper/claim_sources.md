# Claim sources for `paper.md`

Maintainer and agent notes on where each manuscript claim comes from. These
were HTML comments inside `paper.md`; they moved here on 2026-11-10 because
the JOSS bot counts comment text toward the paper's word count. Each note is
listed under the section it annotated, with the sentence it followed.

## Statement of need

After: “…sleep architecture [@kjaerby2022norepinephrine] and may drive the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].”

Maintainer confirmed (2026-10-19) that these two citations frame the
program fairly. Neither study's scoring used this
application: Kjaerby et al. scored in Viewpoint's SleepScore (Methods), and
users report no publications from the app's scores yet (2026-10-15).

## State of the field

After: “…[Visbrain Sleep](https://github.com/EtienneCmb/visbrain) [@combrisson2019visbrain] is a general polysomnography viewer and hypnogram editor.”

Maintainer verified each characterization against the cited papers and
current documentation on 2026-10-19. Add commercial packages only with a
citable source. As of 2026-10-07 the AccuSleep,
AccuSleePy, and somnotate documentation describe EEG/EMG(/LFP) inputs and do
not mention photometry or video. Scoring methods checked 2026-10-14:
AccuSleep CNN on EEG/EMG (PLOS ONE 2019); SPINDLE CNN+HMM on 2 EEG + 1 EMG,
4 s epochs (PLoS Comput Biol 2019); somnotate LDA+HMM on EEG/LFP/EMG
spectrograms (repo README); Visbrain detects spindles/K-complexes/slow
waves/REMs but does not stage. Viewpoint SleepScore uses an undisclosed
"probability method" on EEG/EMG (product page); check its manual before
extending the NE claim to it. sDREAMER's NE-aware variant does use NE, so
keep the claim scoped to the tools listed here. SPINDLE's server moved
(checked 2026-10-19): sleeplearning.ethz.ch redirects to sleepyrat.ai,
whose FAQ says the scoring method is unchanged from SleepLearning and still
described by the 2019 paper. Rodent scoring is free up to a weekly quota,
with an optional paid annual lab subscription for unlimited use; the
operator is not named on the pages checked. Installation (READMEs, checked
2026-10-19): AccuSleep needs MATLAB 2017b+ and toolboxes; AccuSleePy is
`pip install accusleepy`; Somnotate is a conda environment plus
command-line pipeline scripts; Visbrain is `pip install visbrain`; none
ships a standalone executable. AccuSleep links a YouTube video
walkthrough, so do not claim demonstration videos as a distinction.

After: “…rammer experimenters, yet every open tool above requires either setup in a programming environment or uploading recordings to a web service.”

History from the Project 2 users (2026-10-14); they confirm the wording
in the full-draft review. SleepScore supports synchronized video, so the
stated gap is NE photometry integration only.

## 2. Visualizing and annotating at scale

After: “…roughly 65–75% from about 935 ms in an earlier version; on an Apple M4 laptop it was 64 and 181 ms. None of these is a controlled benchmark.”

Source: paper/latency_measurements.md (v0.17.4, 2026-10-08/09; same
recording at Sampling Level x1; first three updates per task excluded;
34–69 updates per task at 256- to 380-second views; 95th percentiles
Windows 371/239 ms, Mac 191/69 ms for keyboard/mouse). Windows and Mac runs
differ in machine and viewport; do not frame the difference as a speedup or
platform ranking. The 935 ms figure is from
ui_response_time_optimization_progress.txt (May 2026, Windows i9 laptop): an
earlier app version, not stock Plotly Resampler, under unmatched conditions;
keep it framed as a rough comparison, not a before/after benchmark. The
65-75% range is 1 - 329/935 and 1 - 212/935. The
one-to-two-second lags were seen by the maintainer while watching users
navigate over screen share in feedback meetings (no written record).

## 3. Automatic scoring that keeps and learns from expert labels

After: “…ng set. Agreement with the user's labels shows only that the fit succeeded, not how accurate the predictions are elsewhere in the recording.”

Rule origin: Project 2 researchers agreed the rules as a group. Kjaerby
et al. (Methods) define Wake/NREM/REM from EEG and EMG in 1 s epochs and MA
as wake bouts under 15 s, and note NE is lowest in REM; the app's numeric
thresholds are not in that paper, so keep the citation at the level of
stage criteria. Detail deliberately left to the README (2026-10-15): Wake
uses a normalized 1-7 Hz EEG feature; REM needs bouts of at least 30 s
whose within-bout low NE percentile reaches a recording-wide one (shape gating is
disabled in code), and each REM bout ends where NE recovers from its trough,
the rebound being relabeled Wake (maintainer, 2026-10-20: the defining
feature is a large valley that bounces back, not merely sustained low NE); five settings are
fitted (Wake threshold, minimum Wake and REM durations, two NE percentiles),
scored by average per-stage error plus a penalty for changing unlabeled
predictions; MA labels are kept but not fitted.

## Research impact statement

Removed 2026-11-10: "Results scored with the application have appeared in
conference posters." The track editor asked for citations, and no poster
titles or abstracts are on record, so the maintainer chose to drop the claim
rather than collect them from users. Re-add it only with citable details.

After: “…have together scored about 1,200 hours, a lower bound on total use. Results scored with the application have appeared in conference posters.”

Source: replies from five users, 2026-10-15/16: ~300 h, ~300 h, ~250 h,
~200 h, and 50-100 recordings at a typical ~3 h (150-300 h); about
1,200-1,350 h in total. No publications or datasets from these scores yet, and no known users
outside the program. As of 2026-10-18 there are no published studies,
preprints, or datasets to name; add them only if one appears before
submission.

## Acknowledgments

After: “…elley for supervising this work, Verena Untiet for leading the Copenhagen group, and Maiken Nedergaard for leading the BrainFlowZZZ program.”

The funding sentence above is the PI-approved wording and should be
kept verbatim. Names (2026-10-16) are from the user consensus so far; the
Copenhagen users are running a poll, so the user list may grow. Names are
alphabetical by surname within each group. The "especially thank" group
must meet both criteria: among the earliest users (a matter of record; some
have since left for other groups) AND extensive one-on-one feedback (only
Yue can confirm this). Don't add anyone on one alone (assumed surname:
Schiøler Nielsen); check spelling and diacritics with each person.

After: “…ue can confirm this). Don't add anyone on one alone (assumed surname: Schiøler Nielsen); check spelling and diacritics with each person. -->”

Recording contributors are credited through the "especially thank"
group (maintainer, 2026-10-20); the public example dataset's provider is among
them, so she is not named twice. The sDREAMER developers are cited, not
thanked separately. Acknowledgments final (maintainer, 2026-10-20).
