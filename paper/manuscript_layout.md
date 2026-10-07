# JOSS manuscript layout and feature assessment

First-pass editorial plan, 2026-10-06, reviewed against shipped v0.17.4 and
`main` commit `4353b76`. Second-pass review, 2026-10-07: independent
re-assessment of the cookbook against source, followed by revision of this
plan (see [Second-pass changes](#second-pass-changes-2026-10-07)). This is a
planning document; `paper.md` was rewritten to it on 2026-10-07.

**Review attribution:** The visible blockquotes labeled **Codex feedback
(2026-10-07)** below are Codex's comments on Claude's second-pass revision.
They record agreement and proposed refinements separately from that revision.
Claude applied them on 2026-10-07 (see [Second-pass changes](#second-pass-changes-2026-10-07));
the blocks are kept as the review record.

## Recommended argument

Lead with a practical research problem: a scorer must inspect several kinds
of evidence, correct brief sleep-state boundaries, and retain those decisions
while reviewing a long recording. The application's contribution is the
integrated **inspect → select → label → predict → check → correct → export**
workflow at one-second resolution, with NE photometry treated as first-class
evidence and selection-linked behavior video.

The application-owned contributions fall into four groups. Each needs a
precise boundary against the dependency it builds on:

1. **Correction-oriented selection and labeling.** Whole-bout right-click
   selection, drag selection that continues across viewport edges while the
   newly revealed signal streams in, zoom-adaptive click selection, and
   one-key labels/clears, all converging on one selection model.
2. **An interaction/update pipeline that keeps that work fluid on long
   recordings.** Plotly Resampler supplies the decimation (what to draw for a
   given range). The app does *not* use the library's stock Dash update
   callback; it owns *when and how* updates flow: browser-side gesture
   handling and labeling, coalescing of navigation relayouts into a refresh
   after an idle interval or gesture release, dropping stale in-flight refreshes, applying patches by direct
   restyle, and a raw endpoint through which annotation auto-pan refreshes
   newly revealed signal repeatedly during the drag, outside the Dash
   callback graph. This is the honest answer to "isn't the big-data viewing
   just plotly-resampler?": that library makes long-signal display possible;
   this layer makes *annotating* it interactive. Describe it as design, and
   make no latency claim without measurements (see Research impact).
3. **Regenerated predictions preserve labels supplied to that run.** A
   sparse user-evidence layer is kept separate from displayed scores. When a
   run is confirmed, the app snapshots that layer and overlays it on the
   backend's output, so explicit labels present at confirmation are kept.
   This is not a guarantee for edits made while a run is in progress.
4. **An inspectable, per-recording adaptive scorer that uses NE.** The
   default backend is a small rule set (EEG low-band Wake rule plus low-NE
   REM rules) whose five controls are calibrated from a handful of the user's
   labels on the current recording. It runs without a GPU, Torch, or
   checkpoints. Using NE photometry as REM evidence ties the scorer to the
   motivating science; the biological rationale needs a primary citation
   (candidate: NE dynamics across sleep states from the BrainFlowZZZ
   group; authors to confirm the reference), not an assertion.

> **Codex feedback (2026-10-07) — contribution assessment:** I agree with
> promoting Recipes 7–8 to Core (design). My first pass underweighted the
> application-owned interaction/update pipeline. The code confirms that the
> app coordinates gestures, refresh timing, stale-update rejection, delivery,
> and live auto-pan around Plotly Resampler's patch computation. Keep this
> as a focused explanation of how the correction workflow works, with credit
> to the library for both decimation and patch computation. I also agree with
> foregrounding NE-informed scoring and requiring a primary biology citation;
> that citation supports the rationale, not the scorer's measured accuracy.

> **Codex feedback (2026-10-07) — refresh wording:** "One settled refresh per
> gesture" is too absolute. Ordinary navigation coalesces updates after an
> idle interval or gesture release; annotation auto-pan deliberately fetches
> and merges signals repeatedly while dragging. Suggested wording:
> "Navigation refreshes are coalesced, while annotation auto-pan refreshes
> newly revealed signals during the drag." This distinction also applies to
> Recipe 7's assessment and the Software design outline below.

> **Codex feedback (2026-10-07) — label-preservation wording:** Replace
> "Predictions that never overwrite human decisions" with "Regenerated
> predictions preserve labels supplied to that prediction run." The
> [prediction callbacks](../app_src/callbacks/prediction.py) snapshot the user
> layer at confirmation and overlay that snapshot after inference. This
> does not establish a blanket guarantee about edits made during an ongoing
> run. Retain the existing qualification about saved-score provenance.

Selection-linked video belongs in the workflow narrative as the ambiguity
check. Synchronized views, undo/recovery, native file access, side-by-side
windows, complete exports, and compatible updates support the argument by
reducing interruptions and protecting work. Give them space in proportion to
their practical value, not as separate innovations.

Credit Plotly Resampler once for on-demand decimation of long signals (and
its MinMaxLTTB downsampler), sDREAMER as an upstream model integration,
ffmpeg for clip extraction, and Dash/Plotly/pywebview for the framework.
Resampling algorithms and generic large-data plotting are not the
contribution. Defensible design contributions are not "first/only" claims;
claiming they are unprecedented or improve scientific accuracy requires
comparison and validation evidence. Keep opt-in usage tracking out of the
feature narrative.

Suggested title (either works; the second foregrounds correction):

> sleep_scoring: Interactive review and correction of rodent sleep annotations
> with synchronized electrophysiology, photometry, and behavior video

> sleep_scoring: Second-resolution correction of rodent sleep scores with
> norepinephrine photometry, behavior video, and adaptive prediction

Both drop the old title's emphasis on optional deep learning and leave room
for manual scoring and the default statistical backend.

> **Codex feedback (2026-10-07) — title preference:** I prefer the first title.
> It describes the complete review/correction workflow and accommodates
> recordings without NE. The second title places more weight on the NE and
> prediction paths than the manual workflow needs.

## Assessment of every cookbook recipe

Priority: **Core** gets substantive manuscript space; **Support** gets a brief
mention tied to the workflow; **Credit** identifies a dependency;
**Docs** stays in developer documentation; **Optional** is deferred.

| Recipe | Priority | Manuscript use and contribution boundary |
| --- | --- | --- |
| 1. Desktop shell | Support | Native file access and a familiar local desktop workflow; pywebview is an upstream dependency. |
| 2. Layout/component model | Docs | Ordinary UI plumbing; describe mode-appropriate controls only where useful. |
| 3. Server-side cache | Support | Explain recovery and local data briefly, not cache keys or serialization. |
| 4. File loading/dialogs/validation | Support | Open large local MAT recordings without browser upload; state the fixed input contract and need for adapters. |
| 5. Resampler figure | Support + Credit | Synchronized spectrogram/theta–delta, EEG, EMG, optional NE, and score overlays on one time axis are central context; credit Plotly Resampler for decimation. |
| 6. EventListener bridge | Docs | Implementation mechanism behind contribution 2; not named in the paper. |
| 7. Relayout coalescer | **Core (design)** | Part of contribution 2: navigation refreshes are coalesced instead of driving per-frame server work, and stale requests are dropped; annotation auto-pan deliberately refreshes repeatedly during a drag. App-owned; the stock resampler callback is not used. No latency claim without measurements. |
| 8. Patch/direct-restyle pipeline | **Core (design)** | Part of contribution 2: navigation and label edits patch traces instead of rebuilding the figure. Credit the resampler's patch computation; the delivery path is the app's. |
| 9. Keyboard panning | Support | A small part of efficient keyboard-led review. |
| 10. Custom pointer pan | Support | Navigation tailored to synchronized signals (x plus per-row y); omit low-level pointer/axis details. |
| 11. Mode switching | Support | One-key navigation/annotation switching keeps review and correction in the same workspace. |
| 12. Box/click/whole-bout selection | Core | Select a narrow interval or a complete scored/unscored bout without drawing every boundary by hand. Click width is 0.5% of the visible window, so it is not always exactly one epoch. |
| 13. Auto-pan selection/live refresh | Core | Extend a selection beyond the viewport without zooming away from boundary detail; revealed signal is fetched and merged during the drag. App-owned interaction work. |
| 14. Keypress annotation/overlays | Core | Immediate shared score display, one-second labels, manual MA, and clearing selected ranges, applied in the browser without a server round trip. Explain behavior, not trace patch syntax. |
| 15. Undo/crash recovery | Support | One-step undo and same-file, same-slot recovery protect decisions. Do not call it an unlimited undo stack or a backup system. |
| 16. Saving/export | Support | Partial MAT saves identify remaining gaps; complete saves offer bout, stage, and transition statistics, including MA. |
| 17. Selection-linked video | Core | Inspect behavior for the selected ambiguous interval without leaving the scoring workflow; offsets validated against video bounds. Credit ffmpeg and native playback. |
| 18. Performance instrumentation | **Support (evidence source)** | Not a feature, but the cheapest route to evidence for contribution 2: the existing browser/server profiler can produce a small, reproducible latency table. Existing instrumentation alone is not a benchmark. |
| 19. Multiple desktop instances | Support | Up to three isolated windows for comparing different recordings; the same MAT path is refused in a peer window. This is not simultaneous collaborative scoring. |
| 20. Opt-in aggregate reporting | Optional | Omit from the feature narrative. If used as impact evidence later, app-copy totals cannot establish distinct users/labs, accuracy, or time saved. |
| 21. Adaptive statistical calibration | Core | Explicit examples tune five controls for this recording by a small deterministic search with a default-distance tie-break; examples remain protected in predictions. Not persistent training or established accuracy improvement. |
| 22. Prediction backends/correction | Core | One correction workflow supports manual work, the default statistical scorer (NE-dependent REM), and optional upstream sDREAMER. Attribute the model correctly. |
| 23. Compatible startup updates | Support | One sentence on maintaining packaged installations and supported settings; updater mechanics stay in the cookbook. |

## Proposed JOSS structure

Use the current [JOSS paper guidance](https://joss.readthedocs.io/en/latest/paper.html)
and [review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html),
rechecked on 2026-10-07. The guidance gives a 750–1750-word range and
requires Summary, Statement of need, State of the field (including a
"build vs. contribute" justification), Software design, Research impact
statement, and AI usage disclosure, plus author affiliations, acknowledgment
of financial support, and references. Aim for roughly 1,450–1,650 words of
prose; the planning tables here belong in documentation, not the paper.

### Summary — about 150 words

Introduce rodent sleep annotation for a non-specialist. Describe the local
application, synchronized signal evidence, one-second manual correction,
optional prediction, behavior-video checks, and saved analysis outputs.
Mention EEG/EMG and explain optional photometry in plain language. Avoid a
package list and save the details of calibration for Software design.

Possible opening:

> sleep_scoring is a local desktop application for reviewing and correcting
> sleep-state annotations in rodent recordings. It brings brain electrical
> activity, muscle activity, an optional norepinephrine photometry signal,
> and selected behavior-video clips into a shared scoring workflow. Users
> can label one-second intervals manually or inspect automatic proposals,
> revise individual intervals or whole bouts, and export completed scores
> for subsequent analysis.

### Statement of need — about 170 words

Explain why short events and boundaries matter to the intended NE/sleep
research workflow (for example, microarousals and brief state transitions),
and why switching among signal, prediction, and video tools makes review
cumbersome. Identify experimenters doing rodent EEG/EMG scoring, especially
those with aligned NE photometry and video. Describe BrainFlowZZZ as the
motivating application, with comparable laboratories as the intended
audience; do not assert external adoption without evidence.

State the constraint explicitly: MAT recordings follow the documented field
contract; EEG and EMG share a sampling rate; optional photometry carries its
own rate. This is not yet a general-purpose EDF/acquisition-format importer.
One-second annotation resolution is a design requirement, not proof that
physiological boundaries or model estimates are accurate to one second.

### State of the field — about 200 words

Compare a small number of directly relevant tools using their primary papers
and current documentation (AccuSleep, Visbrain Sleep, somnotate, SPINDLE are
already in the bibliography; check whether any commercial package should be
named, and only with a citable source). Compare actual annotation
granularity, manual correction tools, photometry/video integration,
prediction/correction coupling, and installation/data contracts. Build the
source-backed comparison before claiming an unmet gap.

JOSS now requires an explicit **build vs. contribute** paragraph: why a new
application rather than extending an existing tool. The honest candidates
are the one-second correction workflow and NE-as-evidence requirement, and
the need for a responsive large-signal *annotation* layer, but each must be
checked against what the compared tools actually support.

The old draft's assertions that commercial tools are vendor-locked, most
tools assume 4–10-second epochs, alternatives rarely combine viewing and
scoring, and none accept NE are too broad to carry forward unchecked. Remove
uncited SleepEEGpy unless a relevant source and fair comparison are supplied.
Do not make a first/only claim.

### Software design — about 600 words

Organize this around three user tasks, with tradeoffs woven into each:

1. **Inspect and correct at the needed scale** (about 260 words). Shared time
   axes and score overlays let users read the same interval against multiple
   signals. Narrow, box, and whole-bout selections share one labeling step;
   edge auto-pan retains local detail during long selections. Then the key
   architectural tradeoff (contribution 2): interaction state lives in the
   browser and data work on the local server; navigation refreshes are
   coalesced, while auto-pan refreshes newly revealed signal during the
   drag; stale refreshes are discarded, and
   updates are applied as patches rather than figure rebuilds. Credit Plotly
   Resampler for decimation in one sentence, and state that the app replaces
   its stock update path. If a benchmark exists, cite one or two numbers here.
2. **Use predictions while retaining human decisions** (about 230 words).
   Separate displayed predictions from explicit user evidence. Describe the
   lightweight statistical backend (EEG low-band Wake rule; low-NE REM rules),
   its five configurable controls, recording-specific calibration by a small
   deterministic search, and protected manual overrides. State that without
   usable NE this backend does not identify REM. Note that existing MAT scores
   seed the evidence layer and their provenance is not distinguished. Credit
   optional sDREAMER separately. Avoid promising quality gains from sparse
   examples.
3. **Check evidence and complete a recording** (about 110 words). Show
   selection-linked video as an ambiguity check. Briefly mention one-step
   undo, recovery, isolated comparison windows, partial saves with gap
   feedback, and complete exports. Put distribution/settings preservation
   into one sentence if space permits.

Key tradeoffs: one-second labels versus continuous-time signal display;
browser-side interaction versus server-held full-resolution data;
local files/native dialogs versus a fixed MAT contract; sparse protected
evidence versus undifferentiated saved-score provenance; small deterministic
calibration versus persistent training; process-isolated windows versus
session-aware state and collaborative editing. These explain decisions
better than listing modules and libraries.

### Research impact statement — about 180 words

JOSS asks for evidence that is "compelling and specific, not aspirational":
realized impact (publications, external use, integrations) or credible
near-term significance. Candidates to collect from the authors:

- BrainFlowZZZ studies, preprints, or datasets whose scores were produced or
  corrected in the app, with approximate numbers of recordings/hours scored.
- Integration with the companion
  [preprocess_sleep_data](https://github.com/yzhaoinuw/preprocess_sleep_data)
  pipeline that produces the MAT contract.
- Any user outside the core group, even at pilot scale.

The repository has releases, documentation, tests, demo clips, and an
archived software version; these are readiness signals, not evidence of use.

Existing evidence: `ui_response_time_optimization_progress.txt` records manual
before/after-optimization timings (Windows i9 and Apple M4, one 4.25-hour
recording); the shipped navigation path matches the measured one. The draft
reports these as local manual measurements. Optional stronger version: rerun
on the current release with the existing profiler (Recipe 18), using a
well-defined gesture, repeated trials reported as distributions, and a
controlled direct-restyle on/off comparison (`ENABLE_DIRECT_PLOTLY_RESTYLE`).
Coalescing has no switch, so a with/without-coalescing comparison needs a
separate baseline harness. Latency supports the design claim only, not
adoption, scorer time savings, or accuracy. Record hardware, recording
length/rates, and settings. Stronger but costlier:
correction-task time and boundary error on reviewed intervals, and held-out
scoring agreement before/after calibration. Without such results, describe
implemented capabilities and concrete current use; omit quantified speed,
accuracy, and time-saving claims.

> **Codex feedback (2026-10-07) — benchmark scope:** Keep this optional for the
> next manuscript pass. A reproducible latency table would support the design
> and responsiveness claims; it would not demonstrate external adoption,
> scorer time savings, or scoring accuracy. Direct restyle has an existing
> configuration switch, but coalescing has no equivalent switch, so a
> with/without-coalescing comparison requires additional baseline or harness
> work. Start with a well-defined gesture, current-path measurements, or a
> controlled direct-restyle comparison. Report distributions over repeated
> trials, recording size/rates, hardware, and settings. Keep actual research
> use and integrations as separate evidence in the impact statement.

### AI usage disclosure — about 110 words

JOSS requires the tools and versions, where each was applied (code,
documentation, manuscript), the nature of assistance, and an affirmation that
humans reviewed, modified, and validated all AI output and made the primary
design decisions. Audit (2026-10-07, all work logs and git trailers):
maintainer used web ChatGPT and Claude 2023–2025 (versions unrecorded);
first agent co-authored commit 2026-01-29 (Claude Opus 4.5); Codex GPT-5
from 2026-04 and GPT-6 from 2026-09; Claude Opus 4.8, Fable 5, Opus 5, and
Opus 5.5 from 2026-06. No Grok. Work-log "ChatGPT" mentions in 2026-04 are an
unmerged experimental ChatGPT scoring backend, not development assistance.
Manuscript: Claude Code and Codex drafted and reviewed.

### Acknowledgments — about 60 words; References

Keep the existing approved funding sentence verbatim:

> This work was supported by the BRAIN Initiative of the US National Institutes
> of Health (U19NS128613).

Confirm authors, affiliations, ORCIDs, and contributor acknowledgments with
the maintainer. Cite directly relevant alternatives, Plotly Resampler,
sDREAMER, the NE/sleep biology reference, and the software archive. Recheck
any revised bibliographic claims.

## Recommended figure

One compact, annotated workflow figure is more useful than a feature gallery:
show synchronized signals and score overlays; mark a whole-bout or
cross-viewport selection; show its aligned video check; and indicate manual
labels retained after prediction. Use a redistributable recording or an
explicitly identified synthetic example. Capture a current application
session before finalizing the caption. A still image cannot demonstrate
latency or prove accuracy; link the existing demos as supplementary usage
material where appropriate.

## Release-to-cookbook audit

Scope: all release entries from v0.16.3–v0.17.4, plus the older interaction,
timing, and statistical-backend changes relevant to the existing draft.
Checked changelog claims against current source and targeted test coverage;
this is not a fresh release/package validation.

Evidence entry points: [interaction callbacks](../app_src/assets/clientsideCallbacks.js),
[secondary-click handling](../app_src/assets/graphContextMenu.js),
[figure construction](../app_src/make_figure.py),
[spectral timing](../app_src/get_fft_plots.py),
[prediction integration](../app_src/callbacks/prediction.py),
[statistical rules/calibration](../app_src/run_inference_stats_model.py),
[saving](../app_src/callbacks/saving.py),
[video](../app_src/callbacks/video.py), and
[startup/update boundaries](../run_desktop_app.py).
First-pass verification passed 116 Python tests covering helpers, spectral
timing, exports, score layers, metadata aliases, windows, and startup, plus
all 51 clientside JavaScript tests. These checks establish implemented
behavior within their test scope, not scientific accuracy or human task speed.

| Change | Cookbook action |
| --- | --- |
| v0.15.5 fractional-rate timing | Recipe 5 now explains anchored spectral columns and true EEG/EMG sample times; removed the stale ShortTimeFFT description. |
| v0.16.0 statistical scoring/styling | Added baseline backend behavior in Recipe 22; expanded Recipe 5 display customization. |
| v0.16.1 selection/auto-pan | Recipes 12–13 already cover the workflow; Recipe 12 now explains the later secondary-click race fix. |
| v0.16.2 signed video offsets/bounds | Recipe 17 now explains range validation and unavailable-video feedback. |
| v0.16.3 fp_frequency alias | Recipe 4 now covers the alias and saved-metadata preservation. |
| v0.16.4 incomplete-save feedback | Recipe 16 now covers the first gap and feedback even on dialog cancellation. |
| v0.16.5 multiple windows/optional Torch | Recipe 19 already covers isolation/peer exclusion; Recipe 22 covers optional model dependencies. |
| v0.16.6 path-safe recovery | Corrected Recipe 15's remaining basename-based adaptation guidance. |
| v0.16.7/v0.17.0 colors/config retention | Recipe 5 covers stage/legend colors; added Recipe 23 for updates and settings. |
| v0.16.8 MA export fix | Recipe 16 now explicitly covers MA statistics and regenerating existing workbooks. |
| v0.17.0–v0.17.1 updater behavior | Recipe 23 covers daily checks, current full base, graceful failures, and full-download guidance. |
| v0.17.2 tracking/calibration | Recipe 20 already covers opt-in tracking. Recipe 21 already covers calibration; Recipe 22 supplies the missing backend/correction context. |
| v0.17.3 tuned controls/feedback | Recipe 21 already documents five controls and separate 60-second feedback; retained that coverage. |
| v0.17.4 semantic overlay identity | Updated Recipes 5, 12, 14 and the gotcha catalog to use role/type lookup instead of last-three-trace assumptions. |
| v0.17.4 native video/nonfatal cleanup | Updated Recipe 17 to the shipped player and cleanup behavior, with remaining collision/freeze limits explicit. |

Second-pass spot checks (2026-10-07) confirmed against source: role-tagged
overlay lookup, native `html.Video` playback, `deque(maxlen=2)` paired
histories, auto-pan constants (`EDGE_PX`, `CLICK_PX`, `TRACE_REFRESH_MS`),
0.5% click width, 30% keyboard pan step, and that navigation calls only the
resampler's `construct_update_data_patch` (no stock update callback). Small
remaining gaps, none manuscript-blocking:

- The "Sampling Level" dropdown is referenced by Recipe 5 as covered in
  Recipe 8, but no recipe says it reloads the MAT and rebuilds the whole
  figure (`change_sampling_level` in `callbacks/loading.py`).
- The v0.11.0 selection EEG spectral-density plot is no longer in the app;
  its `update-fft-store` is unused, as is `backup-sleep-scores-store`. Do not
  claim the feature; the stores are cleanup candidates.

> **Codex feedback (2026-10-07) — confirmed gaps:** Both observations check out:
> `change_sampling_level` reloads the MAT and rebuilds the figure while copying
> current displayed scores, and the two named stores have no active consumers
> in `app_src`. The selection PSD plot should remain outside the manuscript.
> These findings do not require runtime changes to proceed with the rewrite.

## Corrections needed in the existing paper

- Replace SWS with NREM and document Wake/NREM/REM/MA accurately. Keyboard
  keys `1–4` map to stored codes `0–3`; key `0` clears selected labels.
- Add the default statistical scorer and calibration. Without usable NE,
  that backend identifies Wake/NREM and does not automatically detect REM.
  Do not imply all backends support every modality equally.
- Replace the claimed undo stack with one-step undo. Predictions and manual
  changes use the same display history. Qualify recovery by file path,
  window slot, and cache availability.
- Replace `dash-player` playback with native `html.Video`. The package is
  still listed in dependencies, but no longer implements clip playback.
- Replace the ShortTimeFFT implementation claim with the current anchored
  FFT-based spectrogram, or simply say spectral analysis without naming the
  former implementation.
- Replace "plotly-resampler keeps interaction responsive" with the split
  described in contribution 2: the library decimates; the app's update
  pipeline keeps navigation and annotation responsive.
- Remove the claim that the public repository ships representative MAT
  recordings: private/local test data and checkpoints are excluded. Add a
  public reproducible example before claiming one is supplied.
- Describe MAT saving as a user-selected path, which may overwrite the
  original; Excel export is offered separately only for complete scores.
- Replace blanket competitor and physiological-validity claims with
  attributed, verified statements. Rule-based cleanup is a heuristic.
- Do not include experimental Active/Quiet Wake, the removed selection PSD
  plot, or the pending full-path video-association/clip-identity fix as
  stable-release features. The original frozen-frame report is not
  established as resolved.
- Update the manuscript date when its revision is prepared; verify all
  author metadata. Preserve attribution and the approved funding text.

## Second-pass changes (2026-10-07)

- **Promoted** the relayout coalescer and patch/direct-restyle pipeline
  (Recipes 7–8) from Support to Core (design). The first pass correctly
  refused to sell plotly-resampler as ours but then under-credited the layer
  the app built in its place; source confirms the stock update path is not
  used.
- **Added** NE as first-class evidence in the scorer, not only a display
  channel and a limitation, with a required biology citation.
- **Added** the JOSS build-vs-contribute requirement to State of the field.
- **Reframed** Recipe 18 as the cheapest route to research-impact evidence
  and proposed a concrete latency table.
- **Expanded** Research impact with specific evidence to collect, and the AI
  disclosure to cover both agents used.
- **Recorded** cookbook gaps (sampling-level rebuild) and changelog drift
  (removed selection PSD plot, unused stores).
- **Applied Codex feedback:** qualified refresh wording so navigation is
  coalesced while auto-pan refreshes during the drag; restated label
  preservation as the confirmation snapshot; replaced the
  with/without-coalescing benchmark with the existing before/after log plus
  an optional direct-restyle comparison; title option 1 adopted. The
  contribution-assessment and confirmed-gaps blocks needed no change.

## Next revision pass

`paper.md` now follows this layout. Remaining: verify the related-tool
comparison from primary sources and confirm the build-vs-contribute
paragraph; collect concrete use/impact evidence and decide whether to rerun
the latency measurements;
supply the NE/sleep biology citation; provide a public example and current
workflow figure; complete author metadata and AI disclosure; then render and
review the JOSS PDF. The first pass should stand without usage-tracking
material or unmeasured performance claims.
