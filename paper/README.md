# JOSS Paper Draft — Under Construction

**Status: active revision, not submission-ready.**

This directory holds an in-progress [JOSS](https://joss.theoj.org/) submission
for Sleep Scoring App (repository and package name: `sleep_scoring`).
The feature/layout assessment is in
[`manuscript_layout.md`](manuscript_layout.md), reviewed against v0.17.4.
`paper.md` was rewritten to that layout on 2026-10-07; inline `TODO`
comments mark claims awaiting author confirmation. Nothing here should be
treated as a citable or finished artifact.

To cite the software today, use `CITATION.cff` at the repository root, or
GitHub's "Cite this repository" button. That file is current and validated;
this draft is not.

## Contents

- `paper.md` — the paper body (Summary, Statement of need, State of the field,
  Software design, Research impact statement, AI usage disclosure,
  Acknowledgments, References).
- `paper.bib` — the bibliography.
- `manuscript_layout.md` — feature priorities, contribution boundaries, release
  audit, the recommended figure, and the revision plan.

## For reviewers: JOSS requirements that shape this paper

JOSS papers are short and follow a fixed template, so several choices that may
look unusual are requirements, not style preferences. Sources: the JOSS
[paper format](https://joss.readthedocs.io/en/latest/paper.html) and
[submission](https://joss.readthedocs.io/en/latest/submitting.html) guides,
checked on 2026-10-07.

### Structure and length

- **Fixed top-level sections.** JOSS requires Summary, Statement of need,
  State of the field, Software design, Research impact statement, and AI usage
  disclosure, plus Acknowledgments (including funding) and References. There
  are no Introduction, Methods, Results, or Discussion sections. The section
  names in `paper.md` are the JOSS names.
- **750–1,750 words.** The body is currently about 1,650 words, excluding
  front matter and comments, so new material generally has to replace
  existing text. The figure caption will also count.
- **Summary is for non-specialists.** That is why it explains sleep scoring,
  EEG, and EMG from scratch.
- **No user manual or API documentation.** JOSS says that belongs in the
  software's own documentation. The paper therefore describes design decisions
  and tradeoffs rather than listing every feature or button.
- **Subsections are allowed** (JOSS supports several heading levels and
  recommends two or three). The three numbered subsections under Software
  design are our choice for readability, not a requirement.

### What each section has to argue

- **State of the field** must compare the software with existing tools *and*
  justify building a new tool instead of contributing to an existing one
  ("build vs. contribute"). The last paragraph of that section exists for
  this reason.
- **Software design** must explain tradeoffs and architecture, not only
  features. That is why it explains, for example, the local server inside a
  desktop window and how updates flow between the interface and the server.
- **Research impact statement** needs evidence of realized impact
  (publications, users, integrations) or credible near-term significance.
  This is currently the weakest section; see the open items below.
- **AI usage disclosure** is mandatory, even when no AI tools were used.

### Software scope: why sDREAMER is still mentioned

JOSS reviewers check that the paper's functional claims match the software.
sDREAMER is a backend users can select in the released app, so the paper
mentions it, but in one sentence that makes clear it is externally developed
and cited to its authors. The paper's own contribution is the default
adaptive, rule-based scorer and the label-preservation design. The claim that
user labels are kept "whichever backend is used" also depends on the app
having more than one backend.

### Naming

The paper calls the software **Sleep Scoring App** and links the
`sleep_scoring` repository at first mention. JOSS does not require the
software's name to match the repository name. We keep the name consistent
across the paper title, `CITATION.cff`, and the Zenodo archive, because JOSS
publishes the paper alongside an archived release of the software.

### Authorship

JOSS's policy: "Purely financial (such as being named on an award) and
organizational (such as general supervision of a research group)
contributions are not considered sufficient for co-authorship of JOSS
submissions, but active project direction and other forms of non-code
contributions are." All co-authors must consent to being listed and accept
accountability for the work. So supervision or funding alone does not qualify
someone, but setting the software's scientific direction or requirements
does. Contributors who do not meet this bar, including people who only
supplied recordings or test data, go in Acknowledgments.

JOSS also does not require citing every dependency. The paper cites the
libraries it discusses as design choices (Plotly, Plotly Resampler, PyTorch)
and leaves out general-purpose ones such as NumPy and pandas.

### References and links

- **The References heading is intentionally empty in `paper.md`.** JOSS
  builds the reference list automatically from `paper.bib`, using the
  `[@key]` citations in the text. Only cited entries appear. To see the
  formatted references, render the PDF (see the next section).
- **Hyperlinks are encouraged** by JOSS for websites and external resources.
  Software tools are linked at first mention. A tool can have both a link (to
  the software) and a citation (to its paper).
- **Full venue names** are required in references (no journal
  abbreviations).
- **Title in plain text**, with no code formatting, which is one reason the
  title uses "Sleep Scoring App" rather than `` `sleep_scoring` ``.

### Software eligibility (not about the text, but checked at review)

- An OSI-approved open-source license (MIT here), and a public repository
  where anyone can browse code, open issues, and propose changes.
- Public for more than six months with active development (since 2023 here).
- An obvious research application, good documentation, and tests. For a
  single-maintainer project, JOSS looks for tagged releases or a changelog,
  tests and CI, and clear documentation.
- After acceptance: a tagged release archived with a DOI (Zenodo). All paper
  authors must appear in the archive's author list, and any differences
  between the two author lists must be explained.

### Rendering the PDF

JOSS renders `paper.md` with its own toolchain, so a plain Markdown preview
does not show the final layout or references. Preview it with the JOSS
draft-PDF GitHub Action or the `openjournals/inara` Docker image.

## What is done

- The draft body is written to the JOSS structure and scoped to Project 2 of
  the U19 BrainFlowZZZ program.
- sDREAMER is framed throughout as an externally developed model the app
  integrates, not a contribution of this paper.
- The software name is set to Sleep Scoring App in the paper title and in
  `CITATION.cff`.
- Software tools are hyperlinked at first mention: AccuSleep, AccuSleePy,
  SPINDLE, Somnotate, Visbrain Sleep, Dash, Plotly, pywebview, Plotly
  Resampler, PyTorch, and FFmpeg.
- Every bibliography entry carrying a DOI was checked against Crossref on
  2026-07-29 (title, venue, volume, pages, year, DOI, author names); entries
  added on 2026-10-07 were checked on that date. Two given names intentionally
  differ from Crossref, which lowercases them; see the header comment in
  `paper.bib`.

## What is still open

### Figure

The maintainer will choose the recording and capture the figure; the current
tooling pass is limited to latency logging.

- [ ] **Capture and add the workflow figure** (maintainer). One annotated
  screenshot at a zoom of roughly 2–5 minutes showing: all four signal rows
  (spectrogram with theta/delta, EEG, EMG, NE) with the score overlay and
  legend; a mix of predicted and manually corrected bouts, ideally an MA and
  a REM bout with a visible NE dip; an active (preferably right-click
  whole-bout) selection; and the video clip window for that selection.
  Optionally, the tuned-parameter message after an adaptive run. Use a
  recording the lab is willing to publish. Full spec:
  [`manuscript_layout.md`, "Recommended figure"](manuscript_layout.md#recommended-figure).
- [ ] Add the figure to `paper.md` with a caption and label, and recheck the
  word count afterward.
- [ ] Publish a public example recording reviewers can open in the app.

### Claims the authors said they would verify

- [ ] **NE biology citations.** Confirm with the Kjaerby/Hauglund co-authors
  that the two citations frame the program fairly, and whether either study's
  scoring used this application.
- [ ] **Related tools.** Verify each characterization of AccuSleep/AccuSleePy,
  SPINDLE, Somnotate, and Visbrain against the cited papers and current
  documentation, including epoch-length options. Add commercial packages only
  with a citable source.
- [ ] **Build vs. contribute.** The rationale in State of the field is
  inferred from the design, not from project history. Confirm it, or replace
  it with the actual reasons a new application was built in 2023.
- [ ] **Research impact evidence.** Add the users' rough estimate of total
  recording hours scored, and name studies, preprints, or datasets scored
  with the app once confirmed.
- [ ] **Latency measurements.** Two current-release Windows sessions are
  reviewed in `latency_measurements.md`. Complete machine/display context
  and integrate the selected results into the manuscript. Optional fresh
  macOS measurements can use the launcher and procedure below.
- [ ] **AI usage disclosure.** Confirm that author review of the manuscript
  is complete.

### Run the latency measurements

The first two maintainer sessions have been reviewed; see
[`latency_measurements.md`](latency_measurements.md) for the selected results,
metric boundaries and proposed manuscript wording.

From the repository folder, activate the existing environment and launch:

```powershell
conda activate sleep_scoring_dash3.0
python paper/measure_latency.py record
```

The same commands can be used in a macOS terminal with the app's working
source-run environment. The capture script uses portable Python APIs; the
desktop launcher selects the native renderer outside Windows. This new
capture launcher has been exercised on Windows, not yet on macOS. Run
`python paper/measure_latency.py record --check` on the Mac first, then
launch `record` and check that the resulting summary contains browser samples.
Record the Mac model/chip, RAM, macOS version and native browser/runtime
context in the notes; the automatic CPU description may only identify the
architecture. If optional `psutil` is absent, fill in RAM manually.

This opens the normal desktop app. **Background server and browser logging
are ON for this run**, using the existing profiling environment overrides.
Choose your MAT file normally. Close other app windows before launching:
only slot 0 profiles, and the launcher checks that ports 8050–8052 are free.
Automatic updates are skipped for this measurement run. App source/config
and recording files are not changed by the tooling; normal app saves remain
under your control.

To verify logging without opening a window:

```powershell
python paper/measure_latency.py record --check
```

The three profiling flags should all print `true`, and `INSTANCE_SLOT` should
be `0`. Closing the app ends the run and automatically writes a summary in
`paper/latency_runs/<timestamp>/`. No copy/paste of terminal logs is needed.
Logging also remains visible in the launching terminal.

Use a separate run for each recording and Sampling Level. For an initial
repeatable task:

1. Use Sampling Level **x1** and a fixed **300-second viewport** away from the
   recording edges. Record the actual width in the notes; arrow panning
   preserves it.
2. Press the right arrow once, wait for the trace refresh to finish, then
   press the left arrow once. Continue for at least **30 individual presses**,
   with about two seconds between presses. Avoid held keys; rapid repeated
   inputs can be coalesced into fewer refresh observations.
3. If measuring pointer pan/zoom too, make each gesture separately, pause
   until the refresh completes, and repeat at least 30 times per task.
   Different viewport widths and event sources produce separate groups.
4. For auto-pan, switch to annotation mode and drag a selection beyond an
   edge for a few seconds. Repeat several separate drags. This measures
   live-refresh operations within drags; those observations are correlated
   and are not equivalent to 30 independent user trials. No labeling or
   saving is required to measure navigation/selection.
5. Close the app normally. Fill in `notes.md` with recording duration and
   rates/sample counts, Sampling Level, viewport/task, window/display scale,
   CPU/GPU model and embedded browser/runtime version, power mode, and other
   active apps (Edge/WebView2 on Windows; the native renderer on macOS).

Each run contains:

- `app.log`: complete stdout/stderr, flushed continuously.
- `profiling-check.json`: verified effective flags before launch.
- `metadata.json`: commit, OS, CPU description/count, RAM, Python and package
  versions, and the direct-restyle setting.
- `events.jsonl` and `samples.csv`: parsed profiler events and valid browser
  samples, with ordinary navigation/server events joined by their profile ID.
- `summary.md`: counts, medians and p95 values grouped by event, source, mode,
  and viewport width. The first three samples in each group are excluded as
  warm-up; raw samples are retained.
- `notes.md`: the recording/task/display details to complete yourself.

To rebuild the report or change the warm-up count:

```powershell
python paper/measure_latency.py summarize paper/latency_runs/<timestamp> --warmup 3
```

Use at least 20 retained observations per ordinary-navigation group before
interpreting its distribution. If a run has no browser samples, the report
says so rather than producing latency numbers. Do not mix Sampling Levels or
recordings within a run, or call the current-path measurements a comparison
with an older baseline. The tool records the current configuration; it does
not disable coalescing or change direct restyle for a comparative experiment.

**Metric boundaries:** ordinary `browser_total` includes coalescing and the
trace-update completion path; server callback time is already inside that
total. The existing auto-pan `browser_total` stops after issuing the
`Plotly.restyle` call and does not await completed rendering. Report those
two event types separately. Neither measures human task time, scoring
accuracy, or exact screen presentation latency. These are instrumented runs,
with logging enabled, rather than an assertion about uninstrumented speed.

The output folder is ignored by Git. Raw app logs may include recording
paths; inspect/redact local logs and notes before sharing them. Keep the
existing manuscript timing claims qualified until the new runs are reviewed.

### Authors and metadata

- [ ] **Credit for the scoring rules (resolve before submission).** The
  default scorer's rules (Wake from 1–7 Hz EEG power, REM from an NE dip,
  minimum bout lengths) were agreed by Project 2 researchers as a group and
  passed on through a few people. The maintainer designed and built the
  adaptive parameter fitting and the rest of the software. Decide with the PI
  and those researchers whether anyone is a co-author. Either way, the paper
  should say where the rules came from, not only in Acknowledgments.
- [ ] Co-authors: the maintainer developed the software largely
  independently, so the paper is currently single-author. Add someone only if
  they meet the JOSS bar above (code, or active project direction), with
  their affiliation and ORCID. Confirm or expand the University of Rochester
  affiliation.
- [ ] Acknowledgments: thank the PI, the users who supplied recordings and
  feedback, and the sDREAMER developers by name. Keep the approved funding
  sentence verbatim.
- [ ] Update the manuscript date when the revision is finalized.
- [ ] Rename the Zenodo archive title to match the paper and `CITATION.cff`.

### Bibliography and final checks

- [ ] `paszke2019pytorch` has no DOI. NeurIPS proceedings papers often lack
  one, so this may be acceptable, but Crossref cannot confirm it.
- [ ] Render the JOSS PDF and review it, including the generated references.

See the "Citation And Publication" section of
[`treaty_docs/next_steps.md`](../treaty_docs/next_steps.md) for the full checklist
and the software archive status.
