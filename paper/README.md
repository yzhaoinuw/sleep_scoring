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
- [ ] **Latency measurements.** Decide whether to rerun the manual timing
  tests on the current release before submission.
- [ ] **AI usage disclosure.** Confirm that author review of the manuscript
  is complete.

### Authors and metadata

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
