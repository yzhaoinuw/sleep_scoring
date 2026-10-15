# JOSS Paper Draft — Under Construction

**Status: active revision, not submission-ready** (updated 2026-10-15).

This directory holds an in-progress [JOSS](https://joss.theoj.org/) submission
for Sleep Scoring App (repository and package name: `sleep_scoring`).
Inline `TODO` comments in `paper.md` mark claims awaiting author
confirmation. Nothing here should be treated as a citable or finished
artifact.

To cite the software today, use `CITATION.cff` at the repository root, or
GitHub's "Cite this repository" button. That file is current and validated;
this draft is not.

The calibration wording in `paper.md` describes the change-penalized
calibrator released in v0.17.6 and archived by Zenodo
([version DOI](https://doi.org/10.5281/zenodo.23385811), verified 2026-10-15).
The software citation now resolves to a release containing this behavior.

**Collaborators reviewing the manuscript:** start with
[What is still open](#what-is-still-open), especially
[Claims the authors said they would verify](#claims-the-authors-said-they-would-verify)
and [Authors and metadata](#authors-and-metadata). Those items need your
knowledge, not the maintainer's. The
[JOSS requirements](#for-reviewers-joss-requirements-that-shape-this-paper)
section explains choices that may look unusual but are required by JOSS.

## Overview

- [Contents](#contents): the files in this directory.
- [For reviewers: JOSS requirements that shape this paper](#for-reviewers-joss-requirements-that-shape-this-paper)
  - [Structure and length](#structure-and-length)
  - [What each section has to argue](#what-each-section-has-to-argue)
  - [Software scope: why sDREAMER is still mentioned](#software-scope-why-sdreamer-is-still-mentioned)
  - [Naming](#naming)
  - [Authorship](#authorship)
  - [References and links](#references-and-links)
  - [Software eligibility](#software-eligibility-not-about-the-text-but-checked-at-review)
  - [Rendering the PDF](#rendering-the-pdf)
- [What is done](#what-is-done)
- [What is still open](#what-is-still-open)
  - [Figures](#figures)
  - [Claims the authors said they would verify](#claims-the-authors-said-they-would-verify)
  - [Authors and metadata](#authors-and-metadata)
  - [Release and final checks](#release-and-final-checks)

## Contents

- `paper.md` — the paper body (Summary, Statement of need, State of the field,
  Software design, Research impact statement, AI usage disclosure,
  Acknowledgments, References).
- `paper.bib` — the bibliography.
- `figures/` — five original application snapshots, their panel mapping
  (`figures/README.md`), `build_figures.py`, and the two composed figures.
- `latency_measurements.md`, `measure_latency.py`, `latency_runs/` — the
  navigation timing measurements quoted in Software design, the script that
  summarizes them, and the raw runs.
- `manuscript_layout.md` — the 2026-10-07 planning document (reviewed against
  v0.17.4): feature priorities, contribution boundaries, release audit, and
  the two-figure plan. `paper.md` has since been revised beyond it.

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
- **750–1,750 words.** As of 2026-10-15 the body (captions included,
  front matter and comments excluded) is about 1,780 words of running text,
  or about 1,830 counting headings and more with rendered citations. Treat
  the stricter count as binding: the draft is over, so new material has to
  replace existing text.
- **Summary is for non-specialists.** That is why it explains sleep scoring,
  EEG, and EMG from scratch.
- **No user manual or API documentation.** JOSS says that belongs in the
  software's own documentation. The paper therefore describes design decisions
  and tradeoffs rather than listing every feature or button.
- **Subsections are allowed** (JOSS supports several heading levels and
  recommends two or three). The five numbered subsections under Software
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
both scoring models keep user labels as-is also depends on the app having
more than one backend.

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
draft-PDF GitHub Action or the `openjournals/inara` Docker image. The
repository's `JOSS draft PDF` workflow (`.github/workflows/draft-pdf.yml`)
runs on every push that touches `paper/` and can also be started by hand;
download the `Sleep_Scoring_App_JOSS_Manuscript_<date>` artifact from the run
(the date is that of the commit it was built from). The PDF is not committed.

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
- **Figures.** Five workflow snapshots of the same recording interval are in
  [`figures/`](figures/README.md), replaced on 2026-10-13 with the current
  calibration and a 96-second video selection. `figures/build_figures.py`
  composes them into the two agreed figures, expert-guided adaptation (three
  panels) and selection-linked video (two panels), which are in `paper.md`
  with captions, labels, and in-text references.
- **Rendered figures checked.** The 2026-10-14 PDF
  ([run 38509732148](https://github.com/yzhaoinuw/sleep_scoring/actions/runs/38509732148))
  is seven pages with both figures at full text width; panel letters,
  titles, and callouts read at print size, while the application's own tick
  labels are about 3–4 pt and need zoom. On 2026-10-15 the figure
  cross-references, which a text edit had broken, were repaired and confirmed
  in the PDF from
  [run 38686198524](https://github.com/yzhaoinuw/sleep_scoring/actions/runs/38686198524).
- **Build vs. contribute.** The inferred rationale was replaced with the
  users' account (2026-10-14): Project 2 previously used Viewpoint's
  SleepScore, which did not integrate fiber-photometry (NE) signals well.
  SleepScore is linked to its product page, not cited. Users confirm the
  wording in the full-draft review.
- **Latency measurements.** Two Windows sessions and one macOS (Apple M4)
  session on v0.17.4 are reviewed in `latency_measurements.md` and
  summarized in Software design, section 1.
- **Research impact evidence.** Four users reported their use on
  2026-10-15: about 1,000 hours of recordings in total, from a channel of
  about 30 members, with results presented in conference posters. The
  per-user breakdown is in a comment in `paper.md`.
- **Zenodo title.** Zenodo v0.17.6 uses the paper's title, and
  `sleepscoring_zenodo` in `paper.bib` matches the archived title and year
  (verified 2026-10-15,
  [version DOI](https://doi.org/10.5281/zenodo.23385811)). The bibliography
  keeps the concept DOI so it resolves to the latest archived release.

## What is still open

### Figures

- [ ] Decide whether print legibility of the in-app text matters enough to
  retake the snapshots with a smaller window (larger relative text).
- [ ] Publish a public example recording reviewers can open in the app.

### Claims the authors said they would verify

- [ ] **NE biology citations.** Confirm with the Kjaerby/Hauglund co-authors
  that the two citations frame the program fairly. (Neither study's scoring
  used this application; see the comment in `paper.md`.)
- [ ] **Related tools.** Verify each characterization of AccuSleep/AccuSleePy,
  SPINDLE, Somnotate, and Visbrain against the cited papers and current
  documentation, including epoch-length options. Add commercial packages only
  with a citable source.
- [ ] **Research impact publications.** Name studies, preprints, or datasets
  scored with the app once published (none yet; results so far are
  conference posters, without titles on hand).
- [ ] **AI usage disclosure.** Confirm that author review of the manuscript
  is complete.

### Authors and metadata

- [ ] **Credit for the scoring rules (resolve before submission).** The
  default scorer's rules (Wake from 1–7 Hz EEG power, REM from an NE dip,
  minimum bout lengths) were agreed by Project 2 researchers as a group and
  passed on through a few people. The maintainer designed and built the
  adaptive parameter fitting and the rest of the software. Decide with the PI
  and those researchers whether anyone is a co-author; users have suggested
  names. The paper now says where the rules came from: Software design,
  section 2 says they encode the stage criteria used in Project 2, citing
  Kjaerby et al. 2022, which a user suggested. That paper defines the stages
  but not the app's numeric thresholds, so the citation stays at that level.
- [ ] Co-authors: the maintainer developed the software largely
  independently, so the paper is currently single-author. Add someone only if
  they meet the JOSS bar above (code, or active project direction), with
  their affiliation and ORCID. Confirm or expand the University of Rochester
  affiliation.
- [ ] Acknowledgments: thank the PI, the users who supplied recordings and
  feedback, and the sDREAMER developers by name. Keep the approved funding
  sentence verbatim.
- [ ] Update the manuscript date when the revision is finalized (set to
  14 October 2026 during the text review).

### Release and final checks

- [x] Release the change-penalized calibrator and confirm Zenodo archived it:
  v0.17.6, DOI `10.5281/zenodo.23385811`, verified 2026-10-15.
- [ ] Bring the body back within 1,750 words under the stricter count.
- [ ] `paszke2019pytorch` has no DOI. NeurIPS proceedings papers often lack
  one, so this may be acceptable, but Crossref cannot confirm it.
- [ ] Review the final rendered PDF in full, including the generated
  references.

See the "Citation And Publication" section of
[`treaty_docs/next_steps.md`](../treaty_docs/next_steps.md) for the full checklist
and the software archive status.
