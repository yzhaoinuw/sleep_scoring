# JOSS Paper Draft — Under Construction

**Status: active revision, not submission-ready** (updated 2026-10-19).

**Review PDF:** [Read the current manuscript](Sleep_Scoring_App_JOSS_Manuscript_2026-10-19.pdf).
The JOSS manuscript PDF is already included in `paper/` for collaborators to
read and comment on. Open the link above to review it. Yue maintains this copy;
the download and refresh steps below are for the maintainer only.

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

**Collaborators reviewing the manuscript:** read the included PDF linked above,
then start with
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

- The dated review PDF linked above — the JOSS manuscript already included in
  this directory for collaborators.
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
- **750–1,750 words.** As of 2026-10-19 the body (captions included,
  front matter, comments, and Acknowledgments excluded) is about 1,740 words
  of running text, and more with headings and rendered citations;
  Acknowledgments add about 150 if counted. Treat
  the stricter count as binding: the draft is at the limit, so new material
  has to replace existing text.
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
  formatted references, open the included review PDF linked above.
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

The PDF already included in `paper/` is rendered with the JOSS toolchain and
shows the manuscript's layout and formatted references. Collaborators can
open it directly using the link at the top of this README.

**Maintainer only (Yue):** after editing the manuscript text (`paper.md`),
bibliography (`paper.bib`), or figures, refresh the included PDF. The
`JOSS draft PDF` workflow (`.github/workflows/draft-pdf.yml`) runs on every
push that touches `paper/` and can also be started by hand. Download and
unpack its `Sleep_Scoring_App_JOSS_Manuscript_<date>` artifact, then place the
PDF in `paper/`, replacing the previous review copy. The date is that of the
commit used for the build. Update the review link at the top of this README
and include the refreshed PDF when sharing the manuscript changes for review.
Keep only the current review PDF in this directory; Git history retains
earlier versions. The workflow does not automatically update the included
PDF. The Markdown, bibliography, and figure sources remain authoritative.

## What is done

- **Public example data (published 2026-10-20).**
  [Example EEG/EMG/NE recording and behavior video](https://doi.org/10.5281/zenodo.23489642),
  dataset v1.0.0, is openly available under CC BY 4.0. It includes the
  unscored MAT, an optional reference-scored copy, matching MP4, and usage
  instructions. Klaudia Anna Tokarska provided the recording/video; the
  reference labels illustrate the manuscript's rules and may differ from
  another expert's interpretation. The record, DOI, and all six published
  file checksums were verified against the prepared copies.
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
  summarized in Software design, section 2.
- **Research impact evidence.** Five users reported their use on
  2026-10-15/16: about 1,200 hours of recordings in total, from a channel of
  about 30 members, with results presented in conference posters. The
  per-user breakdown is in a comment in `paper.md`.
- **Zenodo title.** `sleepscoring_zenodo` in `paper.bib` matches the title
  and year archived in Zenodo v0.17.6 (verified 2026-10-15,
  [version DOI](https://doi.org/10.5281/zenodo.23385811)). The bibliography
  keeps the concept DOI so it resolves to the latest archived release.
- **Paper title.** Changed on 2026-10-18 to "Visualizing and annotating
  rodent sleep recordings with synchronized electrophysiology, photometry, and
  behavior video", matching the Software design wording, in `paper.md` and
  `CITATION.cff`. The bib entry keeps the archived (old) title until the next
  release.
- **Calibrator release.** The change-penalized calibrator is released in
  v0.17.6 and archived by Zenodo (DOI `10.5281/zenodo.23385811`, verified
  2026-10-15).
- **Research impact publications.** Checked 2026-10-18: no studies,
  preprints, or datasets scored with the app are published yet, so the
  section cites conference posters only. Add any that appear before
  submission.
- **PyTorch reference.** `paszke2019pytorch` has no DOI (Crossref has none,
  as usual for NeurIPS proceedings). On 2026-10-18 it was checked against the
  [NeurIPS proceedings page](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)
  instead, and that URL was added to the entry.
- **Acknowledgments (updated 2026-10-19).** Names the seven researchers
  who helped define the sleep-stage rules; especially thanks four of the
  earliest users, who gave extensive one-on-one feedback; lists the other
  users at Copenhagen and Rochester alphabetically; and thanks the
  supervisor, the Copenhagen group leader, and the program PI. The funding
  sentence is unchanged.
- **Authorship.** After reviewing the JOSS criteria, Yue Zhao remains the
  sole person qualified as an author.

- **Figure print legibility.** Closed by the maintainer on 2026-10-19; the
  application's tick labels still need zoom, as noted above.
- **NE biology citations.** The maintainer confirmed on 2026-10-19 that the
  Kjaerby and Hauglund citations frame the program fairly.
- **Related tools.** The maintainer verified the characterizations of
  AccuSleep/AccuSleePy, SPINDLE, Somnotate, and Visbrain against the cited
  papers and current documentation on 2026-10-19. SPINDLE's server now
  redirects to sleepyrat.ai, and State of the field notes that every listed
  tool except SPINDLE needs a MATLAB or Python environment.
- **Credit for the scoring rules.** Confirmed complete on 2026-10-19:
  Acknowledgments names the seven researchers who helped define the rules,
  and Software design, section 3 cites Kjaerby et al. 2022 for the stage
  criteria only.
- **Affiliation.** The University of Rochester affiliation is confirmed.

## What is still open

### Claims the authors said they would verify

- [ ] **AI usage disclosure.** Confirm that author review of the manuscript
  is complete.

### Authors and metadata

- [ ] Acknowledgments: add names from the Copenhagen users' poll, check
  each name's spelling and diacritics (including the assumed surname
  Schiøler Nielsen), and decide whether to thank anyone who supplied
  recordings or the sDREAMER developers (already cited) by name. Keep the
  approved funding sentence verbatim.
- [ ] Update the manuscript date when the revision is finalized (set to
  14 October 2026 during the text review).

### Release and final checks

- [ ] After the next release, confirm Zenodo shows the new title, then
  update the `sleepscoring_zenodo` title in `paper.bib` to match.
- [ ] Review the final rendered PDF in full, including the generated
  references.

See the "Citation And Publication" section of
[`treaty_docs/next_steps.md`](../treaty_docs/next_steps.md) for the full checklist
and the software archive status.
