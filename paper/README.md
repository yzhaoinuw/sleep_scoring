# JOSS Paper Draft — Under Construction

**Status: active revision, not submission-ready.**

This directory holds an in-progress [JOSS](https://joss.theoj.org/) submission.
The first-pass release audit and feature/layout assessment are in
[`manuscript_layout.md`](manuscript_layout.md), reviewed against v0.17.4.
`paper.md` still contains outdated descriptions and awaits the corresponding
rewrite. Nothing here should be treated as a citable or finished artifact.

To cite the software today, use `CITATION.cff` at the repository root, or
GitHub's "Cite this repository" button. That file is current and validated;
this draft is not.

## Contents

- `paper.md` — the paper body (summary, statement of need, key features,
  implementation).
- `paper.bib` — the bibliography.
- `manuscript_layout.md` — feature priorities, contribution boundaries, release
  audit, current JOSS structure, and the next manuscript revision plan.

## What is done

- The draft body is written and scoped to the U19 BrainFlowZZZ program.
- sDREAMER is framed throughout as an externally developed upstream model the
  app integrates, not a contribution of this paper.
- Every bibliography entry carrying a DOI was checked against Crossref on
  2026-07-29 on these fields: title, venue, volume, pages, year, DOI, and
  author names. `paszke2019pytorch` has no registered DOI and is unverified.
- Five defects were found and corrected: the sDREAMER placeholder; the
  somnotate entry (wrong key, wrong year, missing metadata); the AccuSleep
  title, which did not match its DOI; and two wrong given names, SPINDLE's
  Benjamin Gallusser and Visbrain's Aymeric Guillot.
- Two given names intentionally differ from Crossref, which lowercases them.
  See the header comment in `paper.bib`.

## What is still open

- Author TODOs in `paper.md`: co-authors, affiliations, ORCIDs.
- Acknowledgments: PI, data/model contributors, funding and grant numbers.
- SleepEEGpy is named in the Statement of Need without a citation. Reviewers
  reliably catch uncited comparisons.
- `paszke2019pytorch` has no DOI. NeurIPS proceedings papers often lack one, so
  this may be acceptable as is, but it is the one entry Crossref cannot confirm.
- Every claim in the paper still needs checking against the shipped app.
- Apply the corrections and current required sections in `manuscript_layout.md`;
  prioritize user-facing review/correction features and attribute upstream work.

See the "Citation And Publication" section of
[`treaty_docs/next_steps.md`](../treaty_docs/next_steps.md) for the full checklist
and the software archive status.
