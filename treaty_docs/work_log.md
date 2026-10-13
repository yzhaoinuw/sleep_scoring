# Work Log

Prepend new session notes to this file. Record project decisions, reusable
evidence, and shared-state changes—not routine content already explained by the
diff. Each session ends with a `- Verification:` subsection. See
[`treaty_conventions.md`](treaty_conventions.md#work-log-discipline).

If today's date is already at the top, add a new `###` subsection beneath it.
The live log holds at most five unique dates; when a new date would exceed that
limit, move the oldest five-date chunk to
`work_log_archive/work_log_<earliest>_to_<latest>.md`.

Historical commands can contain machine-specific paths. When replaying them,
keep the `sleep_scoring` folder and `sleep_scoring_dash3.0` environment names
but adapt the user prefix and clone location. Default to the two newest dates;
search older entries by date anchor rather than reading every archive.

## 2026-10-13

### v0.17.5 partial-release candidate (Codex, GPT-6; effort/token budget not reported)

- Maintainer requested a partial release so the manuscript describes shipped
  behavior. Candidate includes the per-stage calibration tolerance from PR #17
  and annotation-save protection from local dev commit `a6932cc`.
- Reviewed PR #18: it copies only the publication title/abstract into
  `CITATION.cff`; all seven GitHub checks passed. Normal merge is blocked by
  the required-review policy, and the administrator override is awaiting the
  maintainer's explicit answer. The same reviewed metadata is included locally
  for candidate preparation, with abstract wording aligned to calibration.
- Aligned app/setup/CFF version 0.17.5 and release date 2026-10-13. Runtime
  changes are confined to compatible `app_src/` code; no frozen dependency,
  launcher, model, or packaging change. Supported bases are v0.17.1–v0.17.4.
- Save verification covers complete sibling staging and retained recovery
  exports on destination failure. Native shared-drive semantics still need
  maintainer testing; filesystem durability and the user's original truncation
  timing are not established by the failure-injection regressions.
- Rotated the prior five-date log chunk to its archive; existing local source
  checkout log edits and manuscript/latency artifacts remain preserved.
- Verification:
  - `Get-Date -Format yyyy-MM-dd`: 2026-10-13. PR #18 diff changes title and
    abstract only; all seven checks succeeded on `ccfda9d`.
  - Full lightweight candidate gate and frozen-install update validation will
    run on the committed candidate before any tag or release publication.
