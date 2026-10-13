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
  `CITATION.cff`; all seven GitHub checks passed. Maintainer explicitly
  authorized the administrator override after the required-review policy
  blocked normal merging. PR merged as `715fabe` and its official history is
  included in this final candidate; abstract wording matches calibration.
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
  - Initial candidate `b554dd5` passed the full lightweight gate: 244 Python
    tests, 51 JavaScript tests, Black, compilation, source smoke, schema-2
    validation (20 payload files), and a fresh v0.17.1 frozen update/smoke
    preserving customized settings. Its ZIP SHA-256 was
    `204DA779C648E978552F16A3AF38C2FC6897F0210FDADFA4CBD02C126BE559C4`.
  - Final payload `9113f6532cf901beaa103dc61fef77634f1bfceb` passed the
    complete lightweight gate after the official PR merge: 244 Python tests,
    51 JavaScript tests, Black, compilation, source smoke, schema-2 validation
    (20 files), and a fresh frozen v0.17.1 update/smoke preserving settings.
    Final ZIP SHA-256:
    `31644A007CF638B113F0E4CE0A8751C0140E50FAA0612DD77EF95393662F73D1`.
  - GitHub CI run `38448938238` and CodeQL run `38448937619` both succeeded
    on that exact payload. `dev` is pushed at `9113f65`; documentation follows
    the tested payload. Tag v0.17.5 will point to `9113f65`, with `main`
    fast-forwarded to include the verification record before publication.
    GitHub asset and Zenodo DOI verification remain the publication steps.
