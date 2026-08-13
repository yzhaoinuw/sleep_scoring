# Work Log Archive

Older session notes rotated out of [`../work_log.md`](../work_log.md). Same structure as the live log (newest-first within each file).

## Rotation Policy

The live `work_log.md` holds at most the 5 most recent unique calendar dates. When a new date would push it past 5, move the oldest 5 dates as a chunk into a new file in this directory.

## File Naming

Each archive file is named for the date range it covers:

```
work_log_<earliest>_to_<latest>.md
```

For example: `work_log_2026-01-04_to_2026-02-12.md` for an archive whose oldest entry is dated 2026-01-04 and whose newest entry is dated 2026-02-12.

Each archive file holds exactly 5 unique calendar dates.

## Grepping Across All History

To search every entry (live + archived) at once:

```
rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md work_log_archive/
```
