"""Verify profiler parsing, correlated metrics, and per-task warm-up exclusion."""

import json
import pytest

from paper import measure_latency
from paper.measure_latency import (
    group_samples,
    parse_events,
    percentile,
    profiling_environment,
    samples_from_events,
    summarize,
)


def test_logging_overrides_disabled_environment(monkeypatch):
    monkeypatch.setenv("SLEEP_SCORING_PROFILE_RESAMPLER", "0")
    monkeypatch.setenv("SLEEP_SCORING_BROWSER_NAV_PERF_LOG", "0")
    monkeypatch.setenv("SLEEP_SCORING_INSTANCE_SLOT", "2")
    env = profiling_environment()
    assert env["SLEEP_SCORING_PROFILE_RESAMPLER"] == "1"
    assert env["SLEEP_SCORING_BROWSER_NAV_PERF_LOG"] == "1"
    assert env["SLEEP_SCORING_INSTANCE_SLOT"] == "0"


def test_navigation_joins_server_by_id_without_adding_times():
    events = parse_events(
        [
            "[resampler] id=8, browser_profile_id=3, total=14.2 ms, payload=95.0 KB, "
            "apply_path=direct-restyle, xaxis4.range=[600, 900]",
            "[browser-nav] profile_id=3, mode=final, source=keyboard, coalesce=120.1 ms, "
            "dash_apply=203.0 ms, browser_total=323.1 ms, frame_gap=n/a, x_width=300.0 s",
        ]
    )
    samples = samples_from_events(events)
    assert len(samples) == 1
    assert samples[0]["browser_total_ms"] == 323.1
    assert samples[0]["server_total_ms"] == 14.2
    assert samples[0]["payload_kb"] == 95.0
    assert samples[0]["view_width_s"] == 300


@pytest.mark.parametrize("applied, stale", [("true", "false"), ("True", "False")])
def test_autopan_excludes_stale_updates_and_never_joins_unrelated_server(applied, stale):
    events = parse_events(
        [
            "[resampler] browser_profile_id=4, total=90 ms",
            "[browser-autopan] id=4, mode=merge, browser_total=240 ms, visible_width=300 s, "
            f"payload=100 KB, applied={applied}",
            f"[browser-autopan] id=5, mode=merge, browser_total=10 ms, applied={stale}",
            "[browser-autopan] id=6, mode=merge, browser_total=n/a, applied=true",
        ]
    )
    samples = samples_from_events(events)
    assert len(samples) == 1
    assert samples[0]["server_total_ms"] is None
    assert samples[0]["view_width_s"] == 300


def test_warmup_is_independent_by_source_and_view_width():
    samples = [
        {"event": "browser-nav", "source": source, "mode": "final", "view_width_s": width}
        for source, width in [
            ("keyboard", 300),
            ("plotly", 300),
            ("keyboard", 120),
            ("keyboard", 300),
        ]
    ]
    groups = group_samples(samples, warmup=1)
    assert [len(used) for _, _, used in groups] == [1, 0, 0]
    assert percentile([100, 200, 300], 50) == 200
    assert percentile([100, 200, 300], 95) == 290


def test_empty_run_reports_missing_browser_samples(tmp_path):
    (tmp_path / "app.log").write_text("[startup] app ready\n", encoding="utf-8")
    (tmp_path / "samples.csv").write_text("stale sample\n", encoding="utf-8")
    assert summarize(tmp_path) == []
    assert "No browser samples captured" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "samples.csv").read_text(encoding="utf-8") == ""


def test_unpaired_navigation_is_not_reported_as_autopan():
    samples = samples_from_events(
        parse_events(
            [
                "[resampler] total=90 ms, apply_path=direct-restyle",
                "[browser-nav] mode=final, source=keyboard, browser_total=100 ms",
            ]
        )
    )
    assert samples[0]["server_total_ms"] is None
    assert samples[0]["apply_path"] == ""


def test_report_preserves_raw_samples_and_documents_metric_limits(tmp_path):
    lines = [
        f"[browser-nav] profile_id={i}, mode=final, source=keyboard, "
        f"browser_total={100 + i} ms, x_width=300 s"
        for i in range(5)
    ]
    (tmp_path / "app.log").write_text("\n".join(lines), encoding="utf-8")
    summarize(tmp_path, warmup=3)
    report = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| 5 | 2 | 103.5 |" in report
    assert "does not await completed" in report
    assert len((tmp_path / "samples.csv").read_text(encoding="utf-8").splitlines()) == 6
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(events[0])["event"] == "browser-nav"


def test_record_captures_child_output_with_profiling_enabled(tmp_path, monkeypatch):
    # Exercise the real subprocess/tee/report path without opening the desktop app.
    (tmp_path / "run_desktop_app.py").write_text(
        "import os, sys\n"
        "assert os.environ['SLEEP_SCORING_PROFILE_RESAMPLER'] == '1'\n"
        "assert os.environ['SLEEP_SCORING_RESAMPLER_PERF_LOG'] == '1'\n"
        "assert os.environ['SLEEP_SCORING_BROWSER_NAV_PERF_LOG'] == '1'\n"
        "print('[browser-nav] profile_id=1, mode=final, source=keyboard, '"
        "'browser_total=123 ms, x_width=300 s')\n"
        "print('diagnostic on stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(measure_latency, "ROOT", tmp_path)
    monkeypatch.setattr(measure_latency, "check_profiling", lambda env: {"checked": True})
    monkeypatch.setattr(measure_latency, "assert_slots_free", lambda: None)
    monkeypatch.setattr(measure_latency, "metadata", lambda: {"commit": "fixture"})
    output = tmp_path / "run"
    assert measure_latency.record(output, warmup=0) == 0
    log = (output / "app.log").read_text(encoding="utf-8")
    assert "browser_total=123 ms" in log
    assert "diagnostic on stderr" in log
    assert "| 1 | 1 | 123.0 | 123.0 |" in (output / "summary.md").read_text(encoding="utf-8")
    assert json.loads((output / "profiling-check.json").read_text(encoding="utf-8"))["checked"]
    assert (output / "notes.md").is_file()
