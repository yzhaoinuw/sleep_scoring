"""Record the existing desktop profiler and summarize repeated manual interactions.

Run in the sleep_scoring_dash3.0 environment; see paper/README.md.
No recording is opened automatically and no app configuration is edited.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
EVENT = re.compile(r"\[(browser-nav|browser-autopan|resampler-direct|resampler)\]\s+(.*)")
FIELDS = re.compile(r"(?:^|,\s*)([\w.]+)=([^,]*)")


def profiling_environment():
    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        SLEEP_SCORING_PROFILE_RESAMPLER="1",
        SLEEP_SCORING_RESAMPLER_PERF_LOG="1",
        SLEEP_SCORING_BROWSER_NAV_PERF_LOG="1",
        SLEEP_SCORING_SKIP_UPDATE="1",
        SLEEP_SCORING_INSTANCE_SLOT="0",
    )
    return env


def numeric(value):
    match = re.match(r"^(-?\d+(?:\.\d+)?)\b", value or "")
    return float(match.group(1)) if match else None


def parse_events(lines):
    events = []
    for line_number, line in enumerate(lines, 1):
        match = EVENT.search(line)
        if match:
            events.append(
                {"event": match[1], "line": line_number, **dict(FIELDS.findall(match[2]))}
            )
    return events


def samples_from_events(events):
    # Ordinary navigation has a shared ID; raw auto-pan endpoints do not.
    servers = {
        e["browser_profile_id"]: e
        for e in events
        if e["event"] == "resampler" and e.get("browser_profile_id") not in {None, "n/a"}
    }
    samples = []
    for event in events:
        if event["event"] not in {"browser-nav", "browser-autopan"}:
            continue
        total = numeric(event.get("browser_total"))
        if total is None or not math.isfinite(total) or total < 0:
            continue
        if event["event"] == "browser-autopan" and event.get("applied", "").casefold() != "true":
            continue
        server = servers.get(event.get("profile_id"), {}) if event["event"] == "browser-nav" else {}
        samples.append(
            {
                "event": event["event"],
                "line": event["line"],
                "profile_id": event.get("profile_id", event.get("id", "")),
                "source": event.get("source", "autopan"),
                "mode": event.get("mode", ""),
                "view_width_s": numeric(event.get("visible_width", event.get("x_width"))),
                "browser_total_ms": total,
                "coalesce_ms": numeric(event.get("coalesce")),
                "dash_apply_ms": numeric(event.get("dash_apply")),
                "queued_ms": numeric(event.get("queued")),
                "fetch_ms": numeric(event.get("fetch")),
                "parse_ms": numeric(event.get("parse")),
                "apply_call_ms": numeric(event.get("apply")),
                "server_total_ms": numeric(server.get("total")),
                "payload_kb": numeric(event.get("payload", server.get("payload"))),
                "apply_path": (
                    "raw-autopan"
                    if event["event"] == "browser-autopan"
                    else server.get("apply_path", "")
                ),
            }
        )
    return samples


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def group_samples(samples, warmup):
    groups = {}
    for sample in samples:
        key = (sample["event"], sample["source"], sample["mode"], sample["view_width_s"])
        groups.setdefault(key, []).append(sample)
    return [(key, rows, rows[warmup:]) for key, rows in groups.items()]


def summarize(run, warmup=3):
    log = run / "app.log" if run.is_dir() else run
    events = parse_events(log.read_text(encoding="utf-8", errors="replace").splitlines())
    samples = samples_from_events(events)
    output = log.parent
    (output / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    if samples:
        with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    else:
        (output / "samples.csv").write_text("", encoding="utf-8")
    report = [
        "# Manual latency run",
        "",
        f"Parsed {len(events)} profiler events and {len(samples)} valid browser samples.",
        f"Excluded the first {warmup} samples independently in each event/source/mode/view-width group.",
        "",
        "Navigation browser_total ends at the app's Plotly completion/profiler event and includes",
        "coalescing, callback delivery and trace update. Server time is included, not additive.",
        "Auto-pan browser_total ends after issuing the restyle call; it does not await completed",
        "rendering. Report auto-pan separately from navigation and do not call it paint latency.",
        "",
        "| Event / source / mode | View (s) | Recorded | Used | Median (ms) | p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    limited = []
    for key, rows, used in group_samples(samples, warmup):
        values = [s["browser_total_ms"] for s in used]
        median = f"{percentile(values, 50):.1f}" if values else "—"
        p95 = f"{percentile(values, 95):.1f}" if values else "—"
        report.append(
            f"| {' / '.join(key[:3])} | {key[3]} | {len(rows)} | {len(used)} | {median} | {p95} |"
        )
        if len(used) < 20:
            limited.append(f"- {' / '.join(key[:3])}, view {key[3]} s: {len(used)} retained.")
    if limited:
        report += ["", "Limited groups (fewer than 20 retained observations):", "", *limited]
    report += [
        "",
        "Groups with fewer than 20 retained observations are exploratory; do not publish",
        "a strong distribution or speedup claim from them. Use repeated, fixed-width tasks.",
        "Auto-pan samples within one drag are correlated; repeat separate drags too.",
        "",
        "Review metadata.json and notes.md before interpreting or sharing the results.",
        "Raw logs can contain recording paths. These local files are ignored by Git.",
    ]
    if not samples:
        report += [
            "",
            "**No browser samples captured.** Open a recording and navigate before closing",
            "the app. Confirm the first window slot and that profiling-check.json reports all flags true.",
        ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Summary: {output / 'summary.md'} ({len(samples)} browser samples)")
    return samples


def check_profiling(env):
    code = (
        "import json; from app_src import config; "
        "print(json.dumps({key: getattr(config, key) for key in "
        "['INSTANCE_SLOT','RESAMPLER_PERF_LOG','BROWSER_NAVIGATION_PERF_LOG',"
        "'PROFILE_RESAMPLER_UPDATES','ENABLE_DIRECT_PLOTLY_RESTYLE']}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, check=True, capture_output=True, text=True
    )
    flags = json.loads(result.stdout.strip())
    if flags["INSTANCE_SLOT"] != 0 or not all(
        flags[key]
        for key in (
            "RESAMPLER_PERF_LOG",
            "BROWSER_NAVIGATION_PERF_LOG",
            "PROFILE_RESAMPLER_UPDATES",
        )
    ):
        raise RuntimeError(f"Background profiling is not fully enabled: {flags}")
    return flags


def assert_slots_free():
    for port in range(8050, 8053):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"Port {port} is occupied. Close existing app windows before measuring; "
                    "profiling is disabled in later window slots."
                ) from error


def metadata():
    try:
        import psutil

        ram_gib = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        ram_gib = None

    versions = {}
    for name in ("dash", "plotly", "plotly-resampler", "numpy", "scipy", "pywebview"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    # Read only the literal setting; avoid importing the application in this process.
    settings = {}
    tree = ast.parse((ROOT / "app_src/config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key == "ENABLE_DIRECT_PLOTLY_RESTYLE":
                settings[key] = ast.literal_eval(node.value)
    return {
        "started_local": datetime.now().astimezone().isoformat(),
        "commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": ram_gib,
        "python": sys.version,
        "packages": versions,
        "settings": settings,
        "capture": "stdout/stderr tee; existing app profiler enabled by environment",
    }


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def record(output, warmup):
    env = profiling_environment()
    flags = check_profiling(env)
    assert_slots_free()
    output.mkdir(parents=True, exist_ok=False)
    (output / "profiling-check.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata(), indent=2), encoding="utf-8")
    (output / "notes.md").write_text(
        "# Run notes\n\n"
        "- Recording alias (avoid private identifiers in shared results):\n"
        "- Duration (s), EEG/EMG rate (Hz), NE rate (Hz), signal sample counts:\n"
        "- Sampling Level (keep x1 for the first run):\n"
        "- CPU/GPU model, window size, display scaling and embedded browser/runtime version:\n"
        "- Gesture, viewport width and trial count:\n"
        "- Warm-up, background apps, power mode, unusual pauses/errors:\n",
        encoding="utf-8",
    )
    print(f"Background profiling ON. Raw log: {output / 'app.log'}", flush=True)
    print("Choose your MAT file in the app. Close the app when finished for an automatic summary.")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "run_desktop_app.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        with (output / "app.log").open("w", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        return_code = proc.wait()
    except KeyboardInterrupt:
        print("\nStopping this measurement process.")
        stop_process(proc)
        return_code = proc.returncode
    finally:
        stop_process(proc)
        proc.stdout.close()
        summarize(output, warmup)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("record", help="Launch the desktop app with background profiling")
    launch.add_argument("--output", type=Path)
    launch.add_argument("--warmup", type=int, default=3)
    launch.add_argument(
        "--check", action="store_true", help="Verify logging without opening a window"
    )
    summary = commands.add_parser(
        "summarize", help="Rebuild a report from a run directory or app.log"
    )
    summary.add_argument("run", type=Path)
    summary.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.command == "summarize":
        summarize(args.run, args.warmup)
        return 0
    if args.check:
        print(json.dumps(check_profiling(profiling_environment()), indent=2))
        return 0
    output = args.output or ROOT / "paper/latency_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    return record(output.resolve(), args.warmup)


if __name__ == "__main__":
    raise SystemExit(main())
