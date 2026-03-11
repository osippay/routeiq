"""
RouteIQ — Analytics & reporting.

Parses model-stats.jsonl and generates reports:
- Cost breakdown by model
- Latency percentiles
- Task type distribution
- Cache hit rate
- Hourly/daily spend trends

Usage:
    python cli.py report
    python cli.py report --days 7
    python cli.py report --model sonnet
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_logs(path: str = "state/model-stats.jsonl", days: int = 0) -> list[dict]:
    """Load log entries from JSONL file, optionally filtered by recency."""
    p = Path(path)
    if not p.exists():
        return []

    entries = []
    cutoff = time.time() - (days * 86400) if days > 0 else 0

    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if cutoff and entry.get("ts", 0) < cutoff:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error("Failed to read logs from %s: %s", p, e)

    return entries


def generate_report(
    log_path: str = "state/model-stats.jsonl",
    days: int = 0,
    model_filter: str | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive analytics report.

    Returns a dict with all stats (for JSON output or CLI display).
    """
    entries = load_logs(log_path, days)

    if model_filter:
        entries = [e for e in entries if e.get("model") == model_filter]

    if not entries:
        return {"total_requests": 0, "message": "No data found for the given filters."}

    # ── Basic stats ──
    total_requests = len(entries)
    total_cost = sum(e.get("cost_usd", 0) for e in entries)
    total_tokens_in = sum(e.get("tokens_in", 0) for e in entries)
    total_tokens_out = sum(e.get("tokens_out", 0) for e in entries)
    cached_count = sum(1 for e in entries if e.get("cached"))
    streamed_count = sum(1 for e in entries if e.get("stream"))

    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]
    latencies.sort()

    # ── Per-model breakdown ──
    by_model: dict[str, dict] = defaultdict(lambda: {
        "requests": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
        "latencies": [], "errors": 0,
    })
    for e in entries:
        m = e.get("model", "unknown")
        by_model[m]["requests"] += 1
        by_model[m]["cost_usd"] += e.get("cost_usd", 0)
        by_model[m]["tokens_in"] += e.get("tokens_in", 0)
        by_model[m]["tokens_out"] += e.get("tokens_out", 0)
        if e.get("latency_ms", 0) > 0:
            by_model[m]["latencies"].append(e["latency_ms"])

    model_stats = {}
    for m, stats in sorted(by_model.items(), key=lambda x: x[1]["cost_usd"], reverse=True):
        lats = sorted(stats["latencies"])
        model_stats[m] = {
            "requests": stats["requests"],
            "cost_usd": round(stats["cost_usd"], 5),
            "pct_of_total_cost": round(stats["cost_usd"] / total_cost * 100, 1) if total_cost > 0 else 0,
            "tokens_in": stats["tokens_in"],
            "tokens_out": stats["tokens_out"],
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
            "p50_latency_ms": round(_percentile(lats, 50), 1) if lats else 0,
            "p95_latency_ms": round(_percentile(lats, 95), 1) if lats else 0,
            "p99_latency_ms": round(_percentile(lats, 99), 1) if lats else 0,
        }

    # ── Per-task breakdown ──
    by_task: dict[str, dict] = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0})
    for e in entries:
        t = e.get("task_type", "unknown")
        by_task[t]["requests"] += 1
        by_task[t]["cost_usd"] += e.get("cost_usd", 0)

    task_stats = {
        t: {"requests": s["requests"], "cost_usd": round(s["cost_usd"], 5),
            "pct_requests": round(s["requests"] / total_requests * 100, 1)}
        for t, s in sorted(by_task.items(), key=lambda x: x[1]["requests"], reverse=True)
    }

    # ── Hourly trend (last 24h) ──
    now = time.time()
    hourly: dict[int, float] = defaultdict(float)
    for e in entries:
        hours_ago = int((now - e.get("ts", now)) / 3600)
        if hours_ago < 24:
            hourly[hours_ago] += e.get("cost_usd", 0)
    hourly_trend = {f"{h}h_ago": round(c, 5) for h, c in sorted(hourly.items())}

    # ── Time range ──
    timestamps = [e.get("ts", 0) for e in entries]
    first_ts = min(timestamps) if timestamps else 0
    last_ts = max(timestamps) if timestamps else 0

    import datetime
    first_dt = datetime.datetime.fromtimestamp(first_ts, tz=datetime.timezone.utc).isoformat() if first_ts else None
    last_dt = datetime.datetime.fromtimestamp(last_ts, tz=datetime.timezone.utc).isoformat() if last_ts else None

    return {
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 5),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "cached_requests": cached_count,
        "streamed_requests": streamed_count,
        "cache_hit_rate": round(cached_count / total_requests, 3) if total_requests > 0 else 0,
        "avg_cost_per_request": round(total_cost / total_requests, 6) if total_requests > 0 else 0,
        "latency": {
            "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50_ms": round(_percentile(latencies, 50), 1) if latencies else 0,
            "p95_ms": round(_percentile(latencies, 95), 1) if latencies else 0,
            "p99_ms": round(_percentile(latencies, 99), 1) if latencies else 0,
            "min_ms": round(min(latencies), 1) if latencies else 0,
            "max_ms": round(max(latencies), 1) if latencies else 0,
        },
        "by_model": model_stats,
        "by_task": task_stats,
        "hourly_cost_trend": hourly_trend,
        "time_range": {"first": first_dt, "last": last_dt},
        "filter": {"days": days or "all", "model": model_filter},
    }


def format_report_cli(report: dict) -> str:
    """Format report dict as human-readable CLI output."""
    if report.get("total_requests", 0) == 0:
        return "No data found. Run some requests first!"

    lines = []
    lines.append("=" * 60)
    lines.append("  RouteIQ Analytics Report")
    lines.append("=" * 60)
    lines.append("")

    tr = report["time_range"]
    lines.append(f"  Period: {tr.get('first', '?')} → {tr.get('last', '?')}")
    lines.append(f"  Filter: days={report['filter']['days']}, model={report['filter']['model'] or 'all'}")
    lines.append("")

    lines.append("── Overview ──")
    lines.append(f"  Total requests:    {report['total_requests']}")
    lines.append(f"  Total cost:        ${report['total_cost_usd']:.5f}")
    lines.append(f"  Avg cost/request:  ${report['avg_cost_per_request']:.6f}")
    lines.append(f"  Cached requests:   {report['cached_requests']} ({report['cache_hit_rate']:.1%})")
    lines.append(f"  Streamed:          {report['streamed_requests']}")
    lines.append(f"  Total tokens:      {report['total_tokens_in']} in / {report['total_tokens_out']} out")
    lines.append("")

    lat = report["latency"]
    lines.append("── Latency ──")
    lines.append(f"  Avg:  {lat['avg_ms']}ms")
    lines.append(f"  P50:  {lat['p50_ms']}ms  |  P95: {lat['p95_ms']}ms  |  P99: {lat['p99_ms']}ms")
    lines.append(f"  Min:  {lat['min_ms']}ms  |  Max: {lat['max_ms']}ms")
    lines.append("")

    lines.append("── By Model ──")
    for model, stats in report.get("by_model", {}).items():
        pct = stats["pct_of_total_cost"]
        lines.append(f"  {model}")
        lines.append(f"    {stats['requests']} reqs | ${stats['cost_usd']:.5f} ({pct:.1f}%) | p50={stats['p50_latency_ms']}ms p95={stats['p95_latency_ms']}ms")

    lines.append("")
    lines.append("── By Task Type ──")
    for task, stats in report.get("by_task", {}).items():
        lines.append(f"  {task:12s}  {stats['requests']:4d} reqs ({stats['pct_requests']:.1f}%)  ${stats['cost_usd']:.5f}")

    if report.get("hourly_cost_trend"):
        lines.append("")
        lines.append("── Hourly Cost (last 24h) ──")
        max_cost = max(report["hourly_cost_trend"].values()) or 1
        for label, cost in report["hourly_cost_trend"].items():
            bar_len = int((cost / max_cost) * 30) if max_cost > 0 else 0
            bar = "█" * max(1, bar_len)
            lines.append(f"  {label:>6s}  ${cost:.5f}  {bar}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Calculate percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]
