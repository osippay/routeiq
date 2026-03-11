"""
RouteIQ — Live Terminal Dashboard.

Usage:
    routeiq dashboard
    routeiq dashboard --refresh 2

Requires: pip install rich
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_logs(path: str = "state/model-stats.jsonl", limit: int = 500) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    try:
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries[-limit:]
    except Exception:
        return []


def load_state(path: str = "state/state.json") -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


TASK_STYLES = {
    "code": "bold green", "text": "white", "summarize": "bold cyan",
    "think": "bold magenta", "strategy": "bold red", "vision": "bold yellow",
    "image": "yellow", "audio": "bold blue",
}


def build_dashboard(start_time: float, max_requests: int = 12):
    from rich.console import Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich import box

    entries = load_logs()
    state = load_state()
    now = time.time()
    total = len(entries)

    total_cost = sum(e.get("cost_usd", 0) for e in entries)
    tokens_in = sum(e.get("tokens_in", 0) for e in entries)
    tokens_out = sum(e.get("tokens_out", 0) for e in entries)
    without = (tokens_in * 0.003 + tokens_out * 0.015) / 1000
    saved = max(0, without - total_cost)
    saved_pct = (saved / without * 100) if without > 0 else 0
    balance = state.get("balance_usd", 100.0)

    recent_5m = [e for e in entries if now - e.get("ts", 0) < 300]
    rpm = len(recent_5m) / 5.0 if recent_5m else 0.0

    uptime = now - start_time
    hrs, rem = divmod(int(uptime), 3600)
    mins, secs = divmod(rem, 60)

    from app import __version__

    # ── Header ──
    header = Text()
    header.append("⚡ RouteIQ", style="bold cyan")
    header.append(f" v{__version__}", style="dim cyan")
    header.append(" — Smart LLM Router", style="dim")

    info = Text()
    info.append(f"{hrs}h {mins:02d}m {secs:02d}s", style="dim")
    info.append("  │  ", style="dim")
    info.append("Ctrl+C to quit", style="dim")

    # ── Stats ──
    stats = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    stats.add_column("Label", style="white", ratio=2)
    stats.add_column("Value", justify="right", ratio=1)

    stats.add_row("Total Requests", f"[bold]{total}[/]")
    stats.add_row("Req/min (5m)", f"[green]{rpm:.1f}[/]")
    stats.add_row("Actual Cost", f"${total_cost:.4f}")
    stats.add_row("Without Routing", f"[dim]${without:.4f}[/]")
    stats.add_row("Saved", f"[bold green]${saved:.4f} ({saved_pct:.1f}%)[/]")
    stats.add_row("Balance", f"[bold cyan]${balance:.2f}[/]")

    # ── Routing Distribution ──
    by_task: dict[str, int] = defaultdict(int)
    for e in entries:
        by_task[e.get("task_type", "other")] += 1

    dist = Table(box=None, padding=(0, 1), expand=True)
    dist.add_column("Task", width=10)
    dist.add_column("Count", justify="right", width=5)
    dist.add_column("", width=22)
    dist.add_column("%", justify="right", width=6)

    for task, count in sorted(by_task.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        style = TASK_STYLES.get(task, "dim")
        filled = int(count / total * 20) if total > 0 else 0
        bar = f"[{style}]{'█' * filled}[/][dim]{'░' * (20 - filled)}[/]"
        dist.add_row(f"[{style}]{task}[/]", str(count), bar, f"{pct:.1f}%")

    # ── Recent Requests ──
    recent = list(reversed(entries[-max_requests:]))

    req = Table(box=None, padding=(0, 1), expand=True)
    req.add_column("Time", width=8, style="dim")
    req.add_column("Task", width=10)
    req.add_column("Model", min_width=14)
    req.add_column("Latency", justify="right", width=8)
    req.add_column("Tokens", justify="right", width=6)
    req.add_column("Cost", justify="right", width=10)

    for e in recent:
        ts = e.get("ts", 0)
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
        task = e.get("task_type", "?")
        model = e.get("model", "?")
        lat = e.get("latency_ms", 0)
        toks = e.get("tokens_in", 0) + e.get("tokens_out", 0)
        cost = e.get("cost_usd", 0)

        ts_style = TASK_STYLES.get(task, "white")
        lat_style = "green" if lat < 500 else ("yellow" if lat < 2000 else "red")
        cost_style = "bold green" if cost == 0 else ("green" if cost < 0.001 else ("yellow" if cost < 0.01 else "bold red"))

        req.add_row(
            tstr, f"[{ts_style}]{task}[/]", model,
            f"[{lat_style}]{lat:.0f}ms[/]", str(toks), f"[{cost_style}]${cost:.5f}[/]",
        )

    return Group(
        header,
        info,
        Rule(style="bright_blue"),
        Panel(stats, title="[bold yellow]⚡ Stats[/]", border_style="bright_blue", expand=True),
        Panel(dist, title="[bold yellow]📊 Routing Distribution[/]", border_style="bright_blue", expand=True),
        Panel(req, title="[bold yellow]📋 Recent Requests[/]", border_style="bright_blue", expand=True),
    )


def run_dashboard(refresh_seconds: float = 1.0):
    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        print("Dashboard requires 'rich'. Install: pip install rich")
        return

    console = Console()
    start = time.time()

    try:
        with Live(build_dashboard(start), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(refresh_seconds)
                live.update(build_dashboard(start))
    except KeyboardInterrupt:
        console.print("[dim]Stopped.[/]")


if __name__ == "__main__":
    run_dashboard()
