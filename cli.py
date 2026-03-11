#!/usr/bin/env python3
"""
RouteIQ CLI — test, serve, monitor, and report.

Prompts:
  routeiq "write a fibonacci function"
  routeiq --stream "tell me a story"
  routeiq --model sonnet "review this code"
  routeiq --profile eco "what is recursion"

Commands:
  routeiq serve              Start OpenAI-compatible HTTP proxy
  routeiq dashboard          Live terminal dashboard
  routeiq report             Analytics report from logs
  routeiq doctor             Health check
  routeiq models             List models and aliases
  routeiq credentials        Show discovered API keys
  routeiq status             Full status (JSON)
  routeiq budget             Budget status (JSON)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Load .env
for envfile in [Path(".env"), Path(Path.home() / ".routeiq" / ".env")]:
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break

sys.path.insert(0, str(Path(__file__).parent))
from app.router import Router, RouterRequest, MODE_SETTINGS
from app.policy import ROUTING_PROFILES

# Known subcommands
COMMANDS = {"report", "serve", "dashboard", "models", "credentials", "doctor", "status", "budget"}


def main() -> None:
    # ── Smart detection: is the user sending a prompt or a command? ──
    # If first non-flag arg is not a known command → treat everything as a prompt call
    args_raw = sys.argv[1:]

    # Find the first non-flag argument
    first_positional = None
    for arg in args_raw:
        if not arg.startswith("-"):
            first_positional = arg
            break

    if first_positional and first_positional not in COMMANDS:
        # User is sending a prompt → route to prompt handler
        _handle_prompt(args_raw)
        return

    # Otherwise → standard subcommand dispatch
    _handle_command(args_raw)


def _handle_prompt(args_raw: list[str]) -> None:
    """Handle: routeiq "write hello world" --stream --model sonnet"""
    parser = argparse.ArgumentParser(description="RouteIQ — Send a prompt")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--task", help="Force task type")
    parser.add_argument("--mode", default="medium", choices=list(MODE_SETTINGS.keys()))
    parser.add_argument("--profile", choices=list(ROUTING_PROFILES.keys()), help="Routing profile")
    parser.add_argument("--model", type=str, help="Force model (name or alias)")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--session", type=str, help="Session ID for model pinning")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(args_raw)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    cmd_prompt(args)


def _handle_command(args_raw: list[str]) -> None:
    """Handle: routeiq serve, routeiq report, etc."""
    parser = argparse.ArgumentParser(
        description="RouteIQ — Smart LLM Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "write a fibonacci function"
  %(prog)s --stream "tell me a story"
  %(prog)s --model sonnet "review this code"
  %(prog)s serve
  %(prog)s dashboard
  %(prog)s report
  %(prog)s doctor
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    p_report = sub.add_parser("report", help="Analytics report from logs")
    p_report.add_argument("--days", type=int, default=0)
    p_report.add_argument("--model", type=str)
    p_report.add_argument("--json", action="store_true")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible HTTP proxy")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    p_dash = sub.add_parser("dashboard", help="Live terminal dashboard")
    p_dash.add_argument("--refresh", type=float, default=1.0)

    sub.add_parser("models", help="List available models and aliases")
    sub.add_parser("credentials", help="Show discovered credentials")

    p_doctor = sub.add_parser("doctor", help="Run health checks")
    p_doctor.add_argument("--json", action="store_true")

    sub.add_parser("status", help="Show full status (JSON)")
    sub.add_parser("budget", help="Show budget status (JSON)")

    args = parser.parse_args(args_raw)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    dispatch = {
        "report": cmd_report,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
        "models": cmd_models,
        "credentials": cmd_credentials,
        "doctor": cmd_doctor,
        "status": cmd_status,
        "budget": cmd_budget,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════

def cmd_prompt(args) -> None:
    """Send a prompt to the router."""
    router = Router()

    req = RouterRequest(
        messages=[{"role": "user", "content": args.prompt}],
        task_type=getattr(args, "task", None),
        mode=getattr(args, "mode", "medium"),
        profile=getattr(args, "profile", None),
        model=getattr(args, "model", None),
        max_tokens=getattr(args, "max_tokens", None),
        stream=getattr(args, "stream", False),
        session_id=getattr(args, "session", None),
    )

    profile_desc = ""
    if getattr(args, "profile", None) and args.profile in ROUTING_PROFILES:
        profile_desc = f" | Profile: {ROUTING_PROFILES[args.profile]['description']}"

    mode_desc = MODE_SETTINGS.get(getattr(args, "mode", "medium"), {}).get("description", "")
    print(f"🔍 Task: {getattr(args, 'task', None) or 'auto'} | Mode: {getattr(args, 'mode', 'medium')} {mode_desc}{profile_desc}")

    if getattr(args, "model", None):
        print(f"📌 Model: {args.model}")

    if getattr(args, "stream", False):
        print("⏳ Streaming...\n" + "─" * 60)
        try:
            for chunk in router.route_stream(req):
                print(chunk, end="", flush=True)
            print("\n" + "─" * 60)
        except RuntimeError as e:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("⏳ Routing...\n")
        try:
            resp = router.route(req)
        except RuntimeError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)

        tags = []
        if resp.cached:
            tags.append("CACHED")
        if resp.is_agentic:
            tags.append("AGENTIC")
        if resp.is_reasoning:
            tags.append("REASONING")
        tag_str = " [" + ", ".join(tags) + "]" if tags else ""

        print(f"✅ Model: {resp.model_used} via {resp.provider}{tag_str}")
        print(f"💰 Cost: ${resp.cost_usd:.5f} | Tokens: {resp.tokens_in}→{resp.tokens_out} | {resp.latency_ms:.0f}ms\n")
        print("─" * 60)
        print(resp.content)


def cmd_report(args) -> None:
    """Generate analytics report."""
    from app.analytics import generate_report, format_report_cli
    report = generate_report(
        log_path="state/model-stats.jsonl",
        days=args.days,
        model_filter=getattr(args, "model", None),
    )
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report_cli(report))


def cmd_status(args) -> None:
    """Show full status."""
    router = Router()
    print(json.dumps(router.full_status(), indent=2, ensure_ascii=False))


def cmd_budget(args) -> None:
    """Show budget."""
    router = Router()
    print(json.dumps(router.budget_status(), indent=2, ensure_ascii=False))


def cmd_doctor(args) -> None:
    """Run health checks."""
    from app.doctor import run_doctor, format_doctor_cli
    result = run_doctor()
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_doctor_cli(result))


def cmd_serve(args) -> None:
    """Start HTTP proxy."""
    import uvicorn
    from app.server import app
    print(f"\n🚀 RouteIQ proxy starting on http://{args.host}:{args.port}/v1\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_dashboard(args) -> None:
    """Launch live terminal dashboard."""
    from app.dashboard import run_dashboard
    run_dashboard(refresh_seconds=args.refresh)


def cmd_models(args) -> None:
    """List available models and aliases."""
    from app.policy import TaskPolicy
    policy = TaskPolicy()

    print("\n📋 Available models:\n")
    models = policy.raw_config.get("models", {})
    for name, cfg in models.items():
        model_id = cfg.get("id", "?")
        free = "FREE" if cfg.get("free") else f"${cfg.get('cost_per_1k_input', 0)}/{cfg.get('cost_per_1k_output', 0)} per 1k"
        ctx = cfg.get("context_length", "?")
        caps = ", ".join(cfg.get("capabilities", []))
        provider = cfg.get("provider", "openrouter")
        print(f"  {name:16s}  {model_id}")
        print(f"                    {free} | ctx={ctx} | {caps} | via {provider}")

    print("\n🏷️  Aliases:\n")
    for alias, target in sorted(policy.DEFAULT_ALIASES.items()):
        if alias != target:
            print(f"  {alias:16s} → {target}")

    print(f"\n🔄 Routing profiles: {', '.join(ROUTING_PROFILES.keys())}\n")
    for name, prof in ROUTING_PROFILES.items():
        print(f"  {name:12s}  {prof['description']}")
    print()


def cmd_credentials(args) -> None:
    """Show discovered credentials and their sources."""
    from app.credentials import discover_credentials, get_discovery_status

    creds = discover_credentials()
    status = get_discovery_status()

    print("\n🔑 Credential Discovery\n")

    if not status:
        print("  ⚠️  No credentials found!")
        print("  Set API keys in .env, or install OpenClaw for auto-discovery.\n")
        return

    sources = {}
    for entry in status:
        src = entry["source"]
        sources.setdefault(src, []).append(entry)

    source_labels = {
        "env": "📁 Environment variables (.env)",
        "openclaw": "🐾 OpenClaw (auto-discovered)",
        "claude-code": "🤖 Claude Code (setup-token)",
        "nadirclaw": "🔄 NadirClaw (credentials.json)",
    }

    for source, entries in sources.items():
        label = source_labels.get(source, source)
        print(f"  {label}")
        for e in entries:
            provider = e["provider"]
            key_preview = creds.get(provider, "")[:8] + "..." if creds.get(provider) else "?"
            print(f"    ✅ {provider:12s}  {key_preview}  ({e['key']})")
        print()

    all_providers = {"anthropic", "openai", "google", "openrouter"}
    found = {e["provider"] for e in status}
    missing = all_providers - found
    if missing:
        print(f"  ❌ Not found: {', '.join(sorted(missing))}")
        print("     These providers will not be available for direct routing.\n")


if __name__ == "__main__":
    main()
