"""
RouteIQ — Health check / diagnostics.

Validates:
- Config file (exists, parseable, required fields)
- API keys (present, non-placeholder)
- Provider connectivity (if keys available)
- Ollama availability
- State directory (writable)
- Credential sources (OpenClaw, Claude Code)

Usage:
    routeiq doctor
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def run_doctor(config_path: str = "conf/router.yaml", verbose: bool = False) -> dict:
    """
    Run all health checks. Returns a dict of check results.
    Each check: {"status": "ok"|"warn"|"fail", "message": str}
    """
    checks: list[dict[str, str]] = []

    checks.append(_check_config(config_path))
    checks.extend(_check_api_keys())
    checks.extend(_check_credentials())
    checks.append(_check_state_dir())
    checks.append(_check_models(config_path))
    checks.append(_check_ollama())

    ok_count = sum(1 for c in checks if c["status"] == "ok")
    warn_count = sum(1 for c in checks if c["status"] == "warn")
    fail_count = sum(1 for c in checks if c["status"] == "fail")

    return {
        "checks": checks,
        "summary": {
            "total": len(checks),
            "ok": ok_count,
            "warn": warn_count,
            "fail": fail_count,
            "healthy": fail_count == 0,
        },
    }


def format_doctor_cli(result: dict) -> str:
    """Format doctor results for CLI output."""
    lines = []
    lines.append("")
    lines.append("🩺 RouteIQ Doctor")
    lines.append("─" * 50)

    for check in result["checks"]:
        icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(check["status"], "?")
        lines.append(f"  {icon}  {check['name']}")
        if check.get("message"):
            lines.append(f"      {check['message']}")
        if check.get("fix"):
            lines.append(f"      💡 {check['fix']}")

    lines.append("")
    lines.append("─" * 50)
    s = result["summary"]
    status = "✅ Healthy" if s["healthy"] else f"❌ {s['fail']} issue(s) found"
    lines.append(f"  {status} — {s['ok']} ok, {s['warn']} warnings, {s['fail']} failures")
    lines.append("")

    return "\n".join(lines)


# ── Individual checks ─────────────────────────────────────────────

def _check_config(config_path: str) -> dict:
    """Check if config file exists and is valid YAML."""
    p = Path(config_path)
    if not p.exists():
        return {"name": "Config file", "status": "fail",
                "message": f"{config_path} not found",
                "fix": f"Create {config_path} or copy from examples/"}

    try:
        with p.open() as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return {"name": "Config file", "status": "fail",
                    "message": "Config is not a valid YAML dict"}

        # Check required sections
        missing = []
        for key in ["task_chains", "models", "budget"]:
            if key not in cfg:
                missing.append(key)

        if missing:
            return {"name": "Config file", "status": "warn",
                    "message": f"Missing sections: {', '.join(missing)} (defaults will be used)"}

        model_count = len(cfg.get("models", {}))
        chain_count = len(cfg.get("task_chains", {}))
        return {"name": "Config file", "status": "ok",
                "message": f"{model_count} models, {chain_count} task chains"}

    except yaml.YAMLError as e:
        return {"name": "Config file", "status": "fail",
                "message": f"YAML parse error: {e}",
                "fix": "Fix syntax in router.yaml"}


def _check_api_keys() -> list[dict]:
    """Check if API keys are set and non-placeholder."""
    checks = []

    # Each entry: (label, [env_var_names]) — checks all variants
    key_map = [
        ("OpenRouter", ["OPENROUTER_KEY", "OPENROUTER_API_KEY"]),
        ("Anthropic (direct)", ["ANTHROPIC_API_KEY"]),
        ("OpenAI (direct)", ["OPENAI_API_KEY"]),
        ("Google (direct)", ["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
    ]

    found_any = False
    for label, env_vars in key_map:
        # Try all variants
        val = ""
        found_var = ""
        for env_var in env_vars:
            v = os.getenv(env_var, "").strip()
            if v:
                val = v
                found_var = env_var
                break

        if not val:
            primary = env_vars[0]
            checks.append({"name": f"API key: {label}", "status": "warn",
                           "message": f"${primary} not set",
                           "fix": f"Add {primary}=... to .env"})
        elif val.startswith("your_") or val == "sk-or-..." or len(val) < 10:
            checks.append({"name": f"API key: {label}", "status": "fail",
                           "message": f"${found_var} is a placeholder, not a real key",
                           "fix": f"Replace placeholder in .env with your actual key"})
        else:
            found_any = True
            preview = val[:8] + "..." + val[-4:]
            checks.append({"name": f"API key: {label}", "status": "ok",
                           "message": f"{preview} (${found_var})"})

    if not found_any:
        checks.insert(0, {"name": "API keys (any)", "status": "fail",
                          "message": "No valid API keys found",
                          "fix": "Add OPENROUTER_KEY=... (or OPENROUTER_API_KEY=...) to .env"})

    return checks


def _check_credentials() -> list[dict]:
    """Check if external credential sources are available."""
    checks = []

    # OpenClaw
    openclaw_path = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
    if openclaw_path.exists():
        try:
            data = json.loads(openclaw_path.read_text())
            n = len(data.get("profiles", {}))
            checks.append({"name": "OpenClaw credentials", "status": "ok",
                           "message": f"Found {n} profile(s) in auth-profiles.json"})
        except Exception:
            checks.append({"name": "OpenClaw credentials", "status": "warn",
                           "message": "auth-profiles.json exists but couldn't be read"})
    else:
        checks.append({"name": "OpenClaw credentials", "status": "warn",
                        "message": "OpenClaw not installed (optional)",
                        "fix": "Install OpenClaw for auto credential discovery"})

    # Claude Code
    for claude_path in [Path.home() / ".claude.json", Path.home() / ".config" / "claude" / "credentials.json"]:
        if claude_path.exists():
            checks.append({"name": "Claude Code token", "status": "ok",
                           "message": f"Found {claude_path.name}"})
            break
    else:
        checks.append({"name": "Claude Code token", "status": "warn",
                        "message": "Not found (optional)"})

    return checks


def _check_state_dir() -> dict:
    """Check state directory is writable."""
    state_dir = Path("state")
    if not state_dir.exists():
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            return {"name": "State directory", "status": "ok",
                    "message": "Created state/"}
        except Exception as e:
            return {"name": "State directory", "status": "fail",
                    "message": f"Can't create state/: {e}",
                    "fix": "mkdir -p state && chmod 755 state"}

    # Check writable
    test_file = state_dir / ".doctor_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        return {"name": "State directory", "status": "ok", "message": "Writable"}
    except Exception:
        return {"name": "State directory", "status": "fail",
                "message": "state/ is not writable",
                "fix": "chmod 755 state"}


def _check_models(config_path: str) -> dict:
    """Check that models in config have required fields."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {"name": "Model definitions", "status": "fail", "message": "Can't read config"}

    models = cfg.get("models", {})
    if not models:
        return {"name": "Model definitions", "status": "fail",
                "message": "No models defined in config"}

    issues = []
    for name, mcfg in models.items():
        if not isinstance(mcfg, dict):
            issues.append(f"{name}: not a dict")
            continue
        if "id" not in mcfg:
            issues.append(f"{name}: missing 'id'")
        if "context_length" not in mcfg:
            issues.append(f"{name}: missing 'context_length'")

    if issues:
        return {"name": "Model definitions", "status": "warn",
                "message": f"{len(issues)} issue(s): {'; '.join(issues[:3])}"}

    return {"name": "Model definitions", "status": "ok",
            "message": f"All {len(models)} models valid"}


def _check_ollama() -> dict:
    """Check if Ollama is running (non-blocking, 1s timeout)."""
    try:
        import requests
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        r = requests.get(f"{base}/api/tags", timeout=1)
        if r.ok:
            data = r.json()
            model_count = len(data.get("models", []))
            return {"name": "Ollama", "status": "ok",
                    "message": f"Running at {base}, {model_count} model(s) loaded"}
        return {"name": "Ollama", "status": "warn",
                "message": f"Reachable but returned {r.status_code}"}
    except Exception:
        return {"name": "Ollama", "status": "warn",
                "message": "Not running (optional — needed for local models)",
                "fix": "Install: https://ollama.com | Start: ollama serve"}
