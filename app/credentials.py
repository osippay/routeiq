"""
RouteIQ — Credential discovery.

Auto-discovers API keys and tokens from multiple sources, in priority order:

1. Environment variables (.env)  — explicit, always wins
2. OpenClaw auth-profiles.json   — if OpenClaw is installed
3. Claude Code setup-token       — if Claude Code has been configured
4. NadirClaw credentials.json    — if NadirClaw is installed

This means: if you have OpenClaw set up with OAuth tokens,
RouteIQ will auto-discover and use them without any configuration.

File locations:
  OpenClaw:    ~/.openclaw/agents/main/agent/auth-profiles.json
  Claude Code: ~/.claude/credentials.json  (or ~/.config/claude/credentials.json)
  NadirClaw:   ~/.nadirclaw/credentials.json
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Provider → env var name mapping
ENV_KEY_MAP = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_KEY", "OPENROUTER_API_KEY"],
}

# Track what we discovered (for status reporting)
_discovery_log: list[dict[str, str]] = []


def discover_credentials() -> dict[str, str]:
    """
    Discover all available credentials from all sources.

    Returns a dict of {provider: api_key_or_token}.
    Does NOT override existing env vars — env always has priority.
    """
    global _discovery_log
    _discovery_log = []

    creds: dict[str, str] = {}

    # 1. Check env vars first (highest priority)
    for provider, env_names in ENV_KEY_MAP.items():
        for env_name in env_names:
            val = os.getenv(env_name, "").strip()
            if val and not val.startswith("your_"):
                creds[provider] = val
                _discovery_log.append({
                    "provider": provider, "source": "env", "key": env_name,
                })
                break

    # 2. OpenClaw auth-profiles
    openclaw_creds = _read_openclaw_profiles()
    for provider, key in openclaw_creds.items():
        if provider not in creds:
            creds[provider] = key
            _discovery_log.append({
                "provider": provider, "source": "openclaw",
                "key": "auth-profiles.json",
            })

    # 3. Claude Code setup-token
    claude_token = _read_claude_code_token()
    if claude_token and "anthropic" not in creds:
        creds["anthropic"] = claude_token
        _discovery_log.append({
            "provider": "anthropic", "source": "claude-code",
            "key": "setup-token",
        })

    # 4. NadirClaw credentials
    nadirclaw_creds = _read_nadirclaw_credentials()
    for provider, key in nadirclaw_creds.items():
        if provider not in creds:
            creds[provider] = key
            _discovery_log.append({
                "provider": provider, "source": "nadirclaw",
                "key": "credentials.json",
            })

    return creds


def apply_discovered_credentials() -> dict[str, str]:
    """
    Discover credentials and inject them into environment variables
    so backends can pick them up via os.getenv().

    Only sets env vars that aren't already set.
    Returns the full credentials dict.
    """
    creds = discover_credentials()

    # Map provider → env var to set
    provider_to_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_KEY",
    }

    for provider, key in creds.items():
        env_name = provider_to_env.get(provider)
        if env_name and not os.getenv(env_name):
            os.environ[env_name] = key
            logger.info("Credential discovered: %s via %s",
                        provider, _get_source(provider))

    return creds


def get_discovery_status() -> list[dict[str, str]]:
    """Return what was discovered and from where (for CLI/API status)."""
    if not _discovery_log:
        discover_credentials()
    return list(_discovery_log)


def _get_source(provider: str) -> str:
    for entry in _discovery_log:
        if entry["provider"] == provider:
            return entry["source"]
    return "unknown"


# ── OpenClaw reader ───────────────────────────────────────────────

def _read_openclaw_profiles() -> dict[str, str]:
    """
    Read credentials from OpenClaw's auth-profiles.json.

    File: ~/.openclaw/agents/main/agent/auth-profiles.json

    Format:
    {
      "profiles": {
        "anthropic:default": {
          "type": "api_key",
          "provider": "anthropic",
          "key": "sk-ant-..."
        },
        "openai:default": {
          "type": "api_key",
          "provider": "openai",
          "key": "sk-..."
        },
        "anthropic:oauth": {
          "type": "token",
          "provider": "anthropic",
          "token": "...",
          "access": "...",
          "refresh": "...",
          "expires": 1234567890
        }
      }
    }
    """
    creds: dict[str, str] = {}

    # Check multiple possible locations
    home = Path.home()
    candidates = [
        home / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json",
        # Some setups use a different state dir
        Path(os.getenv("CLAWDBOT_STATE_DIR", "")) / "agents" / "main" / "agent" / "auth-profiles.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles = data.get("profiles", {})

            for profile_id, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue

                provider = profile.get("provider", "").lower()
                if not provider:
                    # Try to extract from profile_id: "anthropic:default" → "anthropic"
                    if ":" in profile_id:
                        provider = profile_id.split(":")[0].lower()

                if not provider:
                    continue

                # Normalize provider names
                provider = _normalize_provider(provider)

                # Already have this provider? Skip (first match wins)
                if provider in creds:
                    continue

                # Extract the actual key/token
                key = _extract_key_from_profile(profile)
                if key:
                    creds[provider] = key
                    logger.debug("OpenClaw: found %s credential (%s)", provider, profile_id)

            if creds:
                logger.info("OpenClaw: discovered %d credential(s) from %s",
                            len(creds), path)
            break  # Stop after first valid file

        except (json.JSONDecodeError, PermissionError, OSError) as e:
            logger.debug("OpenClaw: couldn't read %s: %s", path, e)

    return creds


def _extract_key_from_profile(profile: dict) -> str | None:
    """Extract usable API key or token from an OpenClaw auth profile."""
    profile_type = profile.get("type", "")

    # API key — straightforward
    if profile_type == "api_key":
        return profile.get("key") or profile.get("val") or None

    # Token (OAuth or setup-token)
    if profile_type == "token":
        # Prefer access token (OAuth), fall back to raw token
        access = profile.get("access")
        if access:
            # Check expiry
            expires = profile.get("expires", 0)
            if expires > 0:
                import time
                if time.time() > expires:
                    logger.debug("OpenClaw: token expired for profile")
                    return None
            return access

        return profile.get("token") or None

    # SecretRef — we can't resolve exec-based refs, skip those
    if profile.get("keyRef") or profile.get("tokenRef"):
        logger.debug("OpenClaw: skipping SecretRef profile (not supported)")
        return None

    return None


# ── Claude Code reader ────────────────────────────────────────────

def _read_claude_code_token() -> str | None:
    """
    Read Anthropic setup-token from Claude Code's credential store.

    Claude Code stores its token after `claude setup-token` in:
    - ~/.claude.json (legacy)
    - ~/.config/claude/credentials.json

    The token format is a long JWT-like string starting with specific prefix.
    """
    home = Path.home()
    candidates = [
        home / ".claude.json",
        home / ".config" / "claude" / "credentials.json",
        home / ".claude" / "credentials.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            # Direct token field
            token = data.get("token") or data.get("apiKey") or data.get("api_key")
            if token and isinstance(token, str) and len(token) > 20:
                logger.info("Claude Code: found token in %s", path)
                return token

            # Nested under anthropic
            anthropic = data.get("anthropic", {})
            if isinstance(anthropic, dict):
                token = anthropic.get("token") or anthropic.get("apiKey")
                if token and isinstance(token, str):
                    logger.info("Claude Code: found anthropic token in %s", path)
                    return token

        except (json.JSONDecodeError, PermissionError, OSError) as e:
            logger.debug("Claude Code: couldn't read %s: %s", path, e)

    return None


# ── NadirClaw reader ──────────────────────────────────────────────

def _read_nadirclaw_credentials() -> dict[str, str]:
    """
    Read credentials from NadirClaw's credential store.

    File: ~/.nadirclaw/credentials.json

    Format:
    {
      "google": {"key": "AIza..."},
      "anthropic": {"key": "sk-ant-..."},
      "openai": {"key": "sk-..."}
    }
    """
    creds: dict[str, str] = {}
    path = Path.home() / ".nadirclaw" / "credentials.json"

    if not path.exists():
        return creds

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for provider, entry in data.items():
            if isinstance(entry, dict):
                key = entry.get("key") or entry.get("token") or entry.get("api_key")
            elif isinstance(entry, str):
                key = entry
            else:
                continue

            if key and isinstance(key, str) and len(key) > 10:
                provider = _normalize_provider(provider.lower())
                creds[provider] = key
                logger.debug("NadirClaw: found %s credential", provider)

        if creds:
            logger.info("NadirClaw: discovered %d credential(s)", len(creds))

    except (json.JSONDecodeError, PermissionError, OSError) as e:
        logger.debug("NadirClaw: couldn't read %s: %s", path, e)

    return creds


# ── Helpers ───────────────────────────────────────────────────────

def _normalize_provider(name: str) -> str:
    """Normalize provider names to our canonical form."""
    aliases = {
        "google": "google",
        "gemini": "google",
        "google-ai": "google",
        "vertex": "google",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "openrouter": "openrouter",
    }
    return aliases.get(name, name)
