"""
RouteIQ — Core router (v2.1).

Full pipeline:
  classify → detect agentic/reasoning → apply profile → score →
  filter by context → check cache → select backend → call → track

New in v2.1:
- Multi-provider backends (OpenRouter, Anthropic, OpenAI, Google, Ollama)
- Agentic detection (tool use → force complex model)
- Reasoning detection (CoT markers → reasoning model)
- Routing profiles (auto, eco, premium, free, reasoning)
- Model aliases (sonnet, flash, gpt4 → full model names)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from .alerts import AlertManager
from .backends import get_backend, detect_provider
from .budget import BudgetManager
from .cache import ResponseCache
from .classifier import classify_task, classify_with_modifiers
from .policy import TaskPolicy, ROUTING_PROFILES
from .session import SessionManager
from .storage import append_jsonl

logger = logging.getLogger(__name__)

# Quality modes (kept for backward compat, profiles are preferred)
MODE_SETTINGS = {
    "easy": {"max_tokens": 512, "temperature": 0.5, "chain_limit": 1, "description": "⚡ Fast & cheap"},
    "medium": {"max_tokens": 1024, "temperature": 0.7, "chain_limit": 3, "description": "⚖️ Balanced"},
    "hard": {"max_tokens": 4096, "temperature": 0.8, "chain_limit": 99, "chain_reverse": True, "description": "🔥 Best quality"},
    "god": {"max_tokens": 8192, "temperature": 0.9, "chain_limit": 1, "chain_reverse": True, "force_model": "opus", "description": "👑 God mode"},
}


@dataclass
class RouterRequest:
    messages: list[dict]
    task_type: str | None = None
    hint: str | None = None
    mode: str = "medium"
    profile: str | None = None        # NEW: routing profile (auto/eco/premium/free/reasoning)
    model: str | None = None          # NEW: explicit model or alias
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    session_id: str | None = None
    tools: list | None = None         # NEW: tool definitions (for agentic detection)
    tool_choice: str | None = None    # NEW: tool_choice param
    metadata: dict = field(default_factory=dict)


@dataclass
class RouterResponse:
    content: str
    model_used: str
    provider: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    task_type: str
    cached: bool = False
    is_agentic: bool = False
    is_reasoning: bool = False
    tool_calls: list | None = None    # pass through from model


class CircuitBreaker:
    def __init__(self, max_errors: int = 5, window_s: float = 60.0) -> None:
        self._max = max_errors
        self._window = window_s
        self._errors: dict[str, list[float]] = {}
        self._open: dict[str, float] = {}

    def is_open(self, model: str) -> bool:
        now = time.time()
        if model in self._open:
            if now - self._open[model] < self._window:
                return True
            del self._open[model]
        return False

    def record_failure(self, model: str) -> None:
        now = time.time()
        errors = [t for t in self._errors.get(model, []) if now - t < self._window]
        errors.append(now)
        self._errors[model] = errors
        if len(errors) >= self._max:
            self._open[model] = now
            logger.warning("Circuit breaker OPEN: %s", model)

    def record_success(self, model: str) -> None:
        self._errors.pop(model, None)
        self._open.pop(model, None)


class Router:
    """Main RouteIQ router with multi-provider support."""

    def __init__(
        self,
        config_path: str = "conf/router.yaml",
        state_path: str = "state/state.json",
        log_path: str = "state/model-stats.jsonl",
    ) -> None:
        # Auto-discover credentials from OpenClaw, Claude Code, NadirClaw
        from .credentials import apply_discovered_credentials
        self._discovered_creds = apply_discovered_credentials()

        self._policy = TaskPolicy(config_path)
        self._budget = BudgetManager(config_path, state_path)
        self._alerts = AlertManager()

        policy_cfg = self._policy.raw_config.get("policy", {})
        self._cb = CircuitBreaker(
            max_errors=policy_cfg.get("circuit_breaker_errors", 5),
            window_s=policy_cfg.get("circuit_breaker_window_s", 60),
        )
        self._retry_429 = policy_cfg.get("retry_on_429", True)
        self._retry_429_max = policy_cfg.get("retry_429_max", 3)
        self._retry_429_backoff = policy_cfg.get("retry_429_backoff_s", 1.0)

        cache_cfg = self._policy.raw_config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        self._cache = ResponseCache(
            max_size=cache_cfg.get("max_size", 256),
            ttl_seconds=cache_cfg.get("ttl_seconds", 3600),
        ) if cache_enabled else None

        self._sessions = SessionManager()
        self._log_path = log_path
        self._latency_stats: dict[str, float] = {}

    # ── Main routing ──

    def route(self, req: RouterRequest) -> RouterResponse:
        """Main entry point. Full classification + routing pipeline."""

        # 0. Hot-reload config if file changed (no restart needed)
        self._policy.reload_if_changed()

        # 1. Budget check
        budget_mode = self._budget.budget_mode()
        if budget_mode == "stopped":
            raise RuntimeError("Budget exhausted. Top up your balance or raise limits.")

        # 2. Explicit model requested? (resolve alias)
        if req.model and req.model != "auto":
            resolved = self._policy.resolve_alias(req.model)
            model_cfg = self._policy.get_model_config(resolved)
            if model_cfg:
                resp = self._call_single_model(req, resolved, "explicit")
                if resp:
                    return resp
                # Explicit model failed — fall through to normal routing
                logger.warning("Explicit model %s failed, falling back to auto-routing", resolved)

        # 3. Classify with modifiers (agentic + reasoning detection)
        classification = classify_with_modifiers(
            req.messages, req.hint, req.tools, req.tool_choice,
        )
        task_type = req.task_type or classification["task_type"]
        is_agentic = classification["is_agentic"]
        is_reasoning = classification["is_reasoning"]

        if is_agentic:
            logger.info("🤖 Agentic request detected — forcing complex model")
        if is_reasoning:
            logger.info("🧠 Reasoning request detected — forcing reasoning model")

        # 4. Session persistence — reuse pinned model
        if req.session_id:
            session = self._sessions.get(req.session_id)
            if session:
                resp = self._call_single_model(req, session.model_name, session.task_type)
                if resp:
                    session.touch()
                    resp.is_agentic = is_agentic
                    resp.is_reasoning = is_reasoning
                    return resp

        # 5. Apply routing profile (or mode for backward compat)
        profile = self._resolve_profile(req, classification)

        # Profile can override task type
        if profile.get("override_task"):
            task_type = profile["override_task"]

        # 6. Build model chain
        chain = self._build_chain(task_type, profile, budget_mode, is_agentic, is_reasoning)

        # 7. Get request params
        mode_cfg = MODE_SETTINGS.get(req.mode, MODE_SETTINGS["medium"])
        max_tokens = req.max_tokens or mode_cfg["max_tokens"]
        temperature = req.temperature if req.temperature is not None else mode_cfg["temperature"]

        # 8. Context-window filtering
        chain = self._policy.filter_by_context(chain, req.messages)
        if not chain:
            raise RuntimeError(f"No model has enough context window for this input (task={task_type})")

        # 9. Sort by score
        chain = self._policy.sort_chain_by_score(chain, task_type, self._latency_stats)

        # 10. Check cache (skip for streaming and tool use)
        if self._cache and not req.stream and not req.tools:
            cached = self._cache.get(req.messages)
            if cached:
                return RouterResponse(
                    content=cached["content"], model_used=cached["model"],
                    provider=cached.get("provider", "cache"),
                    tokens_in=0, tokens_out=0, cost_usd=0.0, latency_ms=0.0,
                    task_type=task_type, cached=True,
                    is_agentic=is_agentic, is_reasoning=is_reasoning,
                )

        # 11. Try chain
        resp = self._try_chain(req, task_type, chain, max_tokens, temperature)
        if resp:
            resp.is_agentic = is_agentic
            resp.is_reasoning = is_reasoning

            if req.session_id:
                self._sessions.pin(req.session_id, resp.model_used, task_type)
            if self._cache and not req.stream and not req.tools:
                self._cache.put(req.messages, {
                    "content": resp.content, "model": resp.model_used,
                    "provider": resp.provider,
                })
            return resp

        raise RuntimeError(f"All models failed for task_type={task_type}")

    def _resolve_profile(self, req: RouterRequest, classification: dict) -> dict:
        """Merge routing profile + mode settings."""
        profile = {}

        # Profile takes priority
        if req.profile and req.profile in ROUTING_PROFILES:
            profile = dict(ROUTING_PROFILES[req.profile])
        elif classification["force_complex"]:
            # Auto-escalate for agentic/reasoning
            profile = dict(ROUTING_PROFILES["premium"])

        return profile

    def _build_chain(
        self, task_type: str, profile: dict,
        budget_mode: str, is_agentic: bool, is_reasoning: bool,
    ) -> list[str]:
        """Build and filter the model chain based on task, profile, and budget."""
        chain = self._policy.get_chain(task_type)

        # Profile modifiers
        if profile.get("only_free") or profile.get("force_free"):
            chain = [m for m in chain if self._policy.is_free(m)]

        if profile.get("chain_reverse"):
            chain = list(reversed(chain))

        limit = profile.get("chain_limit", 99)
        chain = chain[:limit]

        # Agentic/reasoning → force complex models (reverse chain = expensive first)
        if (is_agentic or is_reasoning) and not profile.get("chain_reverse"):
            chain = list(reversed(chain))

        # Budget-driven filtering
        if budget_mode == "critical":
            free_chain = [m for m in chain if self._policy.is_free(m)]
            chain = free_chain or chain[:1]

        return chain

    def _call_single_model(self, req: RouterRequest, model_name: str, task_type: str) -> Optional[RouterResponse]:
        """Try a single specific model. Used for explicit model requests and session pinning."""
        if self._cb.is_open(model_name):
            return None

        model_cfg = self._policy.get_model_config(model_name)
        if not model_cfg:
            return None

        mode_cfg = MODE_SETTINGS.get(req.mode, MODE_SETTINGS["medium"])
        max_tokens = req.max_tokens or mode_cfg["max_tokens"]
        temperature = req.temperature if req.temperature is not None else mode_cfg["temperature"]

        return self._call_model(req, model_name, model_cfg, task_type, max_tokens, temperature)

    def _try_chain(
        self, req: RouterRequest, task_type: str, chain: list[str],
        max_tokens: int, temperature: float,
    ) -> Optional[RouterResponse]:
        """Try each model in chain. Returns first success or None."""
        for model_name in chain:
            if self._cb.is_open(model_name):
                continue

            model_cfg = self._policy.get_model_config(model_name)
            resp = self._call_model(req, model_name, model_cfg, task_type, max_tokens, temperature)
            if resp:
                return resp

        return None

    def _call_model(
        self, req: RouterRequest, model_name: str, model_cfg: dict,
        task_type: str, max_tokens: int, temperature: float,
    ) -> Optional[RouterResponse]:
        """Call a specific model via its provider backend."""
        model_id = model_cfg.get("id", model_name)
        provider_name = self._policy.get_provider(model_name)
        backend = get_backend(provider_name)

        last_err = None
        attempts = 1 + (self._retry_429_max if self._retry_429 else 0)

        for attempt in range(attempts):
            try:
                t0 = time.time()
                result = backend.call(
                    model_id=model_id,
                    messages=req.messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=req.tools,
                )
                latency_ms = (time.time() - t0) * 1000

                content = result["content"]
                usage = result["usage"]
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)

                # Prefer real cost from provider, fall back to estimate
                real_cost = result.get("real_cost_usd")
                cost = real_cost if real_cost is not None else self._estimate_cost(model_cfg, tokens_in, tokens_out)

                # Track
                self._budget.track(model_name, tokens_in, tokens_out, cost)
                self._budget.check_and_alert(self._alerts.send)
                self._cb.record_success(model_name)
                self._latency_stats[model_name] = latency_ms

                append_jsonl(self._log_path, {
                    "ts": time.time(), "task_type": task_type,
                    "model": model_name, "provider": provider_name,
                    "tokens_in": tokens_in, "tokens_out": tokens_out,
                    "cost_usd": cost, "latency_ms": round(latency_ms, 1),
                    "real_cost": real_cost is not None,
                })

                return RouterResponse(
                    content=content, model_used=model_name, provider=provider_name,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cost_usd=cost, latency_ms=latency_ms, task_type=task_type,
                    tool_calls=result.get("tool_calls"),
                )

            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str and attempt < attempts - 1:
                    wait = self._retry_429_backoff * (2 ** attempt)
                    logger.info("429 on %s, retry in %.1fs (%d/%d)", model_name, wait, attempt + 1, attempts)
                    time.sleep(wait)
                    continue
                logger.warning("Model %s failed: %s", model_name, e)
                self._cb.record_failure(model_name)
                return None

        return None

    # ── Streaming ──

    def route_stream(self, req: RouterRequest) -> Generator[str, None, None]:
        """Stream response via SSE. Yields content chunks."""
        budget_mode = self._budget.budget_mode()
        if budget_mode == "stopped":
            raise RuntimeError("Budget exhausted.")

        classification = classify_with_modifiers(req.messages, req.hint, req.tools, req.tool_choice)
        task_type = req.task_type or classification["task_type"]
        is_agentic = classification["is_agentic"]
        is_reasoning = classification["is_reasoning"]
        profile = self._resolve_profile(req, classification)

        if profile.get("override_task"):
            task_type = profile["override_task"]

        mode_cfg = MODE_SETTINGS.get(req.mode, MODE_SETTINGS["medium"])
        max_tokens = req.max_tokens or mode_cfg["max_tokens"]
        temperature = req.temperature if req.temperature is not None else mode_cfg["temperature"]

        chain = self._build_chain(task_type, profile, budget_mode, is_agentic, is_reasoning)
        chain = self._policy.filter_by_context(chain, req.messages)
        if not chain:
            raise RuntimeError("No model has enough context window")

        for model_name in chain:
            if self._cb.is_open(model_name):
                continue

            model_cfg = self._policy.get_model_config(model_name)
            model_id = model_cfg.get("id", model_name)
            provider_name = self._policy.get_provider(model_name)
            backend = get_backend(provider_name)

            try:
                t0 = time.time()
                full_content = ""

                for chunk in backend.call_stream(model_id, req.messages, max_tokens, temperature, tools=req.tools):
                    full_content += chunk
                    yield chunk

                latency_ms = (time.time() - t0) * 1000
                user_text = " ".join(
                    m.get("content", "") for m in req.messages if isinstance(m.get("content"), str)
                )
                est_tokens_in = self._policy.estimate_tokens(user_text)
                est_tokens_out = self._policy.estimate_tokens(full_content)
                cost = self._estimate_cost(model_cfg, est_tokens_in, est_tokens_out)

                self._budget.track(model_name, est_tokens_in, est_tokens_out, cost)
                self._budget.check_and_alert(self._alerts.send)
                self._cb.record_success(model_name)

                append_jsonl(self._log_path, {
                    "ts": time.time(), "task_type": task_type,
                    "model": model_name, "provider": provider_name,
                    "tokens_in": est_tokens_in, "tokens_out": est_tokens_out,
                    "cost_usd": cost, "latency_ms": round(latency_ms, 1),
                    "stream": True,
                })
                return

            except Exception as e:
                logger.warning("Stream %s failed: %s", model_name, e)
                self._cb.record_failure(model_name)
                continue

        raise RuntimeError(f"All models failed for streaming task_type={task_type}")

    # ── Helpers ──

    @staticmethod
    def _estimate_cost(model_cfg: dict, tokens_in: int, tokens_out: int) -> float:
        cost_in = model_cfg.get("cost_per_1k_input", 0.0) * tokens_in / 1000
        cost_out = model_cfg.get("cost_per_1k_output", 0.0) * tokens_out / 1000
        return cost_in + cost_out

    # ── Status ──

    def budget_status(self) -> dict:
        return self._budget.get_status()

    def cache_stats(self) -> dict:
        return self._cache.stats() if self._cache else {"enabled": False}

    def session_stats(self) -> dict:
        return self._sessions.stats()

    def full_status(self) -> dict:
        from .credentials import get_discovery_status
        return {
            "budget": self.budget_status(),
            "cache": self.cache_stats(),
            "sessions": self.session_stats(),
            "credentials": get_discovery_status(),
        }
