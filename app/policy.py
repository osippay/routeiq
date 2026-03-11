"""
RouteIQ — Routing policy and model configuration.

Loads YAML config, provides model chains, scoring, context-window checks.
Classification logic is in classifier.py (separated for clarity).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 3.5


class TaskPolicy:
    """Loads routing config from YAML. Provides chains, scoring, context checks."""

    def __init__(self, config_path: str = "conf/router.yaml") -> None:
        self._path = Path(config_path)
        self._cfg: dict[str, Any] = {}
        self._mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("TaskPolicy: config %s not found, using defaults", self._path)
            self._cfg = self._defaults()
            return
        try:
            self._mtime = self._path.stat().st_mtime
            with self._path.open("r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.exception("Failed to load policy config %s: %s", self._path, e)
            self._cfg = self._defaults()

        defaults = self._defaults()
        for k, v in defaults.items():
            self._cfg.setdefault(k, v)

        # COMPAT: if task_chains nested under 'policy', hoist to top level
        if "task_chains" not in self._cfg and "policy" in self._cfg:
            nested = self._cfg["policy"]
            if isinstance(nested, dict) and "task_chains" in nested:
                self._cfg["task_chains"] = nested["task_chains"]

    def reload_if_changed(self) -> bool:
        """Check if config file was modified and reload. Returns True if reloaded."""
        if not self._path.exists():
            return False
        try:
            current_mtime = self._path.stat().st_mtime
            if current_mtime > self._mtime:
                logger.info("Config changed, reloading %s", self._path)
                self._load()
                return True
        except OSError:
            pass
        return False

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "task_chains": {
                "text": ["gpt4o_mini", "gemini_flash", "sonnet"],
                "code": ["qwen_coder", "sonnet", "opus"],
                "image": ["dall_e_3"],
                "audio": ["whisper"],
                "vision": ["sonnet", "gpt4o_mini"],
                "think": ["opus", "sonnet"],
                "strategy": ["opus"],
                "summarize": ["gemini_flash", "gpt4o_mini", "sonnet"],
            },
            "models": {
                "gpt4o_mini": {"id": "openai/gpt-4o-mini", "priority": 70, "free": False,
                               "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006,
                               "context_length": 128000, "capabilities": ["text", "vision"]},
                "sonnet": {"id": "anthropic/claude-sonnet-4-5", "priority": 85, "free": False,
                           "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015,
                           "context_length": 200000, "capabilities": ["text", "code", "vision", "think"]},
                "opus": {"id": "anthropic/claude-opus-4-6", "priority": 100, "free": False,
                         "cost_per_1k_input": 0.015, "cost_per_1k_output": 0.075,
                         "context_length": 200000, "capabilities": ["text", "code", "vision", "think", "strategy"]},
                "qwen_coder": {"id": "qwen/qwen-2.5-coder-32b-instruct", "priority": 90, "free": True,
                               "cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0,
                               "context_length": 32768, "capabilities": ["code"]},
                "gemini_flash": {"id": "google/gemini-2.5-flash-lite", "priority": 65, "free": False,
                                 "cost_per_1k_input": 0.000075, "cost_per_1k_output": 0.0003,
                                 "context_length": 1048576, "capabilities": ["text", "vision", "summarize"]},
                "dall_e_3": {"id": "openai/dall-e-3", "priority": 100, "free": False,
                             "cost_per_image": 0.04, "context_length": 4096, "capabilities": ["image"]},
                "whisper": {"id": "openai/whisper-1", "priority": 100, "free": False,
                            "cost_per_minute": 0.006, "context_length": 0, "capabilities": ["audio"]},
            },
            "weights": {"cost_weight": 0.5, "quality_weight": 0.3, "latency_weight": 0.2},
        }

    # ── Chain helpers ──

    def get_chain(self, task_type: str) -> list[str]:
        chains = self._cfg.get("task_chains", {})
        if task_type in chains and isinstance(chains[task_type], list):
            return [str(x) for x in chains[task_type]]
        fallback = chains.get("text") or self._defaults()["task_chains"]["text"]
        return [str(x) for x in fallback]

    # ── Model config ──

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        models = self._cfg.get("models", {})
        raw = models.get(model_name)
        if raw is None:
            return {}
        return dict(raw) if isinstance(raw, dict) else {"value": raw}

    def is_free(self, model_name: str) -> bool:
        return bool(self.get_model_config(model_name).get("free"))

    def get_capabilities(self, model_name: str) -> list[str]:
        cfg = self.get_model_config(model_name)
        caps = cfg.get("capabilities") or []
        if isinstance(caps, list):
            return [str(c) for c in caps]
        return [str(caps)] if isinstance(caps, str) else []

    # ── Context-window awareness ──

    def get_context_length(self, model_name: str) -> int:
        return int(self.get_model_config(model_name).get("context_length", 0))

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))

    def fits_context(self, model_name: str, messages: list[dict]) -> bool:
        ctx = self.get_context_length(model_name)
        if ctx <= 0:
            return True
        total_text = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        estimated = self.estimate_tokens(total_text)
        return estimated < ctx * 0.8

    def filter_by_context(self, chain: list[str], messages: list[dict]) -> list[str]:
        return [m for m in chain if self.fits_context(m, messages)]

    # ── Scoring ──

    def score_model(
        self,
        model_name: str,
        task_type: str,
        est_cost_usd: float = 0.0,
        latency_ms: float = 500.0,
        success_rate: float = 1.0,
    ) -> float:
        weights = self._cfg.get("weights", {})
        w_cost = float(weights.get("cost_weight", 0.4))
        w_qual = float(weights.get("quality_weight", 0.5))
        w_lat = float(weights.get("latency_weight", 0.1))

        model_cfg = self.get_model_config(model_name)
        quality = max(0.0, min(1.0, float(model_cfg.get("priority", 0)) / 100.0))

        cost_c = w_cost * (1.0 / (1.0 + est_cost_usd * 1000.0))
        lat_c = w_lat * (1.0 / (1.0 + latency_ms / 1000.0))
        composite = cost_c + w_qual * quality + lat_c

        success_rate = max(0.0, min(1.0, success_rate))
        composite *= 0.7 + 0.3 * success_rate
        return float(composite)

    def sort_chain_by_score(
        self,
        chain: list[str],
        task_type: str,
        latency_stats: dict[str, float] | None = None,
    ) -> list[str]:
        latency_stats = latency_stats or {}

        def _key(m: str) -> float:
            cfg = self.get_model_config(m)
            est = cfg.get("cost_per_1k_input", 0) + cfg.get("cost_per_1k_output", 0)
            return -self.score_model(m, task_type, est, latency_stats.get(m, 500.0))

        return sorted(chain, key=_key)

    # ── Model aliases ──

    DEFAULT_ALIASES = {
        "sonnet": "sonnet", "claude": "sonnet", "claude-sonnet": "sonnet",
        "opus": "opus", "claude-opus": "opus",
        "gpt4": "gpt4o_mini", "gpt": "gpt4o_mini", "gpt4o": "gpt4o_mini",
        "gemini": "gemini_flash", "flash": "gemini_flash", "gemini-flash": "gemini_flash",
        "qwen": "qwen_coder", "coder": "qwen_coder",
        "llama": "llama_groq", "groq": "llama_groq",
        "dalle": "dall_e_3", "dall-e": "dall_e_3",
    }

    def resolve_alias(self, name: str) -> str:
        """Resolve a model alias to its config name."""
        user_aliases = self._cfg.get("aliases", {})
        if name in user_aliases:
            return str(user_aliases[name])
        if name.lower() in self.DEFAULT_ALIASES:
            return self.DEFAULT_ALIASES[name.lower()]
        if name in self._cfg.get("models", {}):
            return name
        return name

    # ── Provider detection ──

    def get_provider(self, model_name: str) -> str:
        """Get the provider backend for a model. Defaults to 'openrouter'."""
        cfg = self.get_model_config(model_name)
        if cfg.get("provider"):
            return str(cfg["provider"])
        model_id = cfg.get("id", "")
        if model_id.startswith("ollama/"):
            return "ollama"
        return "openrouter"

    @property
    def raw_config(self) -> dict[str, Any]:
        return dict(self._cfg)


# ── Routing profiles ──

ROUTING_PROFILES = {
    "auto": {
        "description": "🔄 Auto — classifier picks the best model",
        "force_free": False, "force_complex": False,
    },
    "eco": {
        "description": "💚 Economy — cheapest models first",
        "force_complex": False, "chain_limit": 2,
    },
    "premium": {
        "description": "💎 Premium — best quality, ignore cost",
        "force_complex": True, "chain_reverse": True, "chain_limit": 1,
    },
    "free": {
        "description": "🆓 Free — only free models",
        "only_free": True, "chain_limit": 99,
    },
    "reasoning": {
        "description": "🧠 Reasoning — chain-of-thought optimized",
        "force_complex": True, "override_task": "think", "chain_limit": 1,
    },
}
