"""
RouteIQ — Multi-provider LLM backend.

Instead of routing everything through OpenRouter, we can call providers directly:
- OpenRouter (default, 100+ models)
- Anthropic (Claude models — direct, lower latency)
- OpenAI (GPT models — direct)
- Google GenAI (Gemini models — direct, free tier)
- Ollama (local models — free, private, offline)

Each backend implements the same interface: call() and call_stream().
The router picks the backend based on model config in YAML.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generator

import requests

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Base class for LLM provider backends."""

    name: str = "base"

    @abstractmethod
    def call(
        self,
        model_id: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list | None = None,
        **kwargs,
    ) -> dict:
        """
        Call the LLM and return a normalized response dict:
        {
            "content": str,
            "usage": {"prompt_tokens": int, "completion_tokens": int},
            "raw": <provider-specific response>,
        }
        """
        ...

    @abstractmethod
    def call_stream(
        self,
        model_id: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Yield content chunks via SSE streaming."""
        ...


# ── OpenRouter backend ────────────────────────────────────────────

class OpenRouterBackend(LLMBackend):
    """Default backend — routes through OpenRouter API."""

    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        pass  # Key read fresh from env in _headers()

    def _headers(self) -> dict:
        key = os.getenv("OPENROUTER_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/routeiq",
            "X-Title": "RouteIQ",
        }

    def call(self, model_id, messages, max_tokens=1024, temperature=0.7, tools=None, **kw) -> dict:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},  # Get real cost from OpenRouter
        }
        if tools:
            payload["tools"] = tools
        r = requests.post(self.URL, headers=self._headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content = message.get("content", "")
        usage = data.get("usage", {})

        # Parse real cost from OpenRouter (if available)
        real_cost = None
        # OpenRouter returns cost in usage.cost or in x-openrouter-cost header
        if "cost" in usage:
            real_cost = float(usage["cost"])

        return {
            "content": content,
            "tool_calls": message.get("tool_calls"),  # Pass through tool calls
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "real_cost_usd": real_cost,
            "raw": data,
        }

    def call_stream(self, model_id, messages, max_tokens=1024, temperature=0.7, **kw):
        payload = {
            "model": model_id, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True,
        }
        if kw.get("tools"):
            payload["tools"] = kw["tools"]
        with requests.post(self.URL, headers=self._headers(), json=payload, timeout=60, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace")
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


# ── Anthropic backend (direct) ────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """Direct Anthropic API — lower latency for Claude models."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self) -> None:
        pass

    def _headers(self) -> dict:
        return {
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split system message from conversation (Anthropic API format)."""
        system = None
        converted = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                converted.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        return system, converted

    def call(self, model_id, messages, max_tokens=1024, temperature=0.7, tools=None, **kw) -> dict:
        # Strip provider prefix: anthropic/claude-sonnet-4-5 → claude-sonnet-4-5
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        system, msgs = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": model, "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        r = requests.post(self.URL, headers=self._headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Anthropic returns content as array of blocks
        content_blocks = data.get("content", [])
        content = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
            "raw": data,
        }

    def call_stream(self, model_id, messages, max_tokens=1024, temperature=0.7, **kw):
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        system, msgs = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": model, "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        with requests.post(self.URL, headers=self._headers(), json=payload, timeout=60, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace")
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:].strip()
                try:
                    event = json.loads(data_str)
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            yield text
                    elif event.get("type") == "message_stop":
                        break
                except json.JSONDecodeError:
                    continue


# ── OpenAI backend (direct) ───────────────────────────────────────

class OpenAIBackend(LLMBackend):
    """Direct OpenAI API."""

    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self) -> None:
        pass

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }

    def call(self, model_id, messages, max_tokens=1024, temperature=0.7, tools=None, **kw) -> dict:
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        payload: dict[str, Any] = {
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        r = requests.post(self.URL, headers=self._headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or [{}]
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")
        usage = data.get("usage", {})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "raw": data,
        }

    def call_stream(self, model_id, messages, max_tokens=1024, temperature=0.7, **kw):
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        payload = {
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True,
        }
        with requests.post(self.URL, headers=self._headers(), json=payload, timeout=60, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace")
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


# ── Google GenAI backend (direct) ─────────────────────────────────

class GoogleBackend(LLMBackend):
    """Direct Google Generative AI API (Gemini models)."""

    name = "google"
    URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    URL_STREAM_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

    def __init__(self) -> None:
        pass

    def _get_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI format to Gemini format."""
        system = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system = text
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return system, contents

    def call(self, model_id, messages, max_tokens=1024, temperature=0.7, tools=None, **kw) -> dict:
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        system, contents = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = self.URL_TEMPLATE.format(model=model)
        r = requests.post(f"{url}?key={self._get_key()}", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates", [{}])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        usage_meta = data.get("usageMetadata", {})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            },
            "raw": data,
        }

    def call_stream(self, model_id, messages, max_tokens=1024, temperature=0.7, **kw):
        model = model_id.split("/")[-1] if "/" in model_id else model_id
        system, contents = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = self.URL_STREAM_TEMPLATE.format(model=model)
        r = requests.post(f"{url}?alt=sse&key={self._get_key()}", json=payload, timeout=60, stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="replace")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:].strip()
            try:
                chunk = json.loads(data_str)
                candidates = chunk.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        text = p.get("text", "")
                        if text:
                            yield text
            except json.JSONDecodeError:
                continue


# ── Ollama backend (local) ────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """Local Ollama server — free, private, offline."""

    name = "ollama"

    def __init__(self) -> None:
        pass

    @property
    def _base(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def call(self, model_id, messages, max_tokens=1024, temperature=0.7, tools=None, **kw) -> dict:
        # ollama/llama3.1:8b → llama3.1:8b
        model = model_id.replace("ollama/", "")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        r = requests.post(f"{self._base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        content = data.get("message", {}).get("content", "")
        return {
            "content": content,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            "raw": data,
        }

    def call_stream(self, model_id, messages, max_tokens=1024, temperature=0.7, **kw):
        model = model_id.replace("ollama/", "")
        payload = {
            "model": model, "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with requests.post(f"{self._base}/api/chat", json=payload, timeout=120, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ── Backend registry ──────────────────────────────────────────────

_BACKENDS: dict[str, LLMBackend] = {}


def get_backend(provider: str) -> LLMBackend:
    """Get or create a backend instance for the given provider."""
    provider = provider.lower().strip()

    if provider not in _BACKENDS:
        cls_map = {
            "openrouter": OpenRouterBackend,
            "anthropic": AnthropicBackend,
            "openai": OpenAIBackend,
            "google": GoogleBackend,
            "gemini": GoogleBackend,
            "ollama": OllamaBackend,
        }
        cls = cls_map.get(provider)
        if cls is None:
            logger.warning("Unknown provider '%s', falling back to openrouter", provider)
            cls = OpenRouterBackend
        _BACKENDS[provider] = cls()

    return _BACKENDS[provider]


def detect_provider(model_id: str) -> str:
    """Auto-detect provider from model ID string."""
    model_lower = model_id.lower()

    if model_lower.startswith("ollama/"):
        return "ollama"

    # Check for direct provider prefixes
    prefix_map = {
        "anthropic/": "anthropic",
        "openai/": "openai",
        "google/": "google",
        "gemini/": "google",
    }
    for prefix, provider in prefix_map.items():
        if model_lower.startswith(prefix):
            # Only use direct backend if the API key is available
            env_keys = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            env_key = env_keys.get(provider)
            if env_key and os.getenv(env_key):
                return provider
            # No direct key → fall back to OpenRouter
            return "openrouter"

    return "openrouter"
