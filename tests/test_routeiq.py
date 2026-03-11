"""
RouteIQ — Test suite.

Covers all 13 modules without requiring API keys or network access.
Run: pytest tests/ -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════

class TestClassifier:
    """Tests for app.classifier — 8-type classification + modifiers."""

    def test_code_detection_en(self):
        from app.classifier import classify_task
        assert classify_task("write a function to parse JSON") == "code"
        assert classify_task("fix the bug in line 42") == "code"
        assert classify_task("refactor this class") == "code"
        assert classify_task("write a hello world function") == "code"

    def test_code_detection_ru(self):
        from app.classifier import classify_task
        assert classify_task("напиши код для сортировки") == "code"
        assert classify_task("напиши функцию на python") == "code"

    def test_image_detection(self):
        from app.classifier import classify_task
        assert classify_task("generate image of a sunset") == "image"
        assert classify_task("сгенерируй картинку кота") == "image"
        assert classify_task("draw me a logo") == "image"

    def test_summarize_detection(self):
        from app.classifier import classify_task
        assert classify_task("summarize this document") == "summarize"
        assert classify_task("TL;DR of this article") == "summarize"
        assert classify_task("суммаризируй текст") == "summarize"

    def test_vision_detection(self):
        from app.classifier import classify_task
        assert classify_task("что на этой картинке?") == "vision"
        assert classify_task("describe this photo") == "vision"

    def test_think_detection(self):
        from app.classifier import classify_task
        assert classify_task("complex reasoning about this problem") == "think"
        assert classify_task("подумай над этой задачей") == "think"

    def test_strategy_detection(self):
        from app.classifier import classify_task
        assert classify_task("write a business plan") == "strategy"
        assert classify_task("составь бизнес-план") == "strategy"

    def test_audio_detection(self):
        from app.classifier import classify_task
        assert classify_task("transcribe this audio file") == "audio"
        assert classify_task("транскрибируй аудио") == "audio"

    def test_default_to_text(self):
        from app.classifier import classify_task
        assert classify_task("hello how are you?") == "text"
        assert classify_task("explain photosynthesis") == "text"
        assert classify_task("") == "text"

    def test_hint_overrides(self):
        from app.classifier import classify_task
        assert classify_task("anything", hint="code") == "code"
        assert classify_task("anything", hint="image") == "image"
        assert classify_task("anything", hint="THINK") == "think"

    def test_confidence_returned(self):
        from app.classifier import classify_task_with_confidence
        task, conf = classify_task_with_confidence("write a python function")
        assert task == "code"
        assert conf > 0

    def test_confidence_with_hint(self):
        from app.classifier import classify_task_with_confidence
        task, conf = classify_task_with_confidence("anything", hint="code")
        assert task == "code"
        assert conf == 1.0

    def test_confidence_empty(self):
        from app.classifier import classify_task_with_confidence
        task, conf = classify_task_with_confidence("")
        assert task == "text"
        assert conf == 0.0


# ═══════════════════════════════════════════════════════════════════
# Agentic & Reasoning Detection
# ═══════════════════════════════════════════════════════════════════

class TestAgenticDetection:
    """Tests for detect_agentic — tool use and agent workflow detection."""

    def test_tools_in_request(self):
        from app.classifier import detect_agentic
        assert detect_agentic([], tools=[{"type": "function", "function": {"name": "get_weather"}}])

    def test_tool_choice_explicit(self):
        from app.classifier import detect_agentic
        assert detect_agentic([], tool_choice="required")
        assert not detect_agentic([], tool_choice="none")
        assert not detect_agentic([], tool_choice="auto")

    def test_tool_role_message(self):
        from app.classifier import detect_agentic
        assert detect_agentic([{"role": "tool", "content": "42", "tool_call_id": "call_1"}])

    def test_agentic_system_prompt(self):
        from app.classifier import detect_agentic
        assert detect_agentic([{"role": "system", "content": "You have access to tools: search, calculator"}])
        assert detect_agentic([{"role": "system", "content": "you are an agent that can execute commands"}])

    def test_assistant_tool_calls(self):
        from app.classifier import detect_agentic
        assert detect_agentic([{"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]}])

    def test_assistant_function_call(self):
        from app.classifier import detect_agentic
        assert detect_agentic([{"role": "assistant", "function_call": {"name": "search"}}])

    def test_normal_not_agentic(self):
        from app.classifier import detect_agentic
        assert not detect_agentic([{"role": "user", "content": "hello"}])
        assert not detect_agentic([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "what's the weather?"},
        ])

    def test_empty_not_agentic(self):
        from app.classifier import detect_agentic
        assert not detect_agentic([])
        assert not detect_agentic([], tools=None, tool_choice=None)


class TestReasoningDetection:
    """Tests for detect_reasoning — chain-of-thought markers."""

    def test_two_markers_detected(self):
        from app.classifier import detect_reasoning
        assert detect_reasoning([{"role": "user", "content": "think carefully and analyze this step by step"}])

    def test_single_marker_not_enough(self):
        from app.classifier import detect_reasoning
        assert not detect_reasoning([{"role": "user", "content": "think about cats"}])

    def test_normal_not_reasoning(self):
        from app.classifier import detect_reasoning
        assert not detect_reasoning([{"role": "user", "content": "hello world"}])

    def test_russian_markers(self):
        from app.classifier import detect_reasoning
        assert detect_reasoning([{"role": "user", "content": "давай подумаем пошагово над этой задачей"}])

    def test_empty_messages(self):
        from app.classifier import detect_reasoning
        assert not detect_reasoning([])


class TestClassifyWithModifiers:
    """Tests for the full classification pipeline."""

    def test_agentic_forces_complex(self):
        from app.classifier import classify_with_modifiers
        result = classify_with_modifiers(
            [{"role": "user", "content": "find weather"}],
            tools=[{"type": "function"}],
        )
        assert result["is_agentic"] is True
        assert result["force_complex"] is True

    def test_reasoning_forces_complex(self):
        from app.classifier import classify_with_modifiers
        result = classify_with_modifiers(
            [{"role": "user", "content": "think carefully step by step and analyze the implications"}],
        )
        assert result["is_reasoning"] is True
        assert result["force_complex"] is True

    def test_normal_request(self):
        from app.classifier import classify_with_modifiers
        result = classify_with_modifiers(
            [{"role": "user", "content": "write a sort function"}],
        )
        assert result["task_type"] == "code"
        assert result["is_agentic"] is False
        assert result["is_reasoning"] is False
        assert result["force_complex"] is False


# ═══════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════

class TestCache:
    """Tests for app.cache — LRU response cache."""

    def test_put_and_get(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put(msgs, {"content": "world", "model": "test"})
        result = cache.get(msgs)
        assert result is not None
        assert result["content"] == "world"

    def test_miss(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=10)
        assert cache.get([{"role": "user", "content": "unknown"}]) is None

    def test_ttl_expiry(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=10, ttl_seconds=0.01)
        msgs = [{"role": "user", "content": "test"}]
        cache.put(msgs, {"content": "ok"})
        time.sleep(0.02)
        assert cache.get(msgs) is None

    def test_lru_eviction(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=2, ttl_seconds=60)
        cache.put([{"c": "a"}], {"content": "1"})
        cache.put([{"c": "b"}], {"content": "2"})
        cache.put([{"c": "c"}], {"content": "3"})
        assert cache.get([{"c": "a"}]) is None
        assert cache.get([{"c": "b"}]) is not None
        assert cache.get([{"c": "c"}]) is not None

    def test_stats(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=10)
        msgs = [{"role": "user", "content": "test"}]
        cache.put(msgs, {"content": "ok"})
        cache.get(msgs)  # hit
        cache.get([{"content": "miss"}])  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] > 0

    def test_clear(self):
        from app.cache import ResponseCache
        cache = ResponseCache(max_size=10)
        cache.put([{"c": "a"}], {"content": "1"})
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get([{"c": "a"}]) is None


# ═══════════════════════════════════════════════════════════════════
# Session
# ═══════════════════════════════════════════════════════════════════

class TestSession:
    """Tests for app.session — model pinning per conversation."""

    def test_pin_and_get(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=60)
        sm.pin("sess1", "sonnet", "code")
        session = sm.get("sess1")
        assert session is not None
        assert session.model_name == "sonnet"
        assert session.task_type == "code"

    def test_expiry(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=0.01)
        sm.pin("sess1", "sonnet", "code")
        time.sleep(0.02)
        assert sm.get("sess1") is None

    def test_touch_resets_timer(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=0.05)
        sm.pin("s", "m", "t")
        time.sleep(0.03)
        s = sm.get("s")
        assert s is not None
        s.touch()
        time.sleep(0.03)
        # Should still be alive because we touched it
        assert sm.get("s") is not None

    def test_cleanup(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=0.01)
        sm.pin("a", "m1", "t1")
        sm.pin("b", "m2", "t2")
        time.sleep(0.02)
        removed = sm.cleanup()
        assert removed == 2
        assert sm.stats()["active_sessions"] == 0

    def test_remove(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=60)
        sm.pin("s", "m", "t")
        sm.remove("s")
        assert sm.get("s") is None

    def test_stats(self):
        from app.session import SessionManager
        sm = SessionManager(ttl_seconds=60)
        sm.pin("a", "m1", "t1")
        sm.pin("b", "m2", "t2")
        stats = sm.stats()
        assert stats["active_sessions"] == 2


# ═══════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════

class TestPolicy:
    """Tests for app.policy — config, chains, scoring, aliases, profiles."""

    @pytest.fixture
    def config_path(self, tmp_path):
        cfg = {
            "task_chains": {"text": ["m_a", "m_b"], "code": ["m_c"]},
            "models": {
                "m_a": {"id": "test/a", "priority": 70, "free": False,
                        "cost_per_1k_input": 0.001, "cost_per_1k_output": 0.002,
                        "context_length": 4096, "capabilities": ["text"]},
                "m_b": {"id": "test/b", "priority": 90, "free": True,
                        "cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0,
                        "context_length": 128000, "capabilities": ["text", "code"]},
                "m_c": {"id": "test/c", "priority": 95, "free": False,
                        "cost_per_1k_input": 0.01, "cost_per_1k_output": 0.03,
                        "context_length": 32000, "capabilities": ["code"]},
            },
            "weights": {"cost_weight": 0.5, "quality_weight": 0.3, "latency_weight": 0.2},
            "aliases": {"mymodel": "m_a"},
        }
        p = tmp_path / "test_config.yaml"
        p.write_text(yaml.dump(cfg))
        return str(p)

    def test_get_chain(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        assert p.get_chain("text") == ["m_a", "m_b"]
        assert p.get_chain("code") == ["m_c"]

    def test_unknown_chain_fallback(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        assert p.get_chain("unknown") == ["m_a", "m_b"]

    def test_is_free(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        assert p.is_free("m_b") is True
        assert p.is_free("m_a") is False

    def test_context_filtering_keeps_large(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        short = [{"content": "hello"}]
        chain = p.filter_by_context(["m_a", "m_b"], short)
        assert "m_a" in chain and "m_b" in chain

    def test_context_filtering_removes_small(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        long_msg = [{"content": "x" * 50000}]
        chain = p.filter_by_context(["m_a", "m_b"], long_msg)
        assert "m_a" not in chain
        assert "m_b" in chain

    def test_score_free_higher(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        s_free = p.score_model("m_b", "text", est_cost_usd=0.0)
        s_paid = p.score_model("m_a", "text", est_cost_usd=0.001)
        assert s_free > s_paid

    def test_sort_chain_by_score(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        sorted_chain = p.sort_chain_by_score(["m_a", "m_b"], "text")
        assert sorted_chain[0] == "m_b"

    def test_resolve_alias_default(self):
        from app.policy import TaskPolicy
        p = TaskPolicy()
        assert p.resolve_alias("flash") == "gemini_flash"
        assert p.resolve_alias("gpt4") == "gpt4o_mini"
        assert p.resolve_alias("sonnet") == "sonnet"

    def test_resolve_alias_user_defined(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        assert p.resolve_alias("mymodel") == "m_a"

    def test_resolve_alias_passthrough(self, config_path):
        from app.policy import TaskPolicy
        p = TaskPolicy(config_path)
        assert p.resolve_alias("nonexistent") == "nonexistent"

    def test_get_provider_default(self):
        from app.policy import TaskPolicy
        p = TaskPolicy()
        assert p.get_provider("gpt4o_mini") == "openrouter"

    def test_get_provider_ollama(self):
        from app.policy import TaskPolicy
        p = TaskPolicy()
        assert p.get_provider("llama_local") == "ollama"

    def test_routing_profiles_exist(self):
        from app.policy import ROUTING_PROFILES
        assert set(ROUTING_PROFILES.keys()) == {"auto", "eco", "premium", "free", "reasoning"}

    def test_routing_profile_eco(self):
        from app.policy import ROUTING_PROFILES
        assert ROUTING_PROFILES["eco"]["chain_limit"] == 2
        assert ROUTING_PROFILES["eco"]["force_complex"] is False

    def test_routing_profile_reasoning(self):
        from app.policy import ROUTING_PROFILES
        assert ROUTING_PROFILES["reasoning"]["override_task"] == "think"


# ═══════════════════════════════════════════════════════════════════
# Budget
# ═══════════════════════════════════════════════════════════════════

class TestBudget:
    """Tests for app.budget — spending, burn rate, modes."""

    @pytest.fixture
    def budget(self, tmp_path):
        cfg = {
            "budget": {
                "starting_balance_usd": 10.0,
                "warn_at_pct": 50, "critical_at_pct": 80, "stop_at_pct": 98,
                "daily_limit_usd": 0, "monthly_limit_usd": 0,
            }
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        state_path = tmp_path / "state.json"
        from app.budget import BudgetManager
        return BudgetManager(str(cfg_path), str(state_path))

    def test_initial_balance(self, budget):
        assert budget.balance_usd() == 10.0
        assert budget.spent_pct() == 0.0
        assert budget.budget_mode() == "normal"

    def test_tracking(self, budget):
        budget.track("test", 100, 50, 1.0)
        assert budget.balance_usd() == 9.0
        assert budget.spent_pct() == 10.0

    def test_economy_mode(self, budget):
        budget.track("m", 0, 0, 6.0)
        assert budget.budget_mode() == "economy"

    def test_critical_mode(self, budget):
        budget.track("m", 0, 0, 8.5)
        assert budget.budget_mode() == "critical"

    def test_stopped_mode(self, budget):
        budget.track("m", 0, 0, 9.9)
        assert budget.budget_mode() == "stopped"

    def test_status_dict(self, budget):
        status = budget.get_status()
        required_keys = ["balance_usd", "spent_pct", "mode", "burn_rate_usd_per_min",
                         "daily_spend_usd", "monthly_spend_usd"]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_daily_limit(self, tmp_path):
        cfg = {
            "budget": {
                "starting_balance_usd": 1000.0,
                "warn_at_pct": 50, "critical_at_pct": 80, "stop_at_pct": 98,
                "daily_limit_usd": 2.0, "monthly_limit_usd": 0,
            }
        }
        cfg_path = tmp_path / "cfg2.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        from app.budget import BudgetManager
        bm = BudgetManager(str(cfg_path), str(tmp_path / "state2.json"))
        bm.track("m", 0, 0, 2.5)
        assert bm.budget_mode() == "stopped"


# ═══════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════

class TestStorage:
    """Tests for app.storage — atomic file operations."""

    def test_atomic_read_missing(self, tmp_path):
        from app.storage import atomic_read
        assert atomic_read(str(tmp_path / "missing.json")) == {}

    def test_atomic_write_read(self, tmp_path):
        from app.storage import atomic_read, atomic_write
        p = str(tmp_path / "test.json")
        atomic_write(p, {"key": "value"})
        assert atomic_read(p) == {"key": "value"}

    def test_append_jsonl(self, tmp_path):
        from app.storage import append_jsonl
        p = str(tmp_path / "test.jsonl")
        append_jsonl(p, {"a": 1})
        append_jsonl(p, {"b": 2})
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["a"] == 1
        assert json.loads(lines[1])["b"] == 2


# ═══════════════════════════════════════════════════════════════════
# Backends
# ═══════════════════════════════════════════════════════════════════

class TestBackends:
    """Tests for app.backends — provider instances and detection."""

    def test_get_openrouter(self):
        from app.backends import get_backend, OpenRouterBackend
        assert isinstance(get_backend("openrouter"), OpenRouterBackend)

    def test_get_anthropic(self):
        from app.backends import get_backend, AnthropicBackend
        assert isinstance(get_backend("anthropic"), AnthropicBackend)

    def test_get_openai(self):
        from app.backends import get_backend, OpenAIBackend
        assert isinstance(get_backend("openai"), OpenAIBackend)

    def test_get_google(self):
        from app.backends import get_backend, GoogleBackend
        assert isinstance(get_backend("google"), GoogleBackend)

    def test_get_gemini_alias(self):
        from app.backends import get_backend, GoogleBackend
        assert isinstance(get_backend("gemini"), GoogleBackend)

    def test_get_ollama(self):
        from app.backends import get_backend, OllamaBackend
        assert isinstance(get_backend("ollama"), OllamaBackend)

    def test_unknown_falls_back(self):
        from app.backends import get_backend, OpenRouterBackend
        assert isinstance(get_backend("nonexistent"), OpenRouterBackend)

    def test_detect_ollama(self):
        from app.backends import detect_provider
        assert detect_provider("ollama/llama3.1:8b") == "ollama"

    def test_detect_default_openrouter(self):
        from app.backends import detect_provider
        assert detect_provider("some/random-model") == "openrouter"


# ═══════════════════════════════════════════════════════════════════
# Credentials
# ═══════════════════════════════════════════════════════════════════

class TestCredentials:
    """Tests for app.credentials — auto-discovery from multiple sources."""

    def test_normalize_provider(self):
        from app.credentials import _normalize_provider
        assert _normalize_provider("gemini") == "google"
        assert _normalize_provider("claude") == "anthropic"
        assert _normalize_provider("google-ai") == "google"
        assert _normalize_provider("openai") == "openai"
        assert _normalize_provider("random") == "random"

    def test_extract_api_key(self):
        from app.credentials import _extract_key_from_profile
        assert _extract_key_from_profile({"type": "api_key", "key": "sk-test"}) == "sk-test"

    def test_extract_valid_token(self):
        from app.credentials import _extract_key_from_profile
        profile = {"type": "token", "access": "tok-123", "expires": time.time() + 3600}
        assert _extract_key_from_profile(profile) == "tok-123"

    def test_extract_expired_token(self):
        from app.credentials import _extract_key_from_profile
        profile = {"type": "token", "access": "old-tok", "expires": time.time() - 100}
        assert _extract_key_from_profile(profile) is None

    def test_extract_raw_token(self):
        from app.credentials import _extract_key_from_profile
        profile = {"type": "token", "token": "raw-setup-token"}
        assert _extract_key_from_profile(profile) == "raw-setup-token"

    def test_extract_secretref_skipped(self):
        from app.credentials import _extract_key_from_profile
        profile = {"type": "api_key", "keyRef": {"source": "exec", "provider": "keychain"}}
        assert _extract_key_from_profile(profile) is None

    def test_extract_empty(self):
        from app.credentials import _extract_key_from_profile
        assert _extract_key_from_profile({}) is None

    def test_read_fake_openclaw(self, tmp_path):
        """Simulate reading from an OpenClaw auth-profiles.json."""
        from app.credentials import _extract_key_from_profile, _normalize_provider
        profiles_data = {
            "profiles": {
                "anthropic:default": {
                    "type": "api_key", "provider": "anthropic", "key": "sk-ant-test"
                },
                "openai:work": {
                    "type": "api_key", "provider": "openai", "key": "sk-openai-test"
                },
            }
        }
        p = tmp_path / "auth-profiles.json"
        p.write_text(json.dumps(profiles_data))
        data = json.loads(p.read_text())
        found = {}
        for pid, profile in data["profiles"].items():
            provider = _normalize_provider(profile.get("provider", ""))
            key = _extract_key_from_profile(profile)
            if key:
                found[provider] = key
        assert found["anthropic"] == "sk-ant-test"
        assert found["openai"] == "sk-openai-test"


# ═══════════════════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════════════════

class TestAnalytics:
    """Tests for app.analytics — report generation from logs."""

    @pytest.fixture
    def log_file(self, tmp_path):
        from app.storage import append_jsonl
        p = str(tmp_path / "stats.jsonl")
        for i in range(10):
            append_jsonl(p, {
                "ts": time.time() - i * 60,
                "task_type": "code" if i % 2 == 0 else "text",
                "model": "qwen_coder" if i % 3 == 0 else "sonnet",
                "provider": "openrouter",
                "tokens_in": 100 + i * 10,
                "tokens_out": 200 + i * 20,
                "cost_usd": 0.001 * (i + 1),
                "latency_ms": 100 + i * 50,
            })
        return p

    def test_report_total(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file)
        assert report["total_requests"] == 10

    def test_report_cost(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file)
        assert report["total_cost_usd"] > 0

    def test_report_by_model(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file)
        assert "by_model" in report
        assert len(report["by_model"]) > 0

    def test_report_by_task(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file)
        assert "by_task" in report
        assert "code" in report["by_task"]

    def test_report_latency(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file)
        assert report["latency"]["p50_ms"] > 0
        assert report["latency"]["p95_ms"] >= report["latency"]["p50_ms"]

    def test_report_model_filter(self, log_file):
        from app.analytics import generate_report
        report = generate_report(log_file, model_filter="sonnet")
        assert report["total_requests"] < 10
        assert all(m == "sonnet" for m in report["by_model"])

    def test_report_empty(self, tmp_path):
        from app.analytics import generate_report
        report = generate_report(str(tmp_path / "empty.jsonl"))
        assert report["total_requests"] == 0

    def test_cli_format(self, log_file):
        from app.analytics import generate_report, format_report_cli
        report = generate_report(log_file)
        text = format_report_cli(report)
        assert "RouteIQ Analytics Report" in text
        assert "By Model" in text


# ═══════════════════════════════════════════════════════════════════
# Tracing
# ═══════════════════════════════════════════════════════════════════

class TestTracing:
    """Tests for app.tracing — OpenTelemetry (no-op without SDK)."""

    def test_trace_context_manager(self):
        from app.tracing import trace_route
        with trace_route("code", "qwen", "openrouter") as span_data:
            span_data["tokens_in"] = 100
            span_data["cost_usd"] = 0.001
        # Should not raise

    def test_trace_with_exception(self):
        from app.tracing import trace_route
        with pytest.raises(ValueError):
            with trace_route("code", "qwen", "openrouter"):
                raise ValueError("test error")


# ═══════════════════════════════════════════════════════════════════
# Alerts
# ═══════════════════════════════════════════════════════════════════

class TestAlerts:
    """Tests for app.alerts — dispatcher doesn't crash."""

    def test_send_none_channel(self):
        from app.alerts import AlertManager
        am = AlertManager()
        am.send("test alert")  # should not raise

    def test_module_shortcut(self):
        from app.alerts import send_alert
        send_alert("test")  # should not raise


# ═══════════════════════════════════════════════════════════════════
# Router (integration-level)
# ═══════════════════════════════════════════════════════════════════

class TestRouter:
    """Tests for app.router — initialization and status."""

    def test_init(self):
        from app.router import Router
        router = Router()
        assert router is not None

    def test_full_status(self):
        from app.router import Router
        router = Router()
        status = router.full_status()
        assert "budget" in status
        assert "cache" in status
        assert "sessions" in status
        assert "credentials" in status

    def test_budget_status(self):
        from app.router import Router
        router = Router()
        budget = router.budget_status()
        assert "balance_usd" in budget
        assert "mode" in budget

    def test_cache_stats(self):
        from app.router import Router
        router = Router()
        stats = router.cache_stats()
        assert "size" in stats

    def test_mode_settings(self):
        from app.router import MODE_SETTINGS
        assert set(MODE_SETTINGS.keys()) == {"easy", "medium", "hard", "god"}
        for mode, cfg in MODE_SETTINGS.items():
            assert "max_tokens" in cfg
            assert "temperature" in cfg

    def test_router_request_defaults(self):
        from app.router import RouterRequest
        req = RouterRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.mode == "medium"
        assert req.stream is False
        assert req.tools is None
        assert req.profile is None


# ═══════════════════════════════════════════════════════════════════
# Package metadata
# ═══════════════════════════════════════════════════════════════════

class TestPackage:
    """Tests for package configuration."""

    def test_version(self):
        from app import __version__
        assert __version__ == "2.4.0"

    def test_pyproject_valid(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            cfg = tomllib.load(f)
        assert cfg["project"]["name"] == "routeiq"
        assert cfg["project"]["version"] == "2.4.0"
        assert "scripts" in cfg["project"]


# ═══════════════════════════════════════════════════════════════════
# Multimodal Detection
# ═══════════════════════════════════════════════════════════════════

class TestMultimodal:
    """Tests for image detection in messages."""

    def test_openai_image_format(self):
        from app.classifier import detect_images
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        assert detect_images(msgs) is True

    def test_anthropic_image_format(self):
        from app.classifier import detect_images
        msgs = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
        ]}]
        assert detect_images(msgs) is True

    def test_no_images(self):
        from app.classifier import detect_images
        assert detect_images([{"role": "user", "content": "hello"}]) is False

    def test_text_array_no_images(self):
        from app.classifier import detect_images
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert detect_images(msgs) is False

    def test_empty_messages(self):
        from app.classifier import detect_images
        assert detect_images([]) is False

    def test_image_escalates_to_vision(self):
        from app.classifier import classify_with_modifiers
        result = classify_with_modifiers([{"role": "user", "content": [
            {"type": "text", "text": "explain this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}])
        assert result["task_type"] == "vision"
        assert result["has_images"] is True


# ═══════════════════════════════════════════════════════════════════
# Doctor
# ═══════════════════════════════════════════════════════════════════

class TestDoctor:
    """Tests for app.doctor — health checks."""

    def test_run_doctor(self):
        from app.doctor import run_doctor
        result = run_doctor()
        assert "checks" in result
        assert "summary" in result
        assert result["summary"]["total"] > 0

    def test_format_cli(self):
        from app.doctor import run_doctor, format_doctor_cli
        result = run_doctor()
        text = format_doctor_cli(result)
        assert "RouteIQ Doctor" in text

    def test_check_has_required_fields(self):
        from app.doctor import run_doctor
        result = run_doctor()
        for check in result["checks"]:
            assert "name" in check
            assert "status" in check
            assert check["status"] in ("ok", "warn", "fail")


# ═══════════════════════════════════════════════════════════════════
# Config Hot-Reload
# ═══════════════════════════════════════════════════════════════════

class TestHotReload:
    """Tests for config hot-reload in TaskPolicy."""

    def test_reload_on_change(self, tmp_path):
        cfg = {"task_chains": {"text": ["m"]}, "models": {"m": {"id": "x", "priority": 50, "context_length": 4096}}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))

        from app.policy import TaskPolicy
        policy = TaskPolicy(str(p))
        assert policy.get_chain("text") == ["m"]

        # Modify
        time.sleep(0.01)
        cfg["task_chains"]["text"] = ["m", "n"]
        p.write_text(yaml.dump(cfg))

        assert policy.reload_if_changed() is True
        assert policy.get_chain("text") == ["m", "n"]

    def test_no_reload_if_unchanged(self, tmp_path):
        cfg = {"task_chains": {"text": ["m"]}, "models": {}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))

        from app.policy import TaskPolicy
        policy = TaskPolicy(str(p))
        assert policy.reload_if_changed() is False


# ═══════════════════════════════════════════════════════════════════
# Tool Calls Passthrough
# ═══════════════════════════════════════════════════════════════════

class TestToolCallsPassthrough:
    """Tests for tool_calls in RouterResponse."""

    def test_response_with_tool_calls(self):
        from app.router import RouterResponse
        resp = RouterResponse(
            content="", model_used="m", provider="p",
            tokens_in=10, tokens_out=20, cost_usd=0.001,
            latency_ms=100, task_type="code",
            tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "get_weather"}}],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_response_without_tool_calls(self):
        from app.router import RouterResponse
        resp = RouterResponse(
            content="hello", model_used="m", provider="p",
            tokens_in=10, tokens_out=20, cost_usd=0.001,
            latency_ms=100, task_type="text",
        )
        assert resp.tool_calls is None
