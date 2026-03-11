#!/usr/bin/env python3
"""
RouteIQ — Integration tests (REAL API calls).

Requires:
  - OPENROUTER_KEY set in .env or environment
  - Internet access
  - ~$0.05 in OpenRouter credits

Usage:
    python tests/test_integration.py
    python tests/test_integration.py --verbose

These tests make actual API calls and cost real money (very little).
They validate the full pipeline: classify → route → call → track.
"""
import json
import os
import sys
import time
from pathlib import Path

# Load .env
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.router import Router, RouterRequest
from app.classifier import classify_task, classify_with_modifiers

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

passed = 0
failed = 0
total_cost = 0.0
errors = []


def ok(name, detail=""):
    global passed
    passed += 1
    detail_str = f"  ({detail})" if detail else ""
    print(f"  ✅ {name}{detail_str}")


def fail(name, msg):
    global failed
    failed += 1
    errors.append(f"{name}: {msg}")
    print(f"  ❌ {name}: {msg}")


def log(msg):
    if VERBOSE:
        print(f"     → {msg}")


# ── Preflight ──

print("=" * 60)
print("RouteIQ — Integration Tests (real API calls)")
print("=" * 60)

key = os.getenv("OPENROUTER_KEY", "")
if not key or key.startswith("your_"):
    print("\n❌ OPENROUTER_KEY not set or is a placeholder.")
    print("   Set it in .env or environment before running integration tests.")
    sys.exit(1)

print(f"\n  Key: {key[:12]}...{key[-4:]}")
print(f"  Verbose: {VERBOSE}")
print()

router = Router()
budget_before = router.budget_status()
print(f"  Balance before: ${budget_before['balance_usd']}")
print()


# ── Test 1: Simple text prompt (should use cheap model) ──

print("── Test 1: Simple text prompt ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
        mode="easy",
    )
    resp = router.route(req)
    assert resp.content and len(resp.content) > 0, "Empty response"
    total_cost += resp.cost_usd
    ok(f"Text → {resp.model_used}",
       f"${resp.cost_usd:.5f}, {resp.latency_ms:.0f}ms, {resp.tokens_out} tokens")
    log(f"Response: {resp.content[:80]}")
except Exception as e:
    fail("Simple text", str(e))


# ── Test 2: Code prompt (should route to code model) ──

print("\n── Test 2: Code prompt ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Write a Python function that returns the factorial of n. Keep it short."}],
        mode="easy",
    )
    resp = router.route(req)
    assert resp.content and ("def " in resp.content or "factorial" in resp.content.lower()), \
        f"Doesn't look like code: {resp.content[:50]}"
    assert resp.task_type == "code", f"Expected task_type=code, got {resp.task_type}"
    total_cost += resp.cost_usd
    ok(f"Code → {resp.model_used}",
       f"${resp.cost_usd:.5f}, {resp.latency_ms:.0f}ms, task={resp.task_type}")
    log(f"Response: {resp.content[:80]}")
except Exception as e:
    fail("Code prompt", str(e))


# ── Test 3: Summarize prompt ──

print("\n── Test 3: Summarize prompt ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Summarize in one sentence: Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991."}],
        mode="easy",
    )
    resp = router.route(req)
    assert resp.content and len(resp.content) > 5, "Response too short"
    total_cost += resp.cost_usd
    ok(f"Summarize → {resp.model_used}",
       f"${resp.cost_usd:.5f}, task={resp.task_type}")
    log(f"Response: {resp.content[:80]}")
except Exception as e:
    fail("Summarize", str(e))


# ── Test 4: Caching (second call should be $0) ──

print("\n── Test 4: Response caching ──")
try:
    msgs = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
    req1 = RouterRequest(messages=msgs, mode="easy")
    resp1 = router.route(req1)
    total_cost += resp1.cost_usd

    req2 = RouterRequest(messages=msgs, mode="easy")
    resp2 = router.route(req2)

    assert resp2.cached is True, f"Second call was not cached (cached={resp2.cached})"
    assert resp2.cost_usd == 0.0, f"Cached response should cost $0, got ${resp2.cost_usd}"
    assert resp2.content == resp1.content, "Cached response content differs"
    ok("Cache hit", f"First: ${resp1.cost_usd:.5f}, Second: $0.00000 (cached)")
except Exception as e:
    fail("Caching", str(e))


# ── Test 5: Session persistence ──

print("\n── Test 5: Session persistence ──")
try:
    session_id = f"test-session-{int(time.time())}"
    req1 = RouterRequest(
        messages=[{"role": "user", "content": "Hello, remember me."}],
        mode="easy", session_id=session_id,
    )
    resp1 = router.route(req1)
    total_cost += resp1.cost_usd

    req2 = RouterRequest(
        messages=[
            {"role": "user", "content": "Hello, remember me."},
            {"role": "assistant", "content": resp1.content},
            {"role": "user", "content": "What did I just say?"},
        ],
        mode="easy", session_id=session_id,
    )
    resp2 = router.route(req2)
    total_cost += resp2.cost_usd

    assert resp2.model_used == resp1.model_used, \
        f"Model changed: {resp1.model_used} → {resp2.model_used}"
    ok(f"Session pinned to {resp1.model_used}",
       f"2 turns, same model")
except Exception as e:
    fail("Session persistence", str(e))


# ── Test 6: Classifier accuracy on edge cases ──

print("\n── Test 6: Classifier edge cases ──")
edge_cases = [
    ("yo fix that thing with the thingamajig in main.py", "code"),
    ("make this component responsive", "code"),
    ("what does this error mean: TypeError: cannot read property", "code"),
    ("can you explain quantum entanglement simply", "text"),
    ("TL;DR this whole thing", "summarize"),
]
for prompt, expected in edge_cases:
    result = classify_task(prompt)
    if result == expected:
        ok(f"classify({prompt[:45]}...) → {result}")
    else:
        # Not a hard fail for edge cases — just a warning
        print(f"  ⚠️  classify({prompt[:45]}...) → {result} (expected {expected})")


# ── Test 7: Budget tracking ──

print("\n── Test 7: Budget tracking ──")
try:
    budget_after = router.budget_status()
    spent_delta = budget_after["spent_usd"] - budget_before.get("spent_usd", 0)
    assert spent_delta > 0, "No spend recorded"
    assert budget_after["balance_usd"] < budget_before["balance_usd"], "Balance didn't decrease"
    ok(f"Budget tracked: ${spent_delta:.5f} spent",
       f"Balance: ${budget_after['balance_usd']:.4f}")
except Exception as e:
    fail("Budget tracking", str(e))


# ── Test 8: Streaming ──

print("\n── Test 8: Streaming ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Write a short paragraph about why the sky is blue. Be detailed."}],
        mode="easy", stream=True,
    )
    chunks = []
    for chunk in router.route_stream(req):
        chunks.append(chunk)
    full_text = "".join(chunks)
    assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
    assert len(full_text) > 10, f"Response too short: {full_text[:50]}"
    ok(f"Streaming: {len(chunks)} chunks", f"{len(full_text)} chars total")
    log(f"Response: {full_text[:80]}")
except Exception as e:
    fail("Streaming", str(e))


# ── Test 9: Profile override ──

print("\n── Test 9: Routing profile ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Hi"}],
        profile="eco",
    )
    resp = router.route(req)
    total_cost += resp.cost_usd
    # Eco should pick cheap/free model
    ok(f"Profile 'eco' → {resp.model_used}", f"${resp.cost_usd:.5f}")
except Exception as e:
    fail("Profile override", str(e))


# ── Test 10: Model alias ──

print("\n── Test 10: Model alias ──")
try:
    req = RouterRequest(
        messages=[{"role": "user", "content": "Hi, just testing."}],
        model="flash",  # alias for gemini_flash
    )
    resp = router.route(req)
    total_cost += resp.cost_usd
    ok(f"Alias 'flash' → {resp.model_used}", f"${resp.cost_usd:.5f}")
except Exception as e:
    fail("Model alias", str(e))


# ── Test 11: Report with real data ──

print("\n── Test 11: Report generation ──")
try:
    from app.analytics import generate_report
    report = generate_report("state/model-stats.jsonl")
    assert report["total_requests"] > 0, "No requests in report"
    ok(f"Report: {report['total_requests']} requests, ${report['total_cost_usd']:.5f}")
except Exception as e:
    fail("Report", str(e))


# ── Summary ──

budget_final = router.budget_status()
total_spent = budget_final["spent_usd"] - budget_before.get("spent_usd", 0)

print()
print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print(f"Total API cost: ${total_spent:.5f}")
print(f"Balance remaining: ${budget_final['balance_usd']:.4f}")

if errors:
    print(f"\nFailures:")
    for e in errors:
        print(f"  ❌ {e}")

print("=" * 60)

if failed > 0:
    sys.exit(1)
