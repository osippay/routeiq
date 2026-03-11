# Integration Test Results — RouteIQ v2.4.0

Last run: **2026-03-08** on macOS (Python 3.14.3)

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 15 |
| Passed | **15** |
| Failed | 0 |
| API cost | $0.00020 |
| Platform | macOS (Apple Silicon) |
| Python | 3.14.3 |

## Results

```
============================================================
RouteIQ — Integration Tests (real API calls)
============================================================

── Test 1: Simple text prompt ──
  ✅ Text → gemini_flash  ($0.00000, 2845ms, 6 tokens)

── Test 2: Code prompt ──
  ✅ Code → qwen_coder  ($0.00006, 5037ms, task=code)

── Test 3: Summarize prompt ──
  ✅ Summarize → gpt4o_mini  ($0.00002, task=summarize)

── Test 4: Response caching ──
  ✅ Cache hit  (First: $0.00002, Second: $0.00000 (cached))

── Test 5: Session persistence ──
  ✅ Session pinned to gemini_flash  (2 turns, same model)

── Test 6: Classifier edge cases ──
  ✅ classify("yo fix that thing in main.py") → code
  ✅ classify("make this component responsive") → code
  ✅ classify("TypeError: cannot read property") → code
  ✅ classify("explain quantum entanglement") → text
  ✅ classify("TL;DR this whole thing") → summarize

── Test 7: Budget tracking ──
  ✅ Budget tracked: $0.00010 spent

── Test 8: Streaming ──
  ✅ Streaming: 215 chunks (1106 chars total)

── Test 9: Routing profile ──
  ✅ Profile 'eco' → llama_groq  ($0.00001)

── Test 10: Model alias ──
  ✅ Alias 'flash' → gemini_flash  ($0.00001)

── Test 11: Report generation ──
  ✅ Report: 109 requests, $1.92295

============================================================
Results: 15 passed, 0 failed
Total API cost: $0.00020
============================================================
```

## Unit Tests

```
116 passed in 2.45s
```

All 20 test classes, 116 test methods — covering classifier, agentic detection,
reasoning detection, multimodal, cache, session, policy, budget, storage,
backends, credentials, analytics, tracing, alerts, router, hot-reload, and
tool calls passthrough.

## Routing Decisions

Real API calls confirmed correct routing:

| Prompt | Task Type | Model Selected | Cost |
|--------|-----------|---------------|------|
| "Say hello" | text | gemini_flash | $0.00000 |
| "Write factorial function" | code | qwen_coder | $0.00006 |
| "Summarize: Python is..." | summarize | gpt4o_mini | $0.00002 |
| "What is 2+2?" (cached) | text | (cache hit) | $0.00000 |
| Profile: eco | text | llama_groq | $0.00001 |
| Alias: flash | text | gemini_flash | $0.00001 |

## Doctor Output

```
🩺 RouteIQ Doctor
──────────────────────────────────────────────────
  ✅  Config file — 9 models, 8 task chains
  ✅  API key: OpenRouter
  ⚠️  API key: Anthropic (direct) — not set (optional)
  ⚠️  API key: OpenAI (direct) — not set (optional)
  ⚠️  API key: Google (direct) — not set (optional)
  ⚠️  OpenClaw credentials — not installed (optional)
  ✅  Claude Code token — found
  ✅  State directory — writable
  ✅  Model definitions — all 9 models valid
  ⚠️  Ollama — not running (optional)

  ✅ Healthy — 5 ok, 5 warnings, 0 failures
```

## OpenRouter Activity (screenshot verified)

- Spend: $0.00066
- Requests: 13
- Tokens: 662
- Models used: Llama 3.3 70B, Qwen2.5 Coder 32B, GPT-4o-mini, Claude Sonnet 4.5
