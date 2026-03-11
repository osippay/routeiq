# RouteIQ 🧠⚡

[![Tests](https://github.com/osippos/routeiq/actions/workflows/test.yml/badge.svg)](https://github.com/osippos/routeiq/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Smart LLM router that auto-selects the optimal model for each task — with agentic detection, multi-provider backends, budget control, and OpenAI-compatible proxy.**

Stop paying for Claude Opus when GPT-4o-mini handles the job. RouteIQ classifies your prompt, detects tool use and reasoning needs, picks the cheapest capable model, and tracks every cent.

```
Your App ──→ RouteIQ (:8000/v1) ──→ simple text   ──→ Gemini Flash  ($0.075/M)
                                 ──→ code          ──→ Qwen Coder   (FREE)
                                 ──→ tool use 🤖   ──→ Claude Sonnet ($3/M)
                                 ──→ reasoning 🧠  ──→ Claude Opus   ($15/M)
                                 ──→ local/private  ──→ Ollama        (FREE)
```

---

## Why RouteIQ

| Problem | Solution |
|---------|----------|
| Paying $250/mo routing everything to Opus | 8-type classifier routes 70%+ of prompts to cheap models |
| Tool-use prompts break on cheap models | Agentic detection auto-escalates tool use to capable models |
| Chain-of-thought on the wrong model | Reasoning detection forces CoT to reasoning-optimized models |
| Locked into one provider | Multi-provider: OpenRouter + direct Anthropic/OpenAI/Google/Ollama |
| No idea where money goes | Real-time budget + analytics with `routeiq report` |
| Can't use with Cursor, Claude Code | OpenAI-compatible proxy — drop-in replacement |
| Same prompt = pay twice | LRU response cache — identical prompts = $0 |
| Model bouncing mid-conversation | Session pinning keeps one model per thread |

---

## Quick Start

### One-line install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/osippos/routeiq/main/install.sh | bash
```

This clones the repo, creates a virtual environment, installs dependencies, and adds `routeiq` to your PATH. Run it again to update.

### Manual install

```bash
git clone https://github.com/osippos/routeiq
cd routeiq
pip install -e .

# Add your keys
cp .env.example .env
# edit .env → OPENROUTER_KEY=sk-or-...
```

### Try it

```bash
# CLI
routeiq "write a fibonacci function"
# ✅ Model: qwen_coder via openrouter | Cost: $0.00000 (FREE)

# HTTP proxy (for Cursor, Claude Code, etc.)
routeiq serve
# 🚀 RouteIQ proxy on http://localhost:8000/v1

# Check what credentials were discovered
routeiq credentials

# Analytics
routeiq report
```

### Use with any OpenAI-compatible tool

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=anything  # RouteIQ uses its own key
```

### Docker

```bash
docker compose up -d
```

---

## Architecture

```
┌────────────────┐
│   Your App     │
│ (Cursor, SDK)  │───────────────────────────────┐
└────────────────┘                               │
                                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                        RouteIQ                               │
│                                                              │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Classifier │  │ Agentic  │  │ Reasoning│  │  Routing  │  │
│  │ (8 types) │  │ Detector │  │ Detector │  │  Profile  │  │
│  │           │  │          │  │          │  │           │  │
│  │ keywords  │  │ tools?   │  │ CoT      │  │ auto/eco  │  │
│  │ or embed  │  │ agents?  │  │ markers? │  │ premium/  │  │
│  └─────┬─────┘  └─────┬────┘  └────┬─────┘  │ free/     │  │
│        └───────────────┴────────────┘        │ reasoning │  │
│                    │                         └─────┬─────┘  │
│                    ▼                               │        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │        │
│  │ Scorer   │→│ Context  │→│  Cache   │←────────┘        │
│  │(cost+    │  │ Filter   │  │  (LRU)  │                   │
│  │ quality+ │  │          │  │         │                   │
│  │ latency) │  │ auto-    │  │ hit→$0  │                   │
│  └──────────┘  │ swap     │  │ miss↓   │                   │
│                └──────────┘  └────┬────┘                   │
│                                   │                         │
│  ┌──────────┐  ┌──────────┐  ┌───┴─────────────────────┐   │
│  │ Session  │  │ Budget   │  │   Multi-Provider        │   │
│  │ Manager  │  │ Tracker  │  │   Backend               │   │
│  │ (pin     │  │ (EWMA    │  │                         │   │
│  │  model)  │  │  alerts) │  │ OpenRouter │ Anthropic  │   │
│  └──────────┘  └──────────┘  │ OpenAI    │ Google     │   │
│                              │ Ollama (local, free)    │   │
│                              └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Features

### Hybrid Task Classifier

Routes prompts to one of **8 task types**:

| Task | Example | Default Model | Cost |
|------|---------|---------------|------|
| `text` | "write a blog post" | GPT-4o-mini | $0.15/M |
| `code` | "write a function" | Qwen Coder | **FREE** |
| `image` | "generate a picture" | DALL-E 3 | $0.04/img |
| `audio` | "transcribe this" | Whisper | $0.006/min |
| `vision` | "what's in this photo?" | Claude Sonnet | $3/M |
| `think` | "complex reasoning" | Claude Opus | $15/M |
| `strategy` | "business plan" | Claude Opus | $15/M |
| `summarize` | "TL;DR this" | Gemini Flash | $0.075/M |

Two classifier backends: sentence embeddings (ML, ~10ms) or weighted keyword scoring (no deps, fast, EN+RU).

### Agentic Detection 🤖

Automatically detects tool-use requests and forces a capable model:

- `tools` or `tool_choice` in the request → agentic
- Messages with `role: "tool"` → agentic
- System prompts mentioning tools/agents → agentic
- Assistant messages with `tool_calls` → agentic

Cheap models break tool use. Agentic detection prevents this.

### Reasoning Detection 🧠

Detects chain-of-thought prompts (2+ reasoning markers) and routes to reasoning-optimized models. Markers include "step by step", "think carefully", "analyze", "prove", "chain of thought", etc.

### Multi-Provider Backends

Not locked into one provider. RouteIQ calls the optimal backend directly:

| Provider | Models | Latency | Setup |
|----------|--------|---------|-------|
| **OpenRouter** | 100+ models | ~40ms overhead | `OPENROUTER_KEY` |
| **Anthropic** | Claude Sonnet, Opus | Direct, low latency | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o, GPT-4o-mini | Direct | `OPENAI_API_KEY` |
| **Google** | Gemini Flash, Pro | Direct, free tier | `GOOGLE_API_KEY` |
| **Ollama** | Llama, Mistral, etc. | Local, ~0ms | Ollama running locally |

If you have a direct API key, RouteIQ uses it. Otherwise, falls back to OpenRouter.

### Credential Auto-Discovery 🔑

RouteIQ automatically finds API keys without manual configuration — just install and go:

| Priority | Source | What it reads |
|----------|--------|---------------|
| 1 | **Environment / .env** | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc. |
| 2 | **OpenClaw** | `~/.openclaw/agents/main/agent/auth-profiles.json` |
| 3 | **Claude Code** | Setup-token from `~/.claude.json` |
| 4 | **NadirClaw** | `~/.nadirclaw/credentials.json` |

If you have OpenClaw installed and authenticated — RouteIQ auto-discovers all your tokens (API keys, OAuth tokens, setup-tokens). No extra config needed.

```bash
routeiq credentials   # see what was discovered and from where

# Example output:
#   🐾 OpenClaw (auto-discovered)
#     ✅ anthropic    sk-ant-a3...  (auth-profiles.json)
#     ✅ openai       sk-proj-...   (auth-profiles.json)
#   📁 Environment variables (.env)
#     ✅ openrouter   sk-or-v1...   (OPENROUTER_KEY)
```

Works standalone (with API keys in `.env`) or as an extension to OpenClaw — universal router for any stack.

### Routing Profiles

```bash
routeiq "prompt" --profile auto       # 🔄 classifier picks
routeiq "prompt" --profile eco        # 💚 cheapest only
routeiq "prompt" --profile premium    # 💎 best quality
routeiq "prompt" --profile free       # 🆓 free models only
routeiq "prompt" --profile reasoning  # 🧠 CoT-optimized
```

### Model Aliases

Short names instead of full IDs:

```bash
routeiq --model sonnet "review this"   # → anthropic/claude-sonnet-4-5
routeiq --model flash "summarize this"  # → google/gemini-2.5-flash-lite
routeiq --model local "explain this"    # → ollama/llama3.1:8b
```

### Live Terminal Dashboard

Monitor routing in real-time:

```bash
routeiq dashboard
```

```
 ╭─────────────────────────────────────╮
 │  ⚡  R O U T E  I Q   v2.3        │
 │     Smart LLM Router               │
 ╰─────────────────────────────────────╯

 ╭─── ⚡ Stats ──────────────────╮  ╭─── 📊 Routing Distribution ────╮
 │ Total Requests        247     │  │ code       144  ████████  58%  │
 │ Req/min (5m)          3.2     │  │ text        71  ████     29%  │
 │ Actual Cost       $1.7373     │  │ summarize   32  ██       13%  │
 │ Without Routing   $3.0270     │  │ think        7  ▌         3%  │
 │ Saved      $1.2897 (42.6%)    │  │ vision       6  ▌         2%  │
 │ Balance           $98.26      │  ╰────────────────────────────────╯
 ╰───────────────────────────────╯
 ╭─── 📋 Recent Requests ─────────────────────────────────────────────╮
 │ Time      Task        Model           Latency  Tokens     Cost    │
 │ 01:22:55  code        qwen_coder        180ms     423  $0.00000  │
 │ 01:20:12  text        gemini_flash       95ms     286  $0.00002  │
 │ 01:18:44  think       opus             1209ms    5242  $0.12340  │
 │ 01:15:33  summarize   gemini_flash      135ms    2150  $0.00034  │
 ╰─────────────────────────────────────────────────────────────────────╯
```

Built with [Rich](https://github.com/Textualize/rich) — no curses, proper Unicode rendering, works on macOS/Linux/Windows.

### Analytics Report

```bash
routeiq report
routeiq report --days 7
routeiq report --model sonnet --json
```

Output:
```
══════════════════════════════════════════
  RouteIQ Analytics Report
══════════════════════════════════════════

── Overview ──
  Total requests:    847
  Total cost:        $2.34
  Avg cost/request:  $0.002763
  Cached requests:   312 (36.8%)

── Latency ──
  P50: 230ms  |  P95: 890ms  |  P99: 1450ms

── By Model ──
  qwen_coder    523 reqs | $0.00000 (0.0%) | p50=180ms
  sonnet        201 reqs | $1.89000 (80.8%) | p50=450ms
  gemini_flash  123 reqs | $0.45000 (19.2%) | p50=120ms
```

### Budget Control

EWMA burn rate, daily/monthly limits, auto-downgrade, alerts at 50/80/100%.

### Response Caching

LRU cache with TTL. Identical prompts = $0, ~0ms.

### Session Persistence

Pin a model per conversation — no bouncing mid-thread.

### Context-Window Awareness

Auto-filters models that can't fit the prompt. No more `context_length_exceeded`.

### OpenAI-Compatible Proxy

```
POST /v1/chat/completions   — with SSE streaming
GET  /v1/models              — list models + aliases
GET  /v1/status              — budget + cache + sessions
GET  /v1/report              — analytics
GET  /v1/budget              — budget only
GET  /health                 — health check
```

### OpenTelemetry Tracing (Optional)

```bash
pip install opentelemetry-api opentelemetry-sdk
```

GenAI semantic conventions for production observability.

---

## Configuration

All config in `conf/router.yaml`. Add models, change chains, tune weights — no code changes, no retraining.

```yaml
# Add an Ollama model
models:
  my_local:
    id: ollama/deepseek-coder:6.7b
    provider: ollama
    priority: 80
    free: true
    context_length: 16384
    capabilities: [code]

# Add it to the code chain
task_chains:
  code: [my_local, qwen_coder, sonnet]

# Add an alias
aliases:
  deepseek: my_local
```

---

## CLI Reference

```bash
routeiq "prompt"                    # send prompt (auto-classify)
routeiq "prompt" --mode god         # quality mode
routeiq "prompt" --profile premium  # routing profile
routeiq "prompt" --model sonnet     # explicit model/alias
routeiq "prompt" --stream           # SSE streaming
routeiq "prompt" --session chat-1   # session pinning
routeiq serve                       # start HTTP proxy
routeiq serve --port 8856           # custom port
routeiq dashboard                   # live terminal dashboard (real-time)
routeiq report                      # analytics
routeiq report --days 7 --json      # filtered, JSON output
routeiq models                      # list models + aliases
routeiq credentials                 # show discovered API keys + sources
routeiq doctor                      # health check — validate config, keys, connectivity
routeiq status                      # full status
routeiq budget                      # budget only
```

---

## Project Structure

```
routeiq/
├── cli.py                 # CLI entry point
├── install.sh             # one-line installer (curl | bash)
├── pyproject.toml         # package config (pip install -e .)
├── conf/router.yaml       # all configuration
├── requirements.txt       # minimal deps
├── Dockerfile
├── docker-compose.yml
├── .env.example           # env template
├── CHANGELOG.md           # version history
├── CONTRIBUTING.md         # contributor guidelines
├── LICENSE                 # MIT
├── app/
│   ├── server.py          # FastAPI OpenAI-compatible proxy
│   ├── router.py          # core routing logic
│   ├── classifier.py      # 8-type classifier + agentic + reasoning detection
│   ├── backends.py        # multi-provider: OpenRouter, Anthropic, OpenAI, Google, Ollama
│   ├── credentials.py     # auto-discovery: OpenClaw, Claude Code, NadirClaw
│   ├── policy.py          # model config, scoring, aliases, profiles, context checks
│   ├── budget.py          # budget tracking + EWMA burn rate
│   ├── cache.py           # LRU response cache
│   ├── session.py         # session persistence
│   ├── analytics.py       # report generation from logs
│   ├── dashboard.py       # live terminal dashboard (curses TUI)
│   ├── alerts.py          # multi-channel alert dispatcher
│   ├── doctor.py          # health checks and diagnostics (routeiq doctor)
│   ├── storage.py         # atomic file operations
│   └── tracing.py         # optional OpenTelemetry
└── tests/
    ├── test_routeiq.py        # 116 unit tests (20 classes)
    ├── test_integration.py    # 15 integration tests (real API calls)
    └── RESULTS.md             # latest test results with proof
```

---

## Tested & Verified

All tests pass on real hardware with real API calls (not mocks):

```
Unit tests:        116 passed in 2.45s (Python 3.14.3, macOS)
Integration tests:  15 passed, 0 failed ($0.00020 spent)
```

| What was tested | Result |
|----------------|--------|
| Text → cheap model (Gemini Flash) | ✅ $0.00000 |
| Code → free model (Qwen Coder) | ✅ $0.00006 |
| Summarize → GPT-4o-mini | ✅ $0.00002 |
| Cache hit (same prompt twice) | ✅ Second call = $0 |
| Session persistence (2 turns) | ✅ Same model pinned |
| Classifier edge cases (5/5) | ✅ ".py", "TypeError", "component" detected |
| SSE streaming (215 chunks) | ✅ Token-by-token |
| Routing profile "eco" | ✅ Cheapest model selected |
| Model alias "flash" | ✅ Resolved and routed |
| Budget tracking | ✅ Matches OpenRouter dashboard |

Full results: [`tests/RESULTS.md`](tests/RESULTS.md)

---

## vs. Alternatives

| | RouteIQ | NadirClaw | RouteLLM | LiteLLM |
|--|---------|-----------|----------|---------|
| Task types | **8** (multi-class) | 2 (binary) | 2 (binary) | manual |
| Agentic detection | **✅** | ✅ | ❌ | ❌ |
| Reasoning detection | **✅** | ✅ | ❌ | ❌ |
| Multi-provider (direct) | **✅** 5 providers | ✅ | ❌ | ✅ |
| Ollama (local) | **✅** | ✅ | ❌ | ✅ |
| Budget auto-downgrade | **✅** | ❌ | ❌ | ❌ |
| Daily/monthly limits | **✅** | ✅ | ❌ | ✅ |
| Response caching | **✅** | ✅ | ❌ | ✅ |
| Session persistence | **✅** | ✅ | ❌ | ❌ |
| Context-window filter | **✅** | ✅ | ❌ | ❌ |
| Composite scoring | **✅** | ❌ | ✅ | ✅ |
| Model aliases | **✅** | ✅ | ❌ | ❌ |
| Routing profiles | **✅** 5 profiles | ✅ | ❌ | ✅ |
| Analytics / report | **✅** | ✅ | ❌ | ✅ |
| Live terminal dashboard | **✅** | ✅ | ❌ | ❌ |
| OpenTelemetry | **✅** | ✅ | ❌ | ✅ |
| OpenAI-compatible proxy | **✅** | ✅ | ✅ | ✅ |
| SSE streaming | **✅** | ✅ | ✅ | ✅ |
| YAML config (no retrain) | **✅** | ❌ | ❌ | ✅ |
| Credential auto-discovery | **✅** OpenClaw+Claude Code | ✅ OpenClaw only | ❌ | ❌ |
| No DB required | **✅** | ✅ | ❌ | ❌ |

---

## Roadmap

- [ ] OAuth login (OpenAI, Anthropic, Google subscriptions)
- [ ] Web dashboard with real-time charts
- [ ] A/B testing between models
- [ ] Custom classifier training on user data
- [ ] Prompt rewriting for model-specific optimization

---

## License

MIT
