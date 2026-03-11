# Changelog

All notable changes to RouteIQ are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.4.0] — 2026-03-08

### Fixed (code audit — 6 bugs found and resolved)
- **Backend singletons no longer cache API keys** — keys are read fresh from env on each request, so credential discovery always works regardless of import order
- **Eco profile now correctly selects cheapest models** — was broken: `force_free` flag wasn't checked in `_build_chain`. Eco now uses `chain_limit=2` (cheapest 2 models); `free` profile filters to free-only
- **Streaming now passes `tools` to providers** — agentic requests via SSE streaming no longer lose tool definitions
- **Version strings unified** — `__version__`, pyproject.toml, FastAPI, health endpoint, and tracing all read from single source (`app/__init__.py`)
- **Thread lock race condition fixed** — `_get_lock()` now holds guard for entire check-and-create operation
- **Unicode word boundaries in classifier** — regex `\b` replaced with `(?<!\w)..(?!\w)` for correct Cyrillic matching

### Changed
- Dashboard rewritten with Rich library (was curses) — proper Unicode panels, no rendering artifacts, graceful fallback if Rich not installed
- `OPENROUTER_API_KEY` accepted as alias for `OPENROUTER_KEY` across all modules
- CLI rewritten — prompts and subcommands coexist cleanly (`routeiq "hello"` and `routeiq serve` both work)

## [2.2.0] — 2026-03-07

### Added
- **Credential auto-discovery** — reads API keys from OpenClaw (`auth-profiles.json`), Claude Code (`setup-token`), and NadirClaw (`credentials.json`) without manual configuration
- **`routeiq doctor`** — health check command that validates config, API keys, credential sources, state directory, Ollama availability
- **Multimodal detection** — auto-detects images in messages (OpenAI and Anthropic formats) and escalates task type to `vision`
- **Config hot-reload** — router checks config file mtime on each request and reloads if changed (no restart needed)
- **Real cost tracking** — reads actual cost from OpenRouter response (`usage.cost`) instead of estimating from token counts
- **Tool calls passthrough** — `tool_calls` from model responses are forwarded to the client; `finish_reason: "tool_calls"` set correctly
- `routeiq credentials` command — shows discovered credentials and their sources
- `pip install -e .` support via `pyproject.toml` with optional deps (`[embeddings]`, `[telemetry]`, `[all]`)
- One-line installer: `curl | bash` via `install.sh`
- `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`
- GitHub Actions CI (`.github/workflows/test.yml`) — tests on Python 3.10–3.13
- Example configs: `examples/economy.yaml`, `examples/premium.yaml`, `examples/local-only.yaml`
- 100 demo log entries in `state/model-stats.jsonl` for out-of-the-box `routeiq report`

### Changed
- Router checks config file on each request for hot-reload
- Router uses real provider cost when available, falls back to estimate
- OpenRouter backend sends `usage: {include: true}` for accurate cost data
- Server returns `tool_calls` in message and `finish_reason: "tool_calls"` when model uses tools

---

## [2.1.0] — 2026-03-07

### Added
- **Agentic detection** — auto-detects tool use (tools, tool_calls, tool role messages, agent system prompts) and forces complex model
- **Reasoning detection** — identifies chain-of-thought prompts (2+ reasoning markers) and routes to reasoning-optimized models
- **Multi-provider backends** — direct API calls to Anthropic, OpenAI, Google GenAI, and Ollama, in addition to OpenRouter
- **Routing profiles** — `auto`, `eco`, `premium`, `free`, `reasoning` strategies selectable per request
- **Model aliases** — short names (`sonnet`, `flash`, `gpt4`, `local`) resolve to full model configs
- **Analytics report** — `routeiq report` with cost breakdown by model, latency percentiles (p50/p95/p99), task distribution, hourly trends
- **OpenTelemetry tracing** — optional distributed tracing with GenAI semantic conventions
- Ollama model (`llama_local`) in default config
- `/v1/report` API endpoint

### Changed
- `classify_with_modifiers()` now returns agentic/reasoning flags alongside task type
- Router builds model chain considering profile, budget mode, and agentic/reasoning overrides
- Server passes through `tools`, `tool_choice`, `profile`, `model` fields from OpenAI-format requests

---

## [2.0.0] — 2026-03-07

### Added
- **OpenAI-compatible HTTP proxy** (`/v1/chat/completions`) — drop-in replacement for Cursor, Claude Code, Continue, and any OpenAI SDK client
- **SSE streaming** — full server-sent events support in proxy and CLI
- **Hybrid task classifier** — sentence embeddings (all-MiniLM-L6-v2) when available, weighted keyword scoring as fallback; supports 8 task types with EN + RU keywords
- **LRU response cache** — identical prompts skip API entirely ($0 cost, ~0ms latency); configurable TTL and max size
- **Session persistence** — pins model to conversation ID so multi-turn threads don't bounce between models
- **Context-window awareness** — auto-filters models whose context window is too small for the prompt
- **Composite model scoring** — `score_model()` now actually used in routing; sorts chain by cost + quality + latency weights
- **Retry with exponential backoff on 429** — configurable max attempts and backoff factor
- **Daily and monthly spend limits** — independent of total balance, hard stop on exceed
- Circuit breaker per model (5 errors in 60s window)
- 4 quality modes: `easy`, `medium`, `hard`, `god`
- `/v1/models`, `/v1/status`, `/v1/budget`, `/v1/cache`, `/health` endpoints
- Docker support (Dockerfile + docker-compose.yml)
- Full test suite (44 tests)

### Fixed
- **API key in .env** — replaced with placeholder, added `.gitignore`
- **task_chains YAML path** — was nested under `policy:` but code looked at top level; now auto-hoists
- **score_model() was dead code** — now integrated into chain sorting
- **EWMA was actually simple moving average** — rewritten with real exponential decay (half-life 60s)
- **Keyword classifier false positives** — replaced naive substring matching with weighted scoring and word boundaries

### Changed
- Classifier extracted to dedicated `classifier.py` (was inline in `policy.py`)
- Budget state persistence uses atomic write (write-to-tmp + rename)
- Config restructured: `task_chains` at top level, `context_length` added to all models

---

## [1.0.0] — 2026-03-06

### Added
- Initial release
- CLI-only interface
- Keyword-based task classifier (8 types)
- OpenRouter as single backend
- Budget tracking with alerts (Telegram, email, webhook, Slack)
- YAML-based model configuration
- State persistence in JSON files
