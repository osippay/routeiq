# Contributing to RouteIQ

Thanks for your interest in contributing! RouteIQ is open to contributions of all kinds: bug reports, feature requests, documentation, and code.

## Quick links

- [Issues](https://github.com/osippos/routeiq/issues) — bug reports and feature requests
- [Discussions](https://github.com/osippos/routeiq/discussions) — questions and ideas
- [CHANGELOG](CHANGELOG.md) — what changed and when

## Development setup

```bash
# Clone
git clone https://github.com/osippos/routeiq.git
cd routeiq

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run tests
python -m pytest tests/ -v

# Start the server locally
python cli.py serve
```

## Running tests

```bash
# Unit tests (no API key needed, no cost)
python -m pytest tests/test_routeiq.py -v

# Integration tests (requires OPENROUTER_KEY in .env, costs ~$0.02)
python tests/test_integration.py --verbose

# Quick smoke test (all modules)
python -c "
from app.router import Router
r = Router()
print(r.full_status())
"
```

Unit tests (116) cover all internal logic without making API calls. Integration tests (15) make real requests to OpenRouter and validate the full pipeline end-to-end: classification → routing → API call → budget tracking → caching → streaming.

## Project structure

```
routeiq/
├── app/
│   ├── server.py        # FastAPI proxy (OpenAI-compatible)
│   ├── router.py        # Core routing pipeline
│   ├── classifier.py    # Task classification + agentic/reasoning detection
│   ├── backends.py      # Provider backends (OpenRouter, Anthropic, OpenAI, Google, Ollama)
│   ├── credentials.py   # Credential auto-discovery (OpenClaw, Claude Code, etc.)
│   ├── policy.py        # Config, scoring, aliases, profiles
│   ├── budget.py        # Budget tracking + EWMA burn rate
│   ├── cache.py         # LRU response cache
│   ├── session.py       # Session persistence (model pinning)
│   ├── analytics.py     # Report generation from logs
│   ├── dashboard.py     # Live terminal dashboard (Rich)
│   ├── alerts.py        # Alert dispatcher (Telegram, Slack, email, webhook)
│   ├── doctor.py        # Health checks and diagnostics
│   ├── storage.py       # Atomic file operations
│   └── tracing.py       # Optional OpenTelemetry
├── conf/router.yaml     # All configuration
├── cli.py               # CLI entry point
├── tests/               # Test suite
└── state/               # Runtime state (gitignored)
```

## How to contribute

### Bug reports

Open an issue with:
- What you expected to happen
- What actually happened
- RouteIQ version (`python -c "from app import __version__; print(__version__)"`)
- Python version
- Steps to reproduce

### Feature requests

Open an issue describing:
- The problem you're trying to solve
- How you think it could be solved
- Whether you'd be willing to implement it

### Code contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add or update tests for your changes
5. Run the test suite (`python -m pytest tests/ -v`)
6. Commit with a clear message (`git commit -m "Add: semantic caching with embeddings"`)
7. Push to your fork (`git push origin feature/my-feature`)
8. Open a Pull Request

### Commit messages

Use prefixes to keep history readable:

- `Add:` new feature
- `Fix:` bug fix
- `Refactor:` code restructuring without behavior change
- `Docs:` documentation only
- `Test:` adding or updating tests
- `Chore:` build, CI, dependencies

### Code style

- Python 3.10+ (use type hints)
- Max line length: 100 characters (soft limit)
- Use `logging` instead of `print` in library code
- Docstrings for public functions and classes
- No external dependencies unless justified — the core should work with just `requests`, `pyyaml`, `fastapi`, `uvicorn`

### Adding a new provider backend

1. Create a class in `app/backends.py` extending `LLMBackend`
2. Implement `call()` and `call_stream()`
3. Add it to the `cls_map` in `get_backend()`
4. Add env var to `app/credentials.py` → `ENV_KEY_MAP`
5. Add a model entry in `conf/router.yaml` with `provider: your_provider`
6. Add tests

### Adding a new task type

1. Add exemplars to `_build_centroids()` in `app/classifier.py`
2. Add keyword rules to `_KEYWORD_RULES`
3. Add a chain in `conf/router.yaml` → `task_chains`
4. Add classifier tests

## Areas where help is most welcome

- **Semantic caching** — replace exact-match hash with embedding similarity
- **Success rate tracking** — measure actual response quality per model per task
- **Web dashboard** — minimal HTML/JS page with budget charts and latency graphs
- **Rate limit awareness** — proactive routing around providers approaching RPM limits
- **New provider backends** — Azure OpenAI, AWS Bedrock, Together AI, Groq direct
- **Localization** — classifier keywords for more languages beyond EN/RU

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
