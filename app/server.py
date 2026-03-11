"""
RouteIQ — OpenAI-compatible HTTP proxy server.

Drop-in replacement: point any tool at http://localhost:8000/v1
Works with Cursor, Claude Code, Continue, OpenClaw, or any OpenAI SDK client.

Usage:
    python -m app.server
    # or
    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Load .env if present
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from .router import Router, RouterRequest, MODE_SETTINGS

logger = logging.getLogger(__name__)

from app import __version__

app = FastAPI(
    title="RouteIQ",
    description="Smart LLM router — OpenAI-compatible proxy",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router instance
_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


# ── OpenAI-compatible endpoints ──


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Accepts standard OpenAI request format, routes to optimal model.

    Extra fields (optional):
        - mode: "easy" | "medium" | "hard" | "god"
        - task_type: force a specific task type
        - session_id: enable session persistence
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    stream = body.get("stream", False)
    mode = body.get("mode", "medium")
    task_type = body.get("task_type")
    session_id = body.get("session_id")
    max_tokens = body.get("max_tokens")
    temperature = body.get("temperature")
    profile = body.get("profile")          # routing profile: auto/eco/premium/free/reasoning
    model = body.get("model")              # explicit model or alias
    tools = body.get("tools")              # tool definitions (OpenAI format)
    tool_choice = body.get("tool_choice")  # tool_choice param

    router = _get_router()

    req = RouterRequest(
        messages=messages,
        task_type=task_type,
        mode=mode,
        profile=profile,
        model=model if model != "auto" else None,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        session_id=session_id,
        tools=tools,
        tool_choice=tool_choice,
    )

    if stream:
        return StreamingResponse(
            _stream_response(router, req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        resp = router.route(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Format as OpenAI response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": resp.content,
    }
    if resp.tool_calls:
        message["tool_calls"] = resp.tool_calls

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model_used,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if resp.tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": resp.tokens_in,
            "completion_tokens": resp.tokens_out,
            "total_tokens": resp.tokens_in + resp.tokens_out,
        },
        # RouteIQ extras
        "routeiq": {
            "task_type": resp.task_type,
            "cost_usd": resp.cost_usd,
            "latency_ms": round(resp.latency_ms, 1),
            "cached": resp.cached,
            "provider": resp.provider,
            "is_agentic": resp.is_agentic,
            "is_reasoning": resp.is_reasoning,
        },
    })


async def _stream_response(router: Router, req: RouterRequest):
    """SSE streaming generator."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    try:
        gen = router.route_stream(req)
        for chunk in gen:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Final chunk
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    except RuntimeError as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


# ── Model listing (for tool compatibility) ──


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    router = _get_router()
    policy = router._policy
    models_cfg = policy.raw_config.get("models", {})

    models = []
    for name, cfg in models_cfg.items():
        models.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "routeiq",
            "permission": [],
            "root": cfg.get("id", name),
            "parent": None,
        })

    # Also expose the special "auto" model
    models.insert(0, {
        "id": "auto",
        "object": "model",
        "created": 0,
        "owned_by": "routeiq",
        "permission": [],
        "root": "auto",
        "parent": None,
    })

    return {"object": "list", "data": models}


# ── RouteIQ-specific endpoints ──


@app.get("/v1/status")
async def status():
    """Full RouteIQ status: budget, cache, sessions."""
    router = _get_router()
    return router.full_status()


@app.get("/v1/budget")
async def budget():
    """Budget status only."""
    router = _get_router()
    return router.budget_status()


@app.get("/v1/cache")
async def cache():
    """Cache stats."""
    router = _get_router()
    return router.cache_stats()


@app.get("/v1/report")
async def report(days: int = 0, model: str | None = None):
    """Analytics report from logs."""
    from .analytics import generate_report
    return generate_report(days=days, model_filter=model)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "routeiq", "version": __version__}


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    config = {}
    try:
        import yaml
        with open("conf/router.yaml") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        pass

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    print(f"""
╔══════════════════════════════════════════════╗
║           RouteIQ v2.2 — LLM Router         ║
║                                              ║
║  OpenAI-compatible proxy on:                 ║
║    http://{host}:{port}/v1                      ║
║                                              ║
║  Point your tools here:                      ║
║    export OPENAI_BASE_URL=http://localhost:{port}/v1  ║
║                                              ║
║  Endpoints:                                  ║
║    POST /v1/chat/completions                 ║
║    GET  /v1/models                           ║
║    GET  /v1/status                           ║
║    GET  /v1/budget                           ║
║    GET  /health                              ║
╚══════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=host, port=port, log_level="info")
