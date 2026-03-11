"""
RouteIQ — Optional OpenTelemetry tracing.

Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

If not installed, all trace calls are no-ops.
Follows GenAI semantic conventions for LLM observability.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_AVAILABLE = None
_tracer = None


def _try_init() -> bool:
    """Lazily initialize OpenTelemetry. Returns True if available."""
    global _AVAILABLE, _tracer
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        from app import __version__
        resource = Resource.create({"service.name": "routeiq", "service.version": __version__})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("routeiq")
        _AVAILABLE = True
        logger.info("OpenTelemetry tracing enabled")
    except ImportError:
        _AVAILABLE = False
        logger.debug("OpenTelemetry not installed — tracing disabled")
    return _AVAILABLE


@contextmanager
def trace_route(
    task_type: str,
    model: str,
    provider: str,
    is_agentic: bool = False,
    is_reasoning: bool = False,
) -> Generator[dict, None, None]:
    """
    Context manager for tracing a route operation.

    Usage:
        with trace_route("code", "qwen_coder", "openrouter") as span_data:
            result = backend.call(...)
            span_data["tokens_in"] = result["usage"]["prompt_tokens"]
            span_data["tokens_out"] = result["usage"]["completion_tokens"]
            span_data["cost_usd"] = 0.001
    """
    span_data: dict[str, Any] = {}

    if not _try_init() or _tracer is None:
        yield span_data
        return

    from opentelemetry import trace

    with _tracer.start_as_current_span("routeiq.route") as span:
        span.set_attribute("gen_ai.system", "routeiq")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("routeiq.task_type", task_type)
        span.set_attribute("routeiq.provider", provider)
        span.set_attribute("routeiq.is_agentic", is_agentic)
        span.set_attribute("routeiq.is_reasoning", is_reasoning)

        try:
            yield span_data
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            if "tokens_in" in span_data:
                span.set_attribute("gen_ai.usage.prompt_tokens", span_data["tokens_in"])
            if "tokens_out" in span_data:
                span.set_attribute("gen_ai.usage.completion_tokens", span_data["tokens_out"])
            if "cost_usd" in span_data:
                span.set_attribute("routeiq.cost_usd", span_data["cost_usd"])
            if "latency_ms" in span_data:
                span.set_attribute("routeiq.latency_ms", span_data["latency_ms"])
            if "cached" in span_data:
                span.set_attribute("routeiq.cached", span_data["cached"])
