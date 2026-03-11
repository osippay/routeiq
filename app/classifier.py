"""
RouteIQ — Hybrid task classifier with post-classification modifiers.

Pipeline:
1. Classify prompt into 8 task types (embeddings or weighted keywords)
2. Detect agentic requests (tool use, system prompts) → force complex model
3. Detect reasoning requests (CoT markers) → force reasoning model

Supports: text, code, image, audio, vision, think, strategy, summarize
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

KNOWN_TYPES = frozenset({"text", "code", "image", "audio", "vision", "think", "strategy", "summarize"})

# ── Embedding-based classifier (loaded lazily) ─────────────────────

_embed_model = None
_centroids: dict[str, list[float]] = {}
_EMBED_AVAILABLE = None


def _try_load_embeddings() -> bool:
    global _embed_model, _EMBED_AVAILABLE
    if _EMBED_AVAILABLE is not None:
        return _EMBED_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _build_centroids()
        _EMBED_AVAILABLE = True
        logger.info("Classifier: using sentence embeddings (all-MiniLM-L6-v2)")
    except ImportError:
        _EMBED_AVAILABLE = False
        logger.info("Classifier: sentence-transformers not found, using keyword scoring")
    return _EMBED_AVAILABLE


def _build_centroids() -> None:
    global _centroids
    if _embed_model is None:
        return

    exemplars = {
        "code": [
            "write a python function", "fix the bug in this code",
            "refactor this class", "implement binary search",
            "напиши функцию на python", "исправь баг в коде",
            "debug this error", "create a REST API endpoint",
        ],
        "image": [
            "generate an image of a sunset", "draw a logo",
            "create a picture of a cat", "сгенерируй картинку",
            "нарисуй иллюстрацию", "make an illustration",
        ],
        "audio": [
            "transcribe this audio", "convert text to speech",
            "транскрибируй аудио", "озвучь этот текст",
            "generate voice from text", "speech recognition",
        ],
        "vision": [
            "what is in this image", "describe this photo",
            "analyze this screenshot", "что на этой картинке",
            "опиши что на фото", "read text from this image",
        ],
        "think": [
            "solve this complex math problem", "reason step by step",
            "think through this carefully", "подумай над этой задачей",
            "complex logical reasoning", "prove this theorem",
        ],
        "strategy": [
            "write a business plan", "create a marketing strategy",
            "plan the product roadmap", "составь бизнес-план",
            "стратегия развития проекта", "competitive analysis",
        ],
        "summarize": [
            "summarize this document", "give me a short summary",
            "shorten this text", "суммаризируй текст",
            "сделай краткое содержание", "TL;DR of this article",
        ],
        "text": [
            "write a blog post", "explain how photosynthesis works",
            "translate this to Spanish", "напиши пост",
            "объясни простыми словами", "rewrite this paragraph",
            "answer my question", "hello how are you",
        ],
    }

    for task_type, sentences in exemplars.items():
        embeddings = _embed_model.encode(sentences)
        centroid = embeddings.mean(axis=0).tolist()
        _centroids[task_type] = centroid


def _classify_embeddings(text: str) -> tuple[str, float]:
    import numpy as np
    prompt_emb = _embed_model.encode([text])[0]
    best_type = "text"
    best_sim = -1.0
    for task_type, centroid in _centroids.items():
        centroid_arr = np.array(centroid)
        sim = float(np.dot(prompt_emb, centroid_arr) / (
            np.linalg.norm(prompt_emb) * np.linalg.norm(centroid_arr) + 1e-9
        ))
        if sim > best_sim:
            best_sim = sim
            best_type = task_type
    return best_type, best_sim


# ── Keyword-based classifier (weighted scoring) ───────────────────

_KEYWORD_RULES: list[tuple[str, list[tuple[str, float]]]] = [
    ("code", [
        ("напиши код", 1.0), ("напиши функцию", 1.0), ("напиши скрипт", 0.9),
        ("напиши программу", 0.9), ("fix bug", 0.9), ("fix the bug", 1.0),
        ("исправь баг", 1.0), ("исправь ошибку", 0.8),
        ("write code", 1.0), ("write a function", 1.0),
        ("python function", 0.9), ("write a script", 0.9),
        ("implement", 0.6), ("реализуй", 0.7),
        ("debug", 0.7), ("дебаг", 0.7),
        ("refactor", 0.8), ("рефактор", 0.8),
        ("compile", 0.6), ("api endpoint", 0.8),
        ("git commit", 0.7), ("pull request", 0.6),
        ("```", 0.9),
        ("function", 0.5), ("функцию", 0.6), ("функция", 0.5),
        ("class ", 0.5), ("def ", 0.7), ("import ", 0.4),
        ("algorithm", 0.6), ("алгоритм", 0.6),
        ("variable", 0.4), ("переменн", 0.4),
        # File extensions → strong code signal
        (".py", 0.8), (".js", 0.8), (".ts", 0.8), (".tsx", 0.7),
        (".jsx", 0.7), (".go", 0.7), (".rs", 0.7), (".java", 0.7),
        (".cpp", 0.7), (".c ", 0.5), (".rb", 0.7), (".php", 0.7),
        # Error messages → code context
        ("error:", 0.6), ("typeerror", 0.8), ("syntaxerror", 0.8),
        ("referenceerror", 0.8), ("nameerror", 0.8), ("valueerror", 0.7),
        ("traceback", 0.8), ("exception", 0.5), ("stack trace", 0.8),
        ("cannot read property", 0.8), ("undefined is not", 0.8),
        ("null pointer", 0.8), ("segfault", 0.9),
        # Dev workflow
        ("component", 0.5), ("responsive", 0.4), ("endpoint", 0.6),
        ("database", 0.4), ("query", 0.4), ("migration", 0.5),
        ("deploy", 0.4), ("docker", 0.5), ("kubernetes", 0.5),
        ("test", 0.3), ("lint", 0.6), ("format", 0.3),
        ("npm", 0.6), ("pip", 0.5), ("cargo", 0.6),
        ("fix", 0.4), ("patch", 0.5), ("почини", 0.7),
    ]),
    ("image", [
        ("сгенерируй картинку", 1.0), ("generate image", 1.0),
        ("generate picture", 1.0), ("generate a picture", 1.0),
        ("нарисуй", 0.9), ("draw me", 0.9), ("draw a", 0.9),
        ("create an image", 1.0), ("create image", 1.0),
        ("dall-e", 0.9), ("dalle", 0.9),
    ]),
    ("audio", [
        ("транскрибируй", 1.0), ("transcribe", 0.9),
        ("text to speech", 1.0), ("tts", 0.7),
        ("speech to text", 1.0), ("voice", 0.5),
        ("аудио", 0.6), ("audio file", 0.8),
        ("озвучь", 0.9), ("whisper", 0.7),
    ]),
    ("vision", [
        ("что на этой картинке", 1.0), ("что на картинке", 1.0),
        ("что на этой фотографии", 1.0), ("что на фото", 0.9),
        ("посмотри на фото", 1.0), ("опиши фото", 1.0),
        ("опиши картинку", 1.0), ("analyze image", 0.9),
        ("analyze photo", 0.9), ("describe this image", 1.0),
        ("describe this photo", 1.0), ("what is in this image", 1.0),
        ("what's in this picture", 1.0), ("распознай", 0.7),
    ]),
    ("think", [
        ("подумай", 0.8), ("продумай", 0.8), ("думай", 0.6),
        ("complex reasoning", 1.0), ("сложное рассуждение", 1.0),
        ("reasoning", 0.6), ("step by step", 0.7), ("шаг за шагом", 0.7),
        ("посчитай", 0.5), ("prove", 0.6), ("докажи", 0.7),
        ("chain of thought", 0.9),
    ]),
    ("strategy", [
        ("стратегия", 0.8), ("бизнес план", 1.0), ("бизнес-план", 1.0),
        ("business plan", 1.0), ("roadmap", 0.8),
        ("план действий", 0.9), ("strategy", 0.7),
        ("go-to-market", 0.9), ("competitive analysis", 0.9),
        ("конкурентный анализ", 0.9), ("market analysis", 0.8),
    ]),
    ("summarize", [
        ("суммаризируй", 1.0), ("сделай кратко", 0.9),
        ("сократи", 0.7), ("краткое содержание", 1.0),
        ("summarize", 0.9), ("summary", 0.7),
        ("shorten", 0.6), ("tldr", 0.9), ("tl;dr", 0.9),
        ("in brief", 0.6), ("key points", 0.6),
    ]),
]


def _classify_keywords(text: str) -> tuple[str, float]:
    text_lower = text.lower().strip()
    scores: dict[str, float] = {}
    for task_type, keywords in _KEYWORD_RULES:
        total = 0.0
        for keyword, weight in keywords:
            if keyword in text_lower:
                total += weight
            else:
                try:
                    if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", text_lower, re.I | re.UNICODE):
                        total += weight * 0.8
                except re.error:
                    pass
        if total > 0:
            scores[task_type] = total
    if not scores:
        return "text", 0.0
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    confidence = min(scores[best_type] / 1.5, 1.0)
    return best_type, confidence


# ── Post-classification modifiers ──────────────────────────────────

# Reasoning markers — if 2+ found, task is reasoning-heavy
_REASONING_MARKERS = [
    "step by step", "think carefully", "reason through",
    "chain of thought", "let's think", "analyze this",
    "work through", "prove that", "show your work",
    "break it down", "рассуди", "порассуждай", "пошагово",
    "давай подумаем", "объясни логику", "докажи",
    "mathematical proof", "logical deduction",
    "evaluate the trade-offs", "compare and contrast",
    "what are the implications", "pros and cons",
]


def detect_reasoning(messages: list[dict]) -> bool:
    """
    Detect if the conversation requires chain-of-thought / deep reasoning.
    Returns True if 2+ reasoning markers found in the last user message.
    """
    text = _extract_last_user_text(messages).lower()
    count = sum(1 for marker in _REASONING_MARKERS if marker in text)
    return count >= 2


def detect_agentic(messages: list[dict], tools: list | None = None, tool_choice: str | None = None) -> bool:
    """
    Detect if this is an agentic request (tool use, multi-step agent workflow).
    Agentic requests MUST go to a capable model — cheap models break tool use.

    Checks:
    1. tools / tool_choice present in request
    2. Any message with role="tool" (tool result)
    3. System prompt contains agentic patterns (agent, tool, function calling)
    4. Assistant messages contain tool_calls
    """
    # 1. Explicit tool use in request
    if tools and len(tools) > 0:
        return True
    if tool_choice and tool_choice not in ("none", "auto"):
        return True

    for msg in messages:
        role = msg.get("role", "")

        # 2. Tool result message
        if role == "tool":
            return True

        # 3. System prompt with agentic patterns
        if role == "system":
            content = (msg.get("content") or "").lower()
            agentic_signals = [
                "you are an agent", "you are an ai agent",
                "you have access to tools", "available tools:",
                "function calling", "tool_use", "tool_calls",
                "you can use the following tools",
                "execute commands", "run code",
                "you are a coding assistant that can",
                "multi-step", "plan and execute",
            ]
            if sum(1 for s in agentic_signals if s in content) >= 1:
                return True

        # 4. Assistant with tool_calls
        if role == "assistant":
            if msg.get("tool_calls"):
                return True
            # OpenAI function_call format
            if msg.get("function_call"):
                return True

    return False


def _extract_last_user_text(messages: list[dict]) -> str:
    """Get text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle multi-part content (OpenAI vision format)
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(parts)
    return ""


# ── Public API ─────────────────────────────────────────────────────

def classify_task(prompt: str, hint: Optional[str] = None) -> str:
    """Classify a user prompt into one of 8 task types."""
    if hint and hint.lower().strip() in KNOWN_TYPES:
        return hint.lower().strip()

    combined = (prompt or "") + " " + (hint or "")
    combined = combined.strip()
    if not combined:
        return "text"

    if _try_load_embeddings():
        task_type, _ = _classify_embeddings(combined)
        return task_type

    task_type, _ = _classify_keywords(combined)
    return task_type


def classify_task_with_confidence(prompt: str, hint: Optional[str] = None) -> tuple[str, float]:
    """Like classify_task but also returns confidence score (0.0–1.0)."""
    if hint and hint.lower().strip() in KNOWN_TYPES:
        return hint.lower().strip(), 1.0

    combined = (prompt or "") + " " + (hint or "")
    combined = combined.strip()
    if not combined:
        return "text", 0.0

    if _try_load_embeddings():
        return _classify_embeddings(combined)
    return _classify_keywords(combined)


def classify_with_modifiers(
    messages: list[dict],
    hint: Optional[str] = None,
    tools: list | None = None,
    tool_choice: str | None = None,
) -> dict:
    """
    Full classification pipeline: task type + modifiers.

    Returns:
        {
            "task_type": "code",
            "confidence": 0.85,
            "is_agentic": True,
            "is_reasoning": False,
            "has_images": False,
            "force_complex": True,
        }
    """
    prompt_text = _extract_last_user_text(messages)
    task_type, confidence = classify_task_with_confidence(prompt_text, hint)

    is_agentic = detect_agentic(messages, tools, tool_choice)
    is_reasoning = detect_reasoning(messages)
    has_images = detect_images(messages)

    # Auto-escalate to vision if images present and task is generic text
    if has_images and task_type in ("text", "code", "summarize"):
        task_type = "vision"

    return {
        "task_type": task_type,
        "confidence": confidence,
        "is_agentic": is_agentic,
        "is_reasoning": is_reasoning,
        "has_images": has_images,
        "force_complex": is_agentic or is_reasoning,
    }


# ── Multimodal detection ──────────────────────────────────────────

def detect_images(messages: list[dict]) -> bool:
    """
    Detect if messages contain images (base64 or URL).

    OpenAI vision format:
        {"role": "user", "content": [
            {"type": "text", "text": "what's this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}

    Anthropic format:
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "data": "..."}}
        ]}
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            # OpenAI: {"type": "image_url", ...}
            if part_type == "image_url":
                return True
            # Anthropic: {"type": "image", ...}
            if part_type == "image":
                return True
    return False
