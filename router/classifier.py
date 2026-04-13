"""Layer 2: Lightweight zero-shot task classifier using sentence-transformers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Task categories the classifier recognizes
TASK_CATEGORIES = [
    "code_generation",
    "code_review",
    "debugging",
    "explanation",
    "architecture",
    "autocomplete",
    "security_audit",
]

# Category → keyword boost for hybrid scoring
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "code_generation": [
        "write", "create", "implement", "build", "function", "class",
        "generate", "code", "script", "endpoint", "api", "component",
    ],
    "code_review": [
        "review", "feedback", "improve", "refactor", "clean up",
        "better way", "best practice", "code quality", "smell",
    ],
    "debugging": [
        "bug", "error", "exception", "crash", "fix", "not working",
        "why does", "unexpected", "broken", "traceback", "stack trace",
        "debug", "issue", "problem",
    ],
    "explanation": [
        "explain", "what does", "how does", "meaning", "understand",
        "describe", "tell me about", "why", "what is",
    ],
    "architecture": [
        "architecture", "design", "pattern", "structure", "system",
        "scalable", "microservice", "database design", "api design",
        "how should i structure",
    ],
    "autocomplete": [
        "complete", "finish", "suggest", "autofill", "tab",
        "continue this", "what comes next", "def fib", "def fibonacci",
        "autocomplete", "auto-complete",
    ],
    "security_audit": [
        "security", "vulnerability", "audit", "safe", "exploit",
        "xss", "csrf", "injection", "auth", "permission", "cve",
        "traceTree", "trace tree",
    ],
}

_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model: Any = None


def _get_model():
    """Lazy-load the embedding model on first use."""
    global _embedding_model
    if _embedding_model is None:
        # Suppress download progress bars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer  # noqa: E402
        _embedding_model = SentenceTransformer(_MODEL_NAME)
    return _embedding_model


def _embed(text: str) -> list[float]:
    model = _get_model()
    embeddings = model.encode([text], convert_to_tensor=False)
    return embeddings[0].tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Cached category embeddings
_category_embeddings: dict[str, list[float]] = {}


def _get_category_embedding(category: str) -> list[float]:
    if category not in _category_embeddings:
        _category_embeddings[category] = _embed(category)
    return _category_embeddings[category]


def classify_task(
    task_description: str,
    conversation_history: list[dict[str, str]] | None = None,
    file_extension: str = "",
) -> str:
    """Classify the task into one of :data:`TASK_CATEGORIES`.

    Hybrid approach:
    1. Compute cosine similarity between the task text and each category
       name embedding (semantic score).
    2. Boost the score with keyword overlap (lexical score).
    3. Return the highest-scoring category.
    """
    # Build a combined text from description + last 3 turns + file ext
    history_text = ""
    if conversation_history:
        last_3 = conversation_history[-3:]
        history_text = " ".join(
            m.get("content", "") for m in last_3
        )

    combined = f"{task_description} {history_text} {file_extension}"
    combined = combined.strip().lower()

    task_emb = _embed(combined)

    best_score = -1.0
    best_category = "code_generation"  # default

    for cat in TASK_CATEGORIES:
        # Semantic similarity
        cat_emb = _get_category_embedding(cat)
        semantic_sim = _cosine_similarity(task_emb, cat_emb)

        # Keyword boost: fraction of category keywords found in text
        keywords = CATEGORY_KEYWORDS.get(cat, [])
        if keywords:
            kw_matches = sum(1 for kw in keywords if kw in combined)
            kw_boost = kw_matches / len(keywords)
        else:
            kw_boost = 0.0

        # Weighted combination: 55% semantic, 45% keyword
        score = 0.55 * semantic_sim + 0.45 * kw_boost

        if score > best_score:
            best_score = score
            best_category = cat

    return best_category


def get_preferred_models_for_task(task_category: str) -> list[dict]:
    """Return models sorted by suitability for *task_category*."""
    from config import settings

    models = settings.get_models_by_strength(task_category)
    return models
