"""Layer 2: Lightweight zero-shot task classifier using sentence-transformers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

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


def _embed(text: str) -> np.ndarray:
    """Compute embedding for text using the lazy-loaded model."""
    model = _get_model()
    # SentenceTransformer.encode returns a numpy array by default
    embeddings = model.encode([text], convert_to_tensor=False)
    return embeddings[0]


# Cached category embeddings as a normalized matrix for batch computation
_category_matrix: np.ndarray | None = None


def _get_category_matrix() -> np.ndarray:
    """Return a normalized matrix of category embeddings (num_categories, dim)."""
    global _category_matrix
    if _category_matrix is None:
        # Compute embeddings for all categories
        embs = [_embed(cat) for cat in TASK_CATEGORIES]
        matrix = np.vstack(embs)
        # Normalize each row for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        _category_matrix = matrix / norms
    return _category_matrix


def classify_task(
    task_description: str,
    conversation_history: list[dict[str, str]] | None = None,
    file_extension: str = "",
) -> str:
    """Classify the task into one of :data:`TASK_CATEGORIES`.

    Hybrid approach:
    1. Compute cosine similarity between the task text and each category
       name embedding (semantic score) using NumPy matrix operations.
    2. Boost the score with keyword overlap (lexical score).
    3. Return the highest-scoring category.
    """
    # Build a combined text from description + last 3 turns + file ext
    history_text = ""
    if conversation_history:
        last_3 = conversation_history[-3:]
        history_text = " ".join(m.get("content", "") for m in last_3)

    combined = f"{task_description} {history_text} {file_extension}"
    combined = combined.strip().lower()

    # Compute task embedding and normalize it
    task_emb = _embed(combined)
    task_norm = np.linalg.norm(task_emb)

    if task_norm == 0:
        semantic_sims = np.zeros(len(TASK_CATEGORIES))
    else:
        # Batch compute all semantic similarities via matrix-vector product
        # (num_categories, dim) @ (dim,) -> (num_categories,)
        semantic_sims = _get_category_matrix() @ (task_emb / task_norm)

    best_score = -1.0
    best_category = "code_generation"  # default

    for i, cat in enumerate(TASK_CATEGORIES):
        semantic_sim = float(semantic_sims[i])

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
