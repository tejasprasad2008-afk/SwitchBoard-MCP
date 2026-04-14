"""Layer 1: Rule-based routing engine."""

from __future__ import annotations

from typing import Any

from config import settings
from config.settings import RoutingPreferences


class RuleRoutingResult:
    """Result from rule engine — may be conclusive or inconclusive."""

    def __init__(
        self,
        model_id: str | None = None,
        reason: str = "",
        conclusive: bool = False,
    ) -> None:
        self.model_id = model_id
        self.reason = reason
        self.conclusive = conclusive

    @classmethod
    def inconclusive(cls) -> RuleRoutingResult:
        return cls(conclusive=False)

    @classmethod
    def select(cls, model_id: str, reason: str) -> RuleRoutingResult:
        return cls(model_id=model_id, reason=reason, conclusive=True)


# Keywords that map to "simple" tasks → cheap models
_SIMPLE_TASK_KEYWORDS = [
    "autocomplete", "explain", "rename", "reformat", "what does",
    "meaning of", "describe", "comment", "docstring",
]

# Keywords indicating need for speed / low latency
_LATENCY_SENSITIVE_KEYWORDS = [
    "autocomplete", "suggest", "complete this", "stream",
]


async def evaluate_rules(
    messages: list[dict[str, str]],
    task_hint: str | None = None,
    preferences: RoutingPreferences | None = None,
    context_size: int = 0,
) -> RuleRoutingResult:
    """Run the rule engine. Returns a conclusive result or inconclusive."""

    prefs = preferences or settings.prefs
    combined_text = (task_hint or "").lower()
    for msg in messages:
        combined_text += " " + msg.get("content", "").lower()

    # ── 1. Prefer cheap (global setting) ──────────────────────────
    if prefs.prefer_cheap:
        cheapest = _get_cheapest_model()
        if cheapest:
            return RuleRoutingResult.select(
                cheapest["id"], reason="user preference: cheapest model"
            )

    # ── 2. Prefer fast (global setting) ───────────────────────────
    if prefs.prefer_fast:
        fastest = _get_fastest_model()
        if fastest:
            return RuleRoutingResult.select(
                fastest["id"], reason="user preference: fastest model"
            )

    # ── 3. Large context → need big window ────────────────────────
    if context_size > 60_000:
        large_ctx = _get_large_context_model()
        if large_ctx:
            return RuleRoutingResult.select(
                large_ctx["id"],
                reason=f"large context ({context_size} tokens) → model with big window",
            )

    # ── 4. Simple / cheap tasks ───────────────────────────────────
    if _matches_any(combined_text, _SIMPLE_TASK_KEYWORDS):
        cheap = _get_cheapest_model()
        if cheap:
            return RuleRoutingResult.select(
                cheap["id"], reason="simple task → cheapest model"
            )

    # ── 5. Latency-sensitive → fast model ─────────────────────────
    if _matches_any(combined_text, _LATENCY_SENSITIVE_KEYWORDS):
        fast = _get_fastest_model()
        if fast:
            return RuleRoutingResult.select(
                fast["id"], reason="latency-sensitive → fast model"
            )

    # ── 6. Budget cap → filter expensive models ───────────────────
    if prefs.max_cost_per_request < 1.0:
        budget_safe = _get_budget_safe_model(prefs.max_cost_per_request)
        if budget_safe:
            return RuleRoutingResult.select(
                budget_safe["id"],
                reason=f"budget cap (${prefs.max_cost_per_request}) → budget-safe model",
            )

    return RuleRoutingResult.inconclusive()


# ── Helpers ────────────────────────────────────────────────────────

def _matches_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def _get_cheapest_model() -> dict[str, Any] | None:
    candidates = [
        m for m in settings.load_models()
        if m.get("tier") != "paid" or m.get("cost_per_1k_tokens", 1) < 0.001
    ]
    if not candidates:
        candidates = settings.load_models()
    return min(candidates, key=lambda m: m.get("cost_per_1k_tokens", 0), default=None)


def _get_fastest_model() -> dict[str, Any] | None:
    fast_models = [m for m in settings.load_models() if m.get("speed") == "fast"]
    if not fast_models:
        return None
    return min(fast_models, key=lambda m: m.get("cost_per_1k_tokens", 0), default=None)


def _get_large_context_model() -> dict[str, Any] | None:
    large = [m for m in settings.load_models() if m.get("context_window", 0) > 200_000]
    if not large:
        large = [m for m in settings.load_models() if m.get("context_window", 0) > 100_000]
    return large[0] if large else None


def _get_budget_safe_model(max_cost: float) -> dict[str, Any] | None:
    """Pick the best model whose cost per 1K is within budget."""
    affordable = [
        m for m in settings.load_models()
        if m.get("cost_per_1k_tokens", 0) <= max_cost
    ]
    if not affordable:
        return None
    # Among affordable, pick the one with the most strengths
    return max(affordable, key=lambda m: len(m.get("strengths", [])), default=None)
