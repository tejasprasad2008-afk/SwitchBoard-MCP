"""Fallback chain — ordered fallback list with health-aware selection."""

from __future__ import annotations

from typing import Any

from config import settings
from providers.health import ProviderHealthTracker

# Whether to require API keys when selecting models. Set False for dry-run/testing.
_require_api_keys = False


def set_require_api_keys(val: bool) -> None:
    global _require_api_keys
    _require_api_keys = val

# Default fallback priority (model ids as they appear in models.yaml)
DEFAULT_FALLBACK_CHAIN: list[str] = [
    # 1. Claude Sonnet via Anthropic direct
    "claude-sonnet-4-20250514",
    # 2. Claude Sonnet via OpenRouter
    "anthropic/claude-sonnet-4",
    # 3. DeepSeek V3
    "deepseek/deepseek-v3",
    # 4. GPT-4o
    "openai/gpt-4o",
    # 5. Gemini 1.5 Pro
    "google/gemini-1.5-pro",
    # 6. Qwen 2.5 Coder
    "qwen/qwen-2.5-coder-32b-instruct",
    # 7. LLaMA 3.3 70B (free)
    "meta-llama/llama-3.3-70b-instruct:free",
    # 8. Qwen 2.5 72B (free — final fallback)
    "qwen/qwen-2.5-72b-instruct:free",
]


class FallbackChain:
    """Manages the ordered fallback list.  Filters out degraded / blacklisted
    providers and returns the next available model."""

    def __init__(
        self,
        health_tracker: ProviderHealthTracker | None = None,
        custom_chain: list[str] | None = None,
    ) -> None:
        self._health = health_tracker
        self._chain = custom_chain or list(DEFAULT_FALLBACK_CHAIN)

    def get_next(
        self,
        skip_model: str | None = None,
        task_category: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the best available model, skipping *skip_model* (the one that
        just failed) and respecting provider health / blacklist."""

        blacklisted = set(settings.prefs.blacklist_providers)

        # If we have a task category, prefer models with that strength first
        if task_category:
            preferred = settings.get_models_by_strength(task_category)
            preferred_ids = {m["id"] for m in preferred}
            # Interleave preferred models into the chain
            ordered = [
                m for m in self._chain
                if m in preferred_ids and m != skip_model
            ]
            ordered += [
                m for m in self._chain
                if m not in preferred_ids and m != skip_model
            ]
        else:
            ordered = [m for m in self._chain if m != skip_model]

        for model_id in ordered:
            model_def = settings.get_model_by_id(model_id)
            if model_def is None:
                continue

            provider = model_def.get("provider", "")
            if provider in blacklisted:
                continue

            if self._health and self._health.is_degraded(provider):
                continue

            if self._health and self._health.is_rate_limited(provider):
                continue

            # Check API key availability (only enforced when _require_api_keys is True)
            if _require_api_keys:
                if provider == "anthropic" and not settings.get_anthropic_key():
                    continue
                if provider == "openrouter" and not settings.get_openrouter_key():
                    continue

            return model_def

        return None

    def get_all_available(self) -> list[dict[str, Any]]:
        """Return all models that are currently available (not degraded/blacklisted)."""
        result: list[dict[str, Any]] = []
        for model_id in self._chain:
            model_def = settings.get_model_by_id(model_id)
            if model_def is None:
                continue
            provider = model_def.get("provider", "")
            if provider in settings.prefs.blacklist_providers:
                continue
            if self._health and self._health.is_degraded(provider):
                continue
            result.append(model_def)
        return result

    @property
    def chain(self) -> list[str]:
        return list(self._chain)
