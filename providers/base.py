"""Abstract provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseProvider(ABC):
    """Every provider (Anthropic, OpenRouter, …) must implement these."""

    name: str = "base"

    # ── Chat completion ────────────────────────────────────────────

    @abstractmethod
    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """
        Return a dict with at least:
            {"text": str, "model": str, "usage": {"input_tokens": int, "output_tokens": int}}
        If *stream* is True, return an async iterator yielding text chunks.
        """

    # ── Health / availability ──────────────────────────────────────

    @abstractmethod
    async def health_check(self) -> bool:
        """Quick probe to check if the provider is reachable."""
