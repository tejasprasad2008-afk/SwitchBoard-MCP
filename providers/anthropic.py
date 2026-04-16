"""Anthropic provider (direct API)."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

from config.settings import get_anthropic_key
from providers.base import BaseProvider

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_anthropic_key() or ""

    # ── Internal helpers ───────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Anthropic format."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                # Anthropic handles system prompt separately; we inline for simplicity
                converted.append({"role": "user", "content": f"[System]\n{content}"})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": "user", "content": content})
        return converted

    # ── Chat completion ────────────────────────────────────────────

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if stream:
            payload["stream"] = True

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return self._stream_response(client, payload)
            else:
                resp = await client.post(
                    ANTHROPIC_API_URL, headers=self._headers(), json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                if data.get("content"):
                    text = data["content"][0].get("text", "")
                usage = data.get("usage", {})
                return {
                    "text": text,
                    "model": model,
                    "provider": self.name,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    },
                }

    async def _stream_response(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        async with client.stream(
            "POST", ANTHROPIC_API_URL, headers=self._headers(), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    import json
                    data = json.loads(data_str)
                    delta = data.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    # ── Health check ───────────────────────────────────────────────

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION},
                )
                return resp.status_code == 200
        except Exception:
            return False
