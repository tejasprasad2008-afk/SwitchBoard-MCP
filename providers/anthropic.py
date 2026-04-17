"""Anthropic provider (direct API)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config.settings import get_anthropic_key
from providers.base import BaseProvider

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _sanitize_error(error: str) -> str:
    """Mask sensitive tokens like API keys in error messages."""
    return re.sub(r"sk-[a-zA-Z0-9]{10,}", "sk-REDACTED", error)


def _safe_error_str(exc: Exception) -> str:
    """Safely convert an exception to string."""
    try:
        return str(exc)
    except Exception:
        try:
            return repr(exc)
        except Exception:
            return f"<{type(exc).__name__}>"


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_anthropic_key() or ""

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
                converted.append({"role": "user", "content": f"[System]\n{content}"})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": "user", "content": content})
        return converted

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

        try:
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
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Anthropic API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            msg = f"Anthropic request failed: {_sanitize_error(_safe_error_str(e))}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Anthropic error: {_sanitize_error(_safe_error_str(e))}"
            raise RuntimeError(msg) from e

    async def _stream_response(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        import json

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
                    data = json.loads(data_str)
                    delta = data.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

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
