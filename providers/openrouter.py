"""OpenRouter provider — gateway to dozens of models."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config.settings import get_openrouter_key
from providers.base import BaseProvider

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


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


class OpenRouterProvider(BaseProvider):
    name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_openrouter_key() or ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/switchboard-mcp",
            "X-Title": "Switchboard MCP",
        }

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                if stream:
                    return self._stream_response(client, payload)
                else:
                    resp = await client.post(
                        OPENROUTER_API_URL, headers=self._headers(), json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    text = message.get("content", "")
                    usage = data.get("usage", {})
                    return {
                        "text": text,
                        "model": model,
                        "provider": self.name,
                        "usage": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                    }
        except httpx.HTTPStatusError as e:
            msg = f"OpenRouter API error: {e.response.status_code}"
            raise RuntimeError(msg) from e
        except httpx.RequestError as e:
            msg = f"OpenRouter request failed: {_sanitize_error(_safe_error_str(e))}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"OpenRouter error: {_sanitize_error(_safe_error_str(e))}"
            raise RuntimeError(msg) from e

    async def _stream_response(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        async with client.stream(
            "POST", OPENROUTER_API_URL, headers=self._headers(), json=payload
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
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
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
                    "https://openrouter.ai/api/v1/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False
